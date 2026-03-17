"""CLI entry-point for the Cu catalyst AI baseline pipeline.

Default task (``task=baseline``) runs the full pipeline end-to-end:
  fetch → clean → featurize → train → explain → report

Individual stages can be re-run by passing ``task=<stage>``.
All output file paths are defined in ``configs/config.yaml`` under ``paths:``.

Real table sources (``data=real_table``) additionally write a review file
containing rows isolated by the cleaning governance layer.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cu_catalyst_ai.clean.deduplicate import drop_duplicates
from cu_catalyst_ai.clean.governance import split_good_review
from cu_catalyst_ai.clean.normalize_units import normalize_units
from cu_catalyst_ai.clean.provenance_validator import validate_provenance
from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.clean.target_validator import validate_target_definition
from cu_catalyst_ai.clean.validate_conditions import validate_required_columns, validate_rows
from cu_catalyst_ai.dataio.mp_fetch import fetch_data
from cu_catalyst_ai.explain.shap_runner import explain_model
from cu_catalyst_ai.features.basic_features import build_feature_table
from cu_catalyst_ai.features.structural_features import add_structural_ratios
from cu_catalyst_ai.models.train import train_model
from cu_catalyst_ai.schemas.catalyst import validate_schema_rows
from cu_catalyst_ai.utils.io import read_table, write_table
from cu_catalyst_ai.utils.logging_utils import get_logger
from cu_catalyst_ai.utils.seeds import set_global_seed
from cu_catalyst_ai.viz.learning_curve import save_learning_curve
from cu_catalyst_ai.viz.parity_plot import save_parity_plot
from cu_catalyst_ai.viz.report_bundle import write_report_bundle
from cu_catalyst_ai.viz.shap_plot import save_importance_plot

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_get(cfg: DictConfig, key: str, default: object | None = None) -> object | None:
    """Safely read an optional key from a DictConfig, returning *default* if absent."""
    try:
        val = OmegaConf.select(cfg, key)
        return val if val is not None else default
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------------------
# Individual stage handlers
# Each function performs exactly one stage and uses paths from cfg.paths.
# ---------------------------------------------------------------------------


def _run_fetch(cfg: DictConfig) -> None:
    source = str(cfg.data.source_name)

    if source == "table":
        fetch_data(
            source_name=source,
            output_path=str(
                cfg.data.get(
                    "raw_output", cfg.data.get("demo_output", "data/raw/real/cu_real_raw.parquet")
                )
            ),
            input_path=str(_cfg_get(cfg, "data.input_path") or ""),
            column_mapping=dict(_cfg_get(cfg, "data.column_mapping") or {}),
            defaults=dict(_cfg_get(cfg, "data.fill_defaults") or {}),
            target_definition=str(_cfg_get(cfg, "data.target_definition") or ""),
            raw_output=str(_cfg_get(cfg, "data.raw_output") or ""),
        )
    elif source == "cathub":
        query_filter_raw = _cfg_get(cfg, "data.query_filter")
        from omegaconf import OmegaConf  # noqa: PLC0415

        query_filter = (
            dict(OmegaConf.to_container(query_filter_raw, resolve=True))  # type: ignore[arg-type]
            if query_filter_raw is not None
            else {}
        )
        fetch_data(
            source_name="cathub",
            output_path=str(
                _cfg_get(cfg, "data.raw_output") or "data/raw/cathub/cu_cathub_raw.parquet"
            ),
            raw_output=str(_cfg_get(cfg, "data.raw_output") or ""),
            target_definition=str(
                _cfg_get(cfg, "data.target_definition") or "co_adsorption_energy_ev_v1"
            ),
            cathub_kwargs={
                "api_url": str(
                    _cfg_get(cfg, "data.api_url") or "https://api.catalysis-hub.org/graphql"
                ),
                "query_filter": query_filter,
            },
        )
    else:
        fetch_data(
            source_name=source,
            output_path=str(cfg.data.demo_output),
            n_samples=int(cfg.data.n_samples),
            seed=int(cfg.project.seed),
        )


def _run_clean(cfg: DictConfig) -> None:
    source = str(cfg.data.source_name)

    # Determine input/output paths for this stage.
    if source in ("table", "cathub"):
        raw_path = str(_cfg_get(cfg, "data.raw_output") or cfg.data.get("demo_output"))
        cleaned_path = str(cfg.data.cleaned_output)
        review_path = str(
            _cfg_get(cfg, "data.review_output") or "data/interim/cu_real_review.parquet"
        )
    else:
        raw_path = str(cfg.data.demo_output)
        cleaned_path = str(cfg.data.cleaned_output)
        review_path = None  # demo source won't produce a separate review file normally

    raw_df = read_table(raw_path)

    # --- Layer 1: Structural column check (raises on failure) ---------------
    raw_df = validate_required_columns(raw_df)

    # --- Layer 2: Unit normalisation + flag unknown units -------------------
    # Use unit conversions from target config when available (single source of truth).
    unit_conversions_raw = _cfg_get(cfg, "target.supported_unit_conversions")
    unit_conversions = dict(unit_conversions_raw) if unit_conversions_raw is not None else None
    raw_df = normalize_units(raw_df, unit_conversions=unit_conversions)

    # --- Layer 3: Row-level governance (flags, does not raise) --------------
    if source in ("table", "cathub"):
        target_def_name = str(_cfg_get(cfg, "data.target_definition") or "")
        required_ads = str(_cfg_get(cfg, "target.required_adsorbate") or "CO")
        raw_df = validate_target_definition(
            raw_df, target_def_name, required_adsorbate=required_ads
        )
        raw_df = validate_provenance(raw_df)
        raw_df = validate_rows(
            raw_df,
            adsorption_energy_abs_max=float(
                _cfg_get(cfg, "target.review_bounds.adsorption_energy_abs_max") or 10.0
            ),
            surface_energy_min=float(
                _cfg_get(cfg, "target.review_bounds.surface_energy_min") or 0.0
            ),
            electronegativity_min=float(
                _cfg_get(cfg, "target.review_bounds.electronegativity_min") or 0.0
            ),
            electronegativity_max=float(
                _cfg_get(cfg, "target.review_bounds.electronegativity_max") or 4.0
            ),
        )
        # --- Layer 4: Pydantic schema validation (flags, does not raise) ----
        raw_df = validate_schema_rows(raw_df)
        clean_df, review_df = split_good_review(raw_df)

        # Write review file regardless of size
        if review_path:
            write_table(review_df, review_path)
            logger.info("Review file: %d rows → %s", len(review_df), review_path)

        logger.info("Governance result: accepted=%d reviewed=%d", len(clean_df), len(review_df))

        if clean_df.empty:
            raise RuntimeError(
                f"Cleaning produced zero accepted rows (all {len(review_df)} rows were isolated). "
                "Check the review file for details: " + (review_path or "<unknown>")
            )
    else:
        clean_df = raw_df

    # --- Deduplication + split assignment ----------------------------------
    clean_df = drop_duplicates(clean_df)
    clean_df = assign_splits(clean_df, seed=int(cfg.project.seed))

    write_table(clean_df, cleaned_path)
    logger.info("Saved cleaned data (%d rows) to %s", len(clean_df), cleaned_path)


def _run_featurize(cfg: DictConfig) -> None:
    clean_df = read_table(cfg.data.cleaned_output)
    enriched_df = add_structural_ratios(clean_df)
    feature_df = build_feature_table(
        enriched_df,
        use_columns=list(cfg.features.use_columns) + ["coordination_to_distance"],
        categorical_columns=list(cfg.features.categorical_columns),
    )
    write_table(feature_df, cfg.data.processed_output)
    logger.info("Saved features to %s", cfg.data.processed_output)


def _run_train(cfg: DictConfig) -> dict:
    df = read_table(cfg.data.processed_output)
    result = train_model(
        df=df,
        model_name=str(cfg.model.name),
        random_state=int(cfg.model.random_state),
        params=OmegaConf.to_container(cfg.model.params, resolve=True) or {},
        target_col=str(cfg.project.target_col),
        n_splits=int(cfg.cv.n_splits),
        shuffle=bool(cfg.cv.shuffle),
        cv_random_state=int(cfg.cv.random_state),
        metrics_output=str(cfg.paths.metrics_output),
        model_output=str(cfg.paths.model_output),
        predictions_output=str(cfg.paths.predictions_output),
    )
    save_parity_plot(result["pred_df"], str(cfg.project.target_col), str(cfg.paths.parity_output))
    save_learning_curve(
        result["model"],
        df,
        str(cfg.project.target_col),
        str(cfg.paths.learning_curve_output),
        int(cfg.cv.n_splits),
    )
    logger.info("Training artifacts saved")
    return result


def _run_explain(cfg: DictConfig) -> None:
    import joblib  # noqa: PLC0415

    df = read_table(cfg.data.processed_output)
    bundle = joblib.load(Path(str(cfg.paths.model_output)))
    explanation_df = explain_model(
        model=bundle["model"],
        df=df,
        target_col=str(cfg.project.target_col),
        output_path=str(cfg.paths.explanation_output),
        random_state=int(cfg.project.seed),
    )
    save_importance_plot(explanation_df, str(cfg.paths.importance_plot_output))
    logger.info("Explanation artifacts saved")


def _run_report(cfg: DictConfig) -> None:
    write_report_bundle(
        model_name=str(cfg.model.name),
        metrics_path=str(cfg.paths.metrics_output),
        explanation_path=str(cfg.paths.explanation_output),
        output_path=str(cfg.paths.report_output),
    )
    logger.info("Summary report saved")


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one pipeline stage or the full baseline pipeline.

    The default task is ``baseline``, which chains all stages in order.
    Individual stages can be re-run via ``task=fetch``, ``task=clean``, etc.
    """
    set_global_seed(int(cfg.project.seed))
    task = str(cfg.task)
    logger.info("Running task=%s with model=%s", task, cfg.model.name)
    logger.debug(OmegaConf.to_yaml(cfg))

    if task == "baseline":
        logger.info("=== Stage 1/6: fetch ===")
        _run_fetch(cfg)
        logger.info("=== Stage 2/6: clean ===")
        _run_clean(cfg)
        logger.info("=== Stage 3/6: featurize ===")
        _run_featurize(cfg)
        logger.info("=== Stage 4/6: train ===")
        _run_train(cfg)
        logger.info("=== Stage 5/6: explain ===")
        _run_explain(cfg)
        logger.info("=== Stage 6/6: report ===")
        _run_report(cfg)
        logger.info("Baseline pipeline complete.")
        return

    if task == "fetch":
        _run_fetch(cfg)
        return

    if task == "clean":
        _run_clean(cfg)
        return

    if task == "featurize":
        _run_featurize(cfg)
        return

    if task == "train":
        _run_train(cfg)
        return

    if task == "explain":
        _run_explain(cfg)
        return

    if task == "report":
        _run_report(cfg)
        return

    _valid = "baseline, fetch, clean, featurize, train, explain, report"
    raise ValueError(f"Unknown task: {task!r}. Valid tasks: {_valid}")


if __name__ == "__main__":
    main()
