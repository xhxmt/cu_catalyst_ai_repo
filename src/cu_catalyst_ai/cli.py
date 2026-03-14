"""CLI entry-point for the Cu catalyst AI baseline pipeline.

Default task (``task=baseline``) runs the full pipeline end-to-end:
  fetch → clean → featurize → train → explain → report

Individual stages can be re-run by passing ``task=<stage>``.
All output file paths are defined in ``configs/config.yaml`` under ``paths:``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from cu_catalyst_ai.clean.deduplicate import drop_duplicates
from cu_catalyst_ai.clean.normalize_units import normalize_units
from cu_catalyst_ai.clean.split_registry import assign_splits
from cu_catalyst_ai.clean.validate_conditions import validate_required_columns
from cu_catalyst_ai.dataio.mp_fetch import fetch_data
from cu_catalyst_ai.explain.shap_runner import explain_model
from cu_catalyst_ai.features.basic_features import build_feature_table
from cu_catalyst_ai.features.structural_features import add_structural_ratios
from cu_catalyst_ai.models.train import train_model
from cu_catalyst_ai.utils.io import read_table, write_table
from cu_catalyst_ai.utils.logging_utils import get_logger
from cu_catalyst_ai.utils.seeds import set_global_seed
from cu_catalyst_ai.viz.learning_curve import save_learning_curve
from cu_catalyst_ai.viz.parity_plot import save_parity_plot
from cu_catalyst_ai.viz.report_bundle import write_report_bundle
from cu_catalyst_ai.viz.shap_plot import save_importance_plot

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual stage handlers
# Each function performs exactly one stage and uses paths from cfg.paths.
# ---------------------------------------------------------------------------


def _run_fetch(cfg: DictConfig) -> None:
    fetch_data(
        source_name=str(cfg.data.source_name),
        output_path=str(cfg.data.demo_output),
        n_samples=int(cfg.data.n_samples),
        seed=int(cfg.project.seed),
    )


def _run_clean(cfg: DictConfig) -> None:
    raw_df = read_table(cfg.data.demo_output)
    clean_df = validate_required_columns(raw_df)
    clean_df = normalize_units(clean_df)
    clean_df = drop_duplicates(clean_df)
    clean_df = assign_splits(clean_df, seed=int(cfg.project.seed))
    write_table(clean_df, cfg.data.cleaned_output)
    logger.info("Saved cleaned data to %s", cfg.data.cleaned_output)


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
