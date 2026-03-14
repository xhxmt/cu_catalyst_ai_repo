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


def _metrics_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.tables_dir}/{cfg.model.name}_metrics.csv"


def _predictions_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.tables_dir}/{cfg.model.name}_predictions.csv"


def _model_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.models_dir}/{cfg.model.name}.joblib"


def _explanation_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.tables_dir}/{cfg.model.name}_feature_importance.csv"


def _parity_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.figures_dir}/{cfg.model.name}_parity.png"


def _learning_curve_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.figures_dir}/{cfg.model.name}_learning_curve.png"


def _importance_plot_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.figures_dir}/{cfg.model.name}_importance.png"


def _report_path(cfg: DictConfig) -> str:
    return f"{cfg.paths.reports_dir}/{cfg.model.name}_summary.md"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.project.seed))
    task = str(cfg.task)
    logger.info("Running task=%s with model=%s", task, cfg.model.name)
    logger.debug(OmegaConf.to_yaml(cfg))

    if task == "fetch":
        fetch_data(
            source_name=str(cfg.data.source_name),
            output_path=str(cfg.data.demo_output),
            n_samples=int(cfg.data.n_samples),
            seed=int(cfg.project.seed),
        )
        return

    if task == "clean":
        raw_df = read_table(cfg.data.demo_output)
        clean_df = validate_required_columns(raw_df)
        clean_df = normalize_units(clean_df)
        clean_df = drop_duplicates(clean_df)
        clean_df = assign_splits(clean_df, seed=int(cfg.project.seed))
        write_table(clean_df, cfg.data.cleaned_output)
        logger.info("Saved cleaned data to %s", cfg.data.cleaned_output)
        return

    if task == "featurize":
        clean_df = read_table(cfg.data.cleaned_output)
        enriched_df = add_structural_ratios(clean_df)
        feature_df = build_feature_table(
            enriched_df,
            use_columns=list(cfg.features.use_columns) + ["coordination_to_distance"],
            categorical_columns=list(cfg.features.categorical_columns),
        )
        write_table(feature_df, cfg.data.processed_output)
        logger.info("Saved features to %s", cfg.data.processed_output)
        return

    if task == "train":
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
            metrics_output=_metrics_path(cfg),
            model_output=_model_path(cfg),
            predictions_output=_predictions_path(cfg),
        )
        save_parity_plot(result["pred_df"], str(cfg.project.target_col), _parity_path(cfg))
        save_learning_curve(
            result["model"],
            df,
            str(cfg.project.target_col),
            _learning_curve_path(cfg),
            int(cfg.cv.n_splits),
        )
        logger.info("Training artifacts saved")
        return

    if task == "explain":
        import joblib

        df = read_table(cfg.data.processed_output)
        bundle = joblib.load(Path(_model_path(cfg)))
        explanation_df = explain_model(
            model=bundle["model"],
            df=df,
            target_col=str(cfg.project.target_col),
            output_path=_explanation_path(cfg),
            random_state=int(cfg.project.seed),
        )
        save_importance_plot(explanation_df, _importance_plot_path(cfg))
        logger.info("Explanation artifacts saved")
        return

    if task == "report":
        write_report_bundle(
            model_name=str(cfg.model.name),
            metrics_path=_metrics_path(cfg),
            explanation_path=_explanation_path(cfg),
            output_path=_report_path(cfg),
        )
        logger.info("Summary report saved")
        return

    raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
