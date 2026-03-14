"""Config smoke tests: YAML parsing and CLI import."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "configs"

# All output-file path keys that must be present in the composed config.
_OUTPUT_PATH_KEYS = (
    "metrics_output",
    "predictions_output",
    "model_output",
    "explanation_output",
    "parity_output",
    "learning_curve_output",
    "importance_plot_output",
    "report_output",
)


def test_config_yaml_default_task_is_baseline() -> None:
    """Default task must be 'baseline' so bare CLI runs the full pipeline."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")
    assert str(cfg.task) == "baseline", (
        f"Default task must be 'baseline', got '{cfg.task}'. "
        "Update configs/config.yaml line 'task: ...' to 'task: baseline'."
    )


def test_config_yaml_output_paths_fully_declared() -> None:
    """All final output-file paths must be declared under paths: in config.yaml."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")
    for key in _OUTPUT_PATH_KEYS:
        assert key in cfg.paths, f"Missing paths.{key} in config.yaml"


def test_config_yaml_parses_and_has_expected_keys() -> None:
    """Ensure config.yaml project/model/cv keys have correct defaults."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "config.yaml")

    assert cfg.project.target_col == "adsorption_energy"
    assert cfg.project.id_col == "catalyst_id"
    assert cfg.project.seed == 42
    assert cfg.model.name == "rf"
    assert cfg.cv.n_splits == 5
    assert cfg.cv.shuffle is True

    # Directory-level paths present
    for key in ("raw_dir", "interim_dir", "processed_dir", "tables_dir", "figures_dir"):
        assert key in cfg.paths, f"Missing path key: {key}"


def test_baseline_feature_config_parses() -> None:
    """Ensure features/baseline.yaml can be parsed and lists expected columns."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    # Hydra group YAMLs are flat (no group wrapper) when read standalone.
    cfg = OmegaConf.load(CONFIG_DIR / "features" / "baseline.yaml")
    use_cols = OmegaConf.to_container(cfg.use_columns, resolve=True)
    assert isinstance(use_cols, list)
    assert "coordination_number" in use_cols
    assert "d_band_center" in use_cols


def test_rf_model_config_parses() -> None:
    """Ensure model/rf.yaml is parseable and specifies model name 'rf'."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "model" / "rf.yaml")
    assert cfg.name == "rf"


def test_kfold5_cv_config_parses() -> None:
    """Ensure cv/kfold5.yaml is parseable and n_splits == 5."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "cv" / "kfold5.yaml")
    assert cfg.n_splits == 5


def test_cli_main_is_importable_and_callable() -> None:
    """Ensure the CLI entry-point can be imported without errors."""
    from cu_catalyst_ai.cli import main  # noqa: PLC0415

    assert callable(main)
