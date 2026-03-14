"""Config smoke tests: YAML parsing and CLI import."""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def test_config_yaml_parses_and_has_expected_keys() -> None:
    """Ensure config.yaml can be loaded and key fields have correct defaults."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg_path = CONFIG_DIR / "config.yaml"
    # OmegaConf can load a single YAML without Hydra runtime.
    # The top-level config.yaml uses Hydra 'defaults:' which OmegaConf
    # won't resolve (that requires the Hydra compose API), so we check
    # only the keys that are explicitly present in config.yaml itself.
    cfg = OmegaConf.load(cfg_path)

    # Project keys
    assert cfg.project.target_col == "adsorption_energy"
    assert cfg.project.id_col == "catalyst_id"
    assert cfg.project.seed == 42

    # Top-level model override present in config.yaml
    assert cfg.model.name == "rf"

    # CV override present in config.yaml
    assert cfg.cv.n_splits == 5
    assert cfg.cv.shuffle is True

    # Output paths present
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
