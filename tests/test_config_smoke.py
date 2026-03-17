"""Config smoke tests: YAML parsing and CLI import."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_real_table_config_parses() -> None:
    """real_table.yaml must be parseable and contain required data source fields."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "data" / "real_table.yaml")
    assert cfg.source_name == "table"
    assert "raw_output" in cfg
    assert "cleaned_output" in cfg
    assert "processed_output" in cfg
    assert "review_output" in cfg
    assert "column_mapping" in cfg
    assert "fill_defaults" in cfg  # renamed from 'defaults' to avoid Hydra conflict
    assert "target_definition" in cfg


def test_co_adsorption_energy_target_config_parses() -> None:
    """co_adsorption_energy_ev_v1.yaml must be parseable and contain key registry fields."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "target" / "co_adsorption_energy_ev_v1.yaml")
    assert cfg.name == "co_adsorption_energy_ev_v1"
    assert cfg.column == "adsorption_energy"
    assert cfg.canonical_unit == "eV"
    assert cfg.required_adsorbate == "CO"
    assert "supported_unit_conversions" in cfg
    assert "review_bounds" in cfg


def test_real_table_fill_defaults_key_not_reserved() -> None:
    """real_table.yaml must NOT contain a 'defaults' key (Hydra-reserved).

    The column-fill defaults must live under 'fill_defaults' to avoid a
    Hydra ConfigKeyError during config-group composition.
    """
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "data" / "real_table.yaml")
    assert "defaults" not in cfg, (
        "real_table.yaml must not use the Hydra-reserved key 'defaults'. "
        "Rename it to 'fill_defaults'."
    )
    assert "fill_defaults" in cfg, "Expected 'fill_defaults' key in real_table.yaml"
    from omegaconf import OmegaConf  # noqa: PLC0415, F811

    fill_defaults_raw = OmegaConf.to_container(cfg.fill_defaults, throw_on_missing=False)
    assert "provenance" in fill_defaults_raw  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Hydra compose-level tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def _clear_hydra():
    """Ensure Hydra global state is clean before and after each compose test."""
    from hydra.core.global_hydra import GlobalHydra  # noqa: PLC0415

    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_hydra_compose_real_table_no_error(_clear_hydra) -> None:  # noqa: ANN001
    """Hydra must be able to compose config with data=real_table without error.

    Catches the Hydra 'defaults' key conflict that existed before the rename
    to 'fill_defaults'.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=real_table",
                "data.input_path=/tmp/dummy.csv",
                "data.fill_defaults.provenance=unit_test",
            ],
        )
    assert cfg.data.source_name == "table"
    assert cfg.data.input_path == "/tmp/dummy.csv"
    assert cfg.data.fill_defaults.provenance == "unit_test"


def test_hydra_compose_real_table_missing_input_path_is_mandatory(_clear_hydra) -> None:  # noqa: ANN001
    """data.input_path=??? must remain a mandatory field (OmegaConf raises on access)."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from omegaconf import MissingMandatoryValue  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["data=real_table", "data.fill_defaults.provenance=unit_test"],
        )
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.data.input_path


def test_hydra_compose_real_table_target_config_accessible(_clear_hydra) -> None:  # noqa: ANN001
    """Composed config must expose cfg.target.review_bounds and unit conversions."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=real_table",
                "data.input_path=/tmp/dummy.csv",
                "data.fill_defaults.provenance=unit_test",
            ],
        )
    assert "review_bounds" in cfg.target
    assert "supported_unit_conversions" in cfg.target
    assert cfg.target.review_bounds.adsorption_energy_abs_max == pytest.approx(10.0)


def test_hydra_compose_conflicting_group_raises_on_bad_key(_clear_hydra) -> None:  # noqa: ANN001
    """Composing with an unknown data group name must raise a Hydra error."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.errors import HydraException  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        with pytest.raises(HydraException):
            compose(config_name="config", overrides=["data=nonexistent_source"])


# ---------------------------------------------------------------------------
# New cathub config smoke tests
# ---------------------------------------------------------------------------


def test_cathub_data_config_parses() -> None:
    """configs/data/cathub.yaml must be parseable with required fields."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "data" / "cathub.yaml")
    assert cfg.source_name == "cathub"
    assert "raw_output" in cfg
    assert "cleaned_output" in cfg
    assert "processed_output" in cfg
    assert "review_output" in cfg
    assert "api_url" in cfg
    assert "query_filter" in cfg
    assert "target_definition" in cfg


def test_cathub_minimal_feature_config_parses() -> None:
    """configs/features/cathub_minimal.yaml must list the four allowed columns."""
    from omegaconf import OmegaConf  # noqa: PLC0415

    cfg = OmegaConf.load(CONFIG_DIR / "features" / "cathub_minimal.yaml")
    use_cols = OmegaConf.to_container(cfg.use_columns, resolve=True)
    assert isinstance(use_cols, list)
    assert "coordination_number" in use_cols
    assert "avg_neighbor_distance" in use_cols
    assert "electronegativity" in use_cols
    assert "facet" in use_cols
    # These must NOT be present — they are unavailable from the API.
    assert "d_band_center" not in use_cols
    assert "surface_energy" not in use_cols


def test_hydra_compose_cathub_no_error(_clear_hydra) -> None:  # noqa: ANN001
    """Hydra can compose config with data=cathub features=cathub_minimal."""
    from hydra import compose, initialize_config_dir  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR.absolute()), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["data=cathub", "features=cathub_minimal"],
        )
    assert cfg.data.source_name == "cathub"
    assert "raw_output" in cfg.data
    assert "api_url" in cfg.data
