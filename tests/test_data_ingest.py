"""Tests for the table data source ingestion path in fetch_data()."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cu_catalyst_ai.dataio.mp_fetch import fetch_data

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_standard_csv(tmp_path: Path) -> Path:
    """Write a minimal CSV with standard column names."""
    data = {
        "catalyst_id": ["cu_0001", "cu_0002"],
        "element": ["Cu", "Cu"],
        "facet": ["111", "100"],
        "adsorbate": ["CO", "CO"],
        "coordination_number": [8.0, 7.5],
        "avg_neighbor_distance": [2.55, 2.60],
        "electronegativity": [1.9, 1.9],
        "d_band_center": [-1.6, -1.5],
        "surface_energy": [1.55, 1.40],
        "adsorption_energy": [-0.6, -0.4],
        "provenance": ["test_db_v1", "test_db_v1"],
        "unit_adsorption_energy": ["eV", "eV"],
    }
    p = tmp_path / "std.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def _make_nonstandard_csv(tmp_path: Path) -> Path:
    """Write a CSV with non-standard column names that need mapping."""
    data = {
        "ID": ["cu_0001", "cu_0002"],
        "Facet": ["111", "100"],
        "CoordNum": [8.0, 7.5],
        "AvgNeighborDist": [2.55, 2.60],
        "Electroneg": [1.9, 1.9],
        "DBand": [-1.6, -1.5],
        "SurfEn": [1.55, 1.40],
        "AdsEnergy": [-0.6, -0.4],
        "Source": ["test_db_v1", "test_db_v1"],
        "Unit": ["eV", "eV"],
    }
    p = tmp_path / "nonstd.csv"
    pd.DataFrame(data).to_csv(p, index=False)
    return p


def _make_parquet(tmp_path: Path) -> Path:
    """Write a minimal Parquet with standard column names."""
    data = {
        "catalyst_id": ["cu_pq_0001"],
        "facet": ["211"],
        "coordination_number": [9.0],
        "avg_neighbor_distance": [2.50],
        "electronegativity": [1.9],
        "d_band_center": [-1.7],
        "surface_energy": [1.60],
        "adsorption_energy": [-0.7],
        "provenance": ["pq_test"],
        "unit_adsorption_energy": ["eV"],
    }
    p = tmp_path / "data.parquet"
    pd.DataFrame(data).to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_table_source_reads_standard_csv(tmp_path: Path) -> None:
    """CSV with standard column names should be read and written unchanged."""
    src = _make_standard_csv(tmp_path)
    out_path = tmp_path / "raw.parquet"

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        raw_output=str(out_path),
    )

    assert len(df) == 2
    assert "catalyst_id" in df.columns
    assert "adsorption_energy" in df.columns
    assert out_path.exists()


def test_table_source_applies_column_mapping(tmp_path: Path) -> None:
    """Non-standard column names should be renamed via column_mapping."""
    src = _make_nonstandard_csv(tmp_path)
    out_path = tmp_path / "raw.parquet"

    mapping = {
        "ID": "catalyst_id",
        "Facet": "facet",
        "CoordNum": "coordination_number",
        "AvgNeighborDist": "avg_neighbor_distance",
        "Electroneg": "electronegativity",
        "DBand": "d_band_center",
        "SurfEn": "surface_energy",
        "AdsEnergy": "adsorption_energy",
        "Source": "provenance",
        "Unit": "unit_adsorption_energy",
    }

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        column_mapping=mapping,
        raw_output=str(out_path),
    )

    assert "catalyst_id" in df.columns
    assert "adsorption_energy" in df.columns
    assert "ID" not in df.columns


def test_table_source_fills_defaults_for_missing_columns(tmp_path: Path) -> None:
    """Defaults should be injected for columns absent from the source file."""
    src = _make_nonstandard_csv(tmp_path)
    out_path = tmp_path / "raw.parquet"

    mapping = {
        "ID": "catalyst_id",
        "Facet": "facet",
        "CoordNum": "coordination_number",
        "AvgNeighborDist": "avg_neighbor_distance",
        "Electroneg": "electronegativity",
        "DBand": "d_band_center",
        "SurfEn": "surface_energy",
        "AdsEnergy": "adsorption_energy",
        "Source": "provenance",
        "Unit": "unit_adsorption_energy",
    }

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        column_mapping=mapping,
        defaults={"element": "Cu", "adsorbate": "CO"},
        raw_output=str(out_path),
    )

    assert "element" in df.columns
    assert (df["element"] == "Cu").all()
    assert "adsorbate" in df.columns
    assert (df["adsorbate"] == "CO").all()


def test_table_source_injects_target_definition(tmp_path: Path) -> None:
    """target_definition should be added as a constant column."""
    src = _make_standard_csv(tmp_path)
    out_path = tmp_path / "raw.parquet"

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        target_definition="co_adsorption_energy_ev_v1",
        raw_output=str(out_path),
    )

    assert "target_definition" in df.columns
    assert (df["target_definition"] == "co_adsorption_energy_ev_v1").all()


def test_table_source_reads_parquet(tmp_path: Path) -> None:
    """Parquet input should be read without errors."""
    src = _make_parquet(tmp_path)
    out_path = tmp_path / "raw.parquet"

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        raw_output=str(out_path),
    )

    assert len(df) == 1
    assert "adsorption_energy" in df.columns


def test_table_source_raises_without_input_path(tmp_path: Path) -> None:
    """Missing input_path must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="data.input_path must be set"):
        fetch_data(
            source_name="table",
            output_path=str(tmp_path / "out.parquet"),
        )


def test_demo_source_still_works(tmp_path: Path) -> None:
    """Demo source must be unaffected by table-source changes."""
    out_path = tmp_path / "demo.parquet"
    df = fetch_data(source_name="demo", output_path=str(out_path), n_samples=30, seed=0)
    assert len(df) == 30
    assert out_path.exists()


def test_table_source_fill_defaults_injects_adsorbate(tmp_path: Path) -> None:
    """fill_defaults (formerly 'defaults') must inject adsorbate column when absent."""
    data = {
        "catalyst_id": ["cu_0001"],
        "facet": ["111"],
        "coordination_number": [8.0],
        "avg_neighbor_distance": [2.55],
        "electronegativity": [1.9],
        "d_band_center": [-1.6],
        "surface_energy": [1.55],
        "adsorption_energy": [-0.6],
        "provenance": ["test_db_v1"],
        "unit_adsorption_energy": ["eV"],
        # 'adsorbate' intentionally absent — fill_defaults should inject it
    }
    src = tmp_path / "no_adsorbate.csv"
    pd.DataFrame(data).to_csv(src, index=False)
    out_path = tmp_path / "raw.parquet"

    df = fetch_data(
        source_name="table",
        output_path=str(out_path),
        input_path=str(src),
        defaults={"adsorbate": "CO", "element": "Cu"},
        raw_output=str(out_path),
    )
    assert "adsorbate" in df.columns
    assert df.loc[0, "adsorbate"] == "CO"


def test_table_source_without_adsorbate_and_no_default(tmp_path: Path) -> None:
    """Without adsorbate in data or fill_defaults, validate_required_columns must raise."""
    from cu_catalyst_ai.clean.validate_conditions import validate_required_columns  # noqa: PLC0415

    data = {
        "catalyst_id": ["cu_0001"],
        "facet": ["111"],
        "coordination_number": [8.0],
        "avg_neighbor_distance": [2.55],
        "electronegativity": [1.9],
        "d_band_center": [-1.6],
        "surface_energy": [1.55],
        "adsorption_energy": [-0.6],
        "provenance": ["test_db_v1"],
        "unit_adsorption_energy": ["eV"],
        # 'adsorbate' absent, no default injected
    }
    import pytest  # noqa: PLC0415

    df = pd.DataFrame(data)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


# ---------------------------------------------------------------------------
# Regression guards — cathub additions must not break existing paths
# ---------------------------------------------------------------------------


def test_demo_source_still_works_after_cathub_changes(tmp_path: Path) -> None:
    """Demo path must remain unaffected by cathub source additions."""
    out_path = tmp_path / "demo_regression.parquet"
    df = fetch_data(source_name="demo", output_path=str(out_path), n_samples=20, seed=7)
    assert len(df) == 20
    assert out_path.exists()
    # Core demo columns must all be present.
    for col in ("catalyst_id", "adsorption_energy", "d_band_center", "surface_energy"):
        assert col in df.columns, f"Missing column after cathub changes: {col}"


def test_fetch_data_unknown_source_raises(tmp_path: Path) -> None:
    """Unknown source_name must raise ValueError, not silently fall through."""
    with pytest.raises(ValueError, match="Unknown data source"):
        fetch_data(
            source_name="nonexistent_source_xyz",
            output_path=str(tmp_path / "out.parquet"),
        )
