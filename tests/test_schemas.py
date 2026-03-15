from cu_catalyst_ai.schemas.catalyst import CatalystRecord


def test_catalyst_schema_accepts_valid_record() -> None:
    record = CatalystRecord(
        catalyst_id="cu_0001",
        facet="111",
        coordination_number=8.0,
        avg_neighbor_distance=2.55,
        electronegativity=1.9,
        d_band_center=-1.6,
        surface_energy=1.5,
        adsorption_energy=-0.6,
        provenance="demo",
    )
    assert record.catalyst_id == "cu_0001"
    assert record.target_definition is None


def test_catalyst_schema_accepts_target_definition() -> None:
    """CatalystRecord should accept and store a target_definition string."""
    record = CatalystRecord(
        catalyst_id="cu_0002",
        facet="100",
        coordination_number=7.5,
        avg_neighbor_distance=2.60,
        electronegativity=1.9,
        d_band_center=-1.5,
        surface_energy=1.4,
        adsorption_energy=-0.4,
        provenance="test_db_v1",
        target_definition="co_adsorption_energy_ev_v1",
    )
    assert record.target_definition == "co_adsorption_energy_ev_v1"
