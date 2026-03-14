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
