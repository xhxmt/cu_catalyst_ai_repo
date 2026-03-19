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


def test_catalyst_schema_accepts_non_cu_element() -> None:
    """CatalystRecord must accept any element symbol, not just 'Cu'."""
    for element in ("Pt", "Pd", "Ru", "Fe", "Ag"):
        record = CatalystRecord(
            catalyst_id=f"{element.lower()}_001",
            element=element,
            facet="111",
            adsorption_energy=-1.0,
            provenance="test",
        )
        assert record.element == element


def test_catalyst_schema_accepts_nan_optional_fields() -> None:
    """Optional fields (coordination_number, d_band_center, etc.) may be None."""
    record = CatalystRecord(
        catalyst_id="cathub_001",
        element="Pt",
        facet="111",
        adsorption_energy=-0.8,
        provenance="cathub|10.1234/test|2023",
        # All optional fields omitted — should not raise
    )
    assert record.coordination_number is None
    assert record.d_band_center is None
    assert record.electronegativity is None
