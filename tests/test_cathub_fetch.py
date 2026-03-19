"""Tests for the Catalysis-Hub GraphQL fetch module.

All tests use mock HTTP responses — no network calls are made.
"""

from __future__ import annotations

import itertools
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from cu_catalyst_ai.dataio.cathub_fetch import (
    _build_provenance,
    _compute_structural_features,
    _derive_adsorbate,
    _make_catalyst_id,
    fetch_cathub_reactions,
    parse_cathub_response,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_reaction(
    *,
    reaction_id: str = "rxn_001",
    pub_id: str = "TestPaper2024",
    reaction_energy: float = -0.65,
    facet: str = "111",
    reactants: str = "COgas",
    products: str = "COstar",
    doi: str = "10.1234/test",
    year: int = 2024,
    systems: list[dict] | None = None,
    surface_composition: str = "Cu",
) -> dict[str, Any]:
    """Build a minimal mock reaction dict matching the API structure."""
    return {
        "id": reaction_id,
        "pubId": pub_id,
        "reactionEnergy": reaction_energy,
        "facet": facet,
        "reactants": reactants,
        "products": products,
        "doi": doi,
        "title": "Test Paper",
        "year": year,
        "dftCode": "VASP",
        "dftFunctional": "PBE",
        "systems": systems or [],
        "surfaceComposition": surface_composition,  # required by _infer_element
        "publication": {"doi": doi, "year": year},
    }


def _mock_graphql_page(
    reactions: list[dict],
    has_next_page: bool = False,
    end_cursor: str | None = None,
) -> dict[str, Any]:
    """Build a mock GraphQL response dict for one page."""
    return {
        "data": {
            "reactions": {
                "pageInfo": {
                    "hasNextPage": has_next_page,
                    "endCursor": end_cursor,
                },
                "edges": [{"node": rxn} for rxn in reactions],
            }
        }
    }


def _minimal_structure_positions() -> list[list[float]]:
    """Return 5 atoms in a pseudo-Cu slab arrangement (Å)."""
    return [
        [0.0, 0.0, 0.0],
        [2.55, 0.0, 0.0],
        [1.275, 2.21, 0.0],
        [1.275, 0.74, 2.08],
        [0.0, 0.0, 2.08],
    ]


def _make_mock_response(page_data: dict) -> MagicMock:
    """Build a mock requests.Response that returns *page_data* as JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = page_data
    return mock_resp


def _make_mock_session(*responses: MagicMock) -> MagicMock:
    """Return a mock Session whose .post() returns *responses* in order.

    NOTE: The production code calls ``_make_session()`` internally and then
    calls ``session.post()``.  We therefore patch ``_make_session`` (not
    ``requests.post``) so the mock is exercised through the real code path.
    """
    mock_session = MagicMock()
    if len(responses) == 1:
        mock_session.post.return_value = responses[0]
    else:
        mock_session.post.side_effect = list(responses)
    return mock_session


# ---------------------------------------------------------------------------
# _derive_adsorbate
# ---------------------------------------------------------------------------


def test_derive_adsorbate_co() -> None:
    """'CO' found in reactants → returns 'CO'."""
    result = _derive_adsorbate("COgas", "star")
    assert result == "CO"


def test_derive_adsorbate_products() -> None:
    """'CO' found in products → returns 'CO'."""
    result = _derive_adsorbate("star", "COstar")
    assert result == "CO"


def test_derive_adsorbate_oh() -> None:
    """'OH' in reactants should return 'OH'."""
    result = _derive_adsorbate("OHgas", "star")
    assert result == "OH"


def test_derive_adsorbate_fallback() -> None:
    """Unrecognised adsorbate falls back to 'CO' without raising."""
    result = _derive_adsorbate("XYZgas", "ZZZstar")
    assert result == "CO"


def test_derive_adsorbate_none_inputs() -> None:
    """None inputs should return 'CO' without raising."""
    result = _derive_adsorbate(None, None)
    assert result == "CO"


# ---------------------------------------------------------------------------
# _make_catalyst_id
# ---------------------------------------------------------------------------


def test_make_catalyst_id_stable() -> None:
    """Same inputs must always produce the same output."""
    id1 = _make_catalyst_id("PaperA2024", "rxn_001")
    id2 = _make_catalyst_id("PaperA2024", "rxn_001")
    assert id1 == id2


def test_make_catalyst_id_contains_both_parts() -> None:
    """Output must contain both pubId and reactionId components."""
    result = _make_catalyst_id("PaperA", "rxn001")
    assert "PaperA" in result
    assert "rxn001" in result


def test_make_catalyst_id_url_safe() -> None:
    """Non-word characters should be replaced with underscores."""
    result = _make_catalyst_id("Paper/A.2024", "rxn-001")
    import re  # noqa: PLC0415

    assert re.match(r"^[\w_]+$", result), f"Not URL-safe: {result!r}"


def test_make_catalyst_id_none_inputs() -> None:
    """None inputs must not raise."""
    result = _make_catalyst_id(None, None)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# _build_provenance
# ---------------------------------------------------------------------------


def test_build_provenance_format() -> None:
    """Provenance string must contain pubId and year."""
    result = _build_provenance("TestPaper2024", "10.1234/test", 2024)
    assert "TestPaper2024" in result
    assert "2024" in result


def test_build_provenance_none_doi() -> None:
    """None doi must not raise; empty string accepted in output."""
    result = _build_provenance("Paper", None, 2023)
    assert "Paper" in result
    assert "2023" in result


def test_build_provenance_none_all() -> None:
    """All-None inputs must not raise."""
    result = _build_provenance(None, None, None)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _compute_structural_features
# ---------------------------------------------------------------------------


def test_compute_structural_features_valid() -> None:
    """Minimal slab structure → non-None coord and dist."""
    systems = [{"positions": _minimal_structure_positions()}]
    coord, dist = _compute_structural_features(systems)
    assert coord is not None
    assert dist is not None
    assert coord > 0
    assert dist > 0


def test_compute_structural_features_missing_systems() -> None:
    """None systems → (None, None)."""
    coord, dist = _compute_structural_features(None)
    assert coord is None
    assert dist is None


def test_compute_structural_features_empty_systems() -> None:
    """Empty systems list → (None, None)."""
    coord, dist = _compute_structural_features([])
    assert coord is None
    assert dist is None


def test_compute_structural_features_missing_positions() -> None:
    """System with no 'positions' key → (None, None)."""
    systems = [{"natoms": 5}]  # no positions
    coord, dist = _compute_structural_features(systems)
    assert coord is None
    assert dist is None


def test_compute_structural_features_single_atom() -> None:
    """Single-atom structure has no neighbours → (None, None)."""
    systems = [{"positions": [[0.0, 0.0, 0.0]]}]
    coord, dist = _compute_structural_features(systems)
    assert coord is None
    assert dist is None


def test_compute_structural_features_widely_spaced() -> None:
    """Atoms far beyond cutoff → all isolated → (None, None)."""
    # Two atoms 100 Å apart — no bond within 3.5 Å.
    systems = [{"positions": [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]}]
    coord, dist = _compute_structural_features(systems)
    assert coord is None
    assert dist is None


# ---------------------------------------------------------------------------
# parse_cathub_response
# ---------------------------------------------------------------------------


def test_parse_cathub_response_empty() -> None:
    """Empty reaction list → empty DataFrame with correct column names."""
    df = parse_cathub_response([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "adsorption_energy" in df.columns
    assert "catalyst_id" in df.columns


def test_parse_cathub_response_basic_mapping() -> None:
    """Single reaction → all direct-mapped fields are correct."""
    rxn = _mock_reaction(reaction_energy=-0.65, facet="111", pub_id="PaperA", reaction_id="r1")
    df = parse_cathub_response([rxn])

    assert len(df) == 1
    assert df.loc[0, "adsorption_energy"] == pytest.approx(-0.65)
    assert df.loc[0, "facet"] == "111"
    assert df.loc[0, "unit_adsorption_energy"] == "eV"
    assert df.loc[0, "element"] == "Cu"  # inferred from surfaceComposition="Cu"
    # electronegativity is NaN at fetch time; filled later by enrich_with_element_features
    assert pd.isna(df.loc[0, "electronegativity"])
    assert df.loc[0, "target_definition"] == "co_adsorption_energy_ev_v1"


def test_parse_cathub_response_d_band_center_not_faked() -> None:
    """d_band_center must be NaN — never faked from API data."""
    rxn = _mock_reaction()
    df = parse_cathub_response([rxn])
    assert pd.isna(df.loc[0, "d_band_center"])


def test_parse_cathub_response_surface_energy_not_faked() -> None:
    """surface_energy must be NaN — never faked from API data."""
    rxn = _mock_reaction()
    df = parse_cathub_response([rxn])
    assert pd.isna(df.loc[0, "surface_energy"])


def test_parse_cathub_response_catalyst_id_stable() -> None:
    """Same pub_id + reaction_id always yields the same catalyst_id."""
    rxn = _mock_reaction(pub_id="PaperA", reaction_id="r1")
    df1 = parse_cathub_response([rxn])
    df2 = parse_cathub_response([rxn])
    assert df1.loc[0, "catalyst_id"] == df2.loc[0, "catalyst_id"]


def test_parse_cathub_response_provenance_contains_pub_id() -> None:
    """Provenance string must contain the pub_id."""
    rxn = _mock_reaction(pub_id="MyPublication2024")
    df = parse_cathub_response([rxn])
    assert "MyPublication2024" in df.loc[0, "provenance"]


def test_parse_cathub_response_structural_features_when_available() -> None:
    """When positions are provided, structural features must be non-NaN."""
    systems = [{"positions": _minimal_structure_positions()}]
    rxn = _mock_reaction(systems=systems)
    df = parse_cathub_response([rxn])
    assert pd.notna(df.loc[0, "coordination_number"])
    assert pd.notna(df.loc[0, "avg_neighbor_distance"])


def test_parse_cathub_response_structural_features_nan_when_absent() -> None:
    """When systems is empty, coord/dist must be NaN."""
    rxn = _mock_reaction(systems=[])
    df = parse_cathub_response([rxn])
    assert pd.isna(df.loc[0, "coordination_number"])
    assert pd.isna(df.loc[0, "avg_neighbor_distance"])


def test_parse_cathub_response_multiple_rows() -> None:
    """Multiple reactions → correct row count."""
    reactions = [
        _mock_reaction(reaction_id=f"r{i}", reaction_energy=-0.5 - i * 0.1) for i in range(5)
    ]
    df = parse_cathub_response(reactions)
    assert len(df) == 5


def test_parse_cathub_response_custom_target_definition() -> None:
    """Custom target_definition is injected correctly."""
    rxn = _mock_reaction()
    df = parse_cathub_response([rxn], target_definition="custom_target_v2")
    assert (df["target_definition"] == "custom_target_v2").all()


# ---------------------------------------------------------------------------
# fetch_cathub_reactions (mocked HTTP via _make_session)
# ---------------------------------------------------------------------------
# NOTE: The production code calls _make_session() and then session.post().
# Patching requests.post directly does NOT intercept session.post() calls.
# We therefore patch _make_session to return a controlled mock session.


def test_fetch_cathub_reactions_single_page() -> None:
    """Single page → all reactions returned, no second request."""
    reactions = [_mock_reaction(reaction_id=f"r{i}") for i in range(3)]
    page = _mock_graphql_page(reactions, has_next_page=False)
    mock_session = _make_mock_session(_make_mock_response(page))

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        result = fetch_cathub_reactions(query_filter={"first": 10})

    assert len(result) == 3
    assert mock_session.post.call_count == 1


def test_fetch_cathub_reactions_pagination() -> None:
    """Two pages → reactions from both pages are combined."""
    page1_reactions = [_mock_reaction(reaction_id=f"r{i}") for i in range(3)]
    page2_reactions = [_mock_reaction(reaction_id=f"r{i + 3}") for i in range(2)]

    page1 = _mock_graphql_page(page1_reactions, has_next_page=True, end_cursor="cursor_abc")
    page2 = _mock_graphql_page(page2_reactions, has_next_page=False)
    mock_session = _make_mock_session(
        _make_mock_response(page1),
        _make_mock_response(page2),
    )

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        result = fetch_cathub_reactions(query_filter={"first": 3})

    assert len(result) == 5
    assert mock_session.post.call_count == 2


def test_fetch_cathub_reactions_empty_result() -> None:
    """Empty edges → empty list, no crash."""
    page = _mock_graphql_page([], has_next_page=False)
    mock_session = _make_mock_session(_make_mock_response(page))

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        result = fetch_cathub_reactions()

    assert result == []


def test_fetch_cathub_reactions_malformed_response_raises() -> None:
    """Malformed API response (missing 'data' key) → ValueError."""
    bad_response = {"errors": [{"message": "Something went wrong"}]}
    mock_session = _make_mock_session(_make_mock_response(bad_response))

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        with pytest.raises(ValueError, match="Unexpected API response structure"):
            fetch_cathub_reactions()


def test_fetch_cathub_reactions_passes_query_filter() -> None:
    """Query filter parameters are forwarded to the HTTP POST call."""
    page = _mock_graphql_page([], has_next_page=False)
    mock_session = _make_mock_session(_make_mock_response(page))

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        fetch_cathub_reactions(
            api_url="https://custom.api/graphql",
            query_filter={"surface_composition": "Pt", "reactants": "O", "first": 5},
        )

    call_kwargs = mock_session.post.call_args
    assert call_kwargs[0][0] == "https://custom.api/graphql"
    variables = call_kwargs[1]["json"]["variables"]
    assert variables["surfaceComposition"] == "Pt"
    assert variables["reactants"] == "O"
    assert variables["first"] == 5


def test_fetch_cathub_reactions_503_raises_http_error() -> None:
    """A 503 response after all retries → HTTPError propagates to the caller.

    Regression test: previously raise_on_status=False on the Retry object
    silently disabled status_forcelist, so the 503 was returned instead of
    retried, and raise_for_status() raised immediately without any retry.
    Now urllib3 retries up to _RETRY_TOTAL times, then raises MaxRetryError
    (wrapped as ConnectionError) or HTTPError on exhaustion.
    We simulate the exhausted-retries end-state directly.
    """
    mock_session = MagicMock()
    mock_session.post.side_effect = requests.exceptions.HTTPError(
        "503 Server Error: Service Unavailable"
    )

    with patch("cu_catalyst_ai.dataio.cathub_fetch._make_session", return_value=mock_session):
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_cathub_reactions()


# ---------------------------------------------------------------------------
# Multi-element graceful degradation (mp_fetch integration)
# ---------------------------------------------------------------------------


def test_fetch_data_cathub_graceful_degradation_on_503(tmp_path: pytest.TempPathFactory) -> None:
    """503 on one element logs a warning and skips it; other elements succeed.

    Regression: previously HTTPError from one element crashed the entire
    multi-element loop in fetch_data('cathub', ...).
    """
    from cu_catalyst_ai.dataio.mp_fetch import fetch_data

    ok_page = _mock_graphql_page(
        [_mock_reaction(reaction_id="r1", surface_composition="Cu")],
        has_next_page=False,
    )

    # Each call to _make_session() returns a fresh mock.
    # Element "Fe" → session raises 503; Element "Cu" → session returns ok_page.
    call_counter = itertools.count()

    def _session_factory() -> MagicMock:
        idx = next(call_counter)
        mock_session = MagicMock()
        if idx == 0:
            # Fe → 503
            mock_session.post.side_effect = requests.exceptions.HTTPError(
                "503 Server Error: Service Unavailable"
            )
        else:
            # Cu → success
            mock_session.post.return_value = _make_mock_response(ok_page)
        return mock_session

    out = tmp_path / "out.parquet"
    with patch(
        "cu_catalyst_ai.dataio.cathub_fetch._make_session",
        side_effect=_session_factory,
    ):
        df = fetch_data(
            source_name="cathub",
            output_path=str(out),
            cathub_kwargs={
                "target_elements": ["Fe", "Cu"],
                "query_filter": {"reactants": "CO", "first": 10},
            },
        )

    # Fe failed → only Cu rows present; pipeline did NOT crash.
    assert len(df) >= 1
    assert "Cu" in df["element"].values
