"""Catalysis-Hub GraphQL data fetcher.

Pulls reaction data from the Catalysis-Hub GraphQL API, handles pagination,
and standardises the response into the project's canonical raw-table schema.

Supported features (v2 — multi-metal)
--------------------------------------
Direct mapping from API:
    adsorption_energy  ← reactionEnergy
    facet              ← facet
    adsorbate          ← _derive_adsorbate(reactants, products)
    element            ← _infer_element(surfaceComposition)
    catalyst_id        ← _make_catalyst_id(pubId, id)
    provenance         ← _build_provenance(pubId, doi, year)
    unit_adsorption_energy = "eV"  (hard-coded)

Left as NaN (not faked):
    electronegativity      (filled later by enrich_with_element_features)
    d_band_center          (filled later by enrich_with_element_features)
    surface_energy         (not available from API)
    coordination_number    (derived from structure when present)
    avg_neighbor_distance  (derived from structure when present)

DFT functional filtering:
    Pass dft_functional_filter="BEEF-vdW" to keep only records computed
    with that functional, ensuring cross-metal energy comparability.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_API_URL = "https://api.catalysis-hub.org/graphql"
_BOND_CUTOFF_ANGSTROM = 3.5  # conservative bond cutoff for neighbour detection

# ---------------------------------------------------------------------------
# GraphQL query builder
# ---------------------------------------------------------------------------

_REACTION_QUERY_TEMPLATE = """
query CathubReactions(
  $first: Int,
  $after: String,
  $surfaceComposition: String,
  $reactants: String
) {
  reactions(
    first: $first,
    after: $after,
    surfaceComposition: $surfaceComposition,
    reactants: $reactants
  ) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        reactionEnergy
        facet
        reactants
        products
        pubId
        dftCode
        dftFunctional
        surfaceComposition
        publication {
          doi
          year
          title
        }
        systems {
          id
          uniqueId
          Cifdata
          natoms
          positions
          numbers
          cell
        }
      }
    }
  }
}
"""


def _build_graphql_variables(
    first: int,
    after: str | None,
    surface_composition: str,
    reactants: str,
) -> dict[str, Any]:
    """Build the GraphQL variables dict."""
    return {
        "first": first,
        "after": after,
        "surfaceComposition": surface_composition,
        "reactants": reactants,
    }


# ---------------------------------------------------------------------------
# Element inference
# ---------------------------------------------------------------------------

# Two-letter element symbols that could be confused with one-letter ones.
_TWO_LETTER_ELEMENTS = {
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
}


def _infer_element(surface_composition: str | None) -> str:
    """Extract the primary metal element from a *surface_composition* string.

    The Catalysis-Hub API returns ``surfaceComposition`` as a chemical
    formula string such as ``"Cu"``, ``"Pt"``, ``"Pd3Fe"`` (alloy), or
    ``"Cu(111)"`` (with Miller index).  This function returns the first
    capitalised element symbol it finds.

    For pure metals this is unambiguous.  For alloys the first element is
    taken as the host metal (majority component) as a best-effort heuristic.
    Falls back to ``"unknown"`` on parse failure.

    Args:
        surface_composition: Raw ``surfaceComposition`` field from the API.

    Returns:
        Element symbol string, e.g. ``"Cu"``, ``"Pt"``.
    """
    if not surface_composition:
        return "unknown"
    # Find first occurrence of an uppercase letter followed by optional
    # lowercase letter — this matches standard element symbol patterns.
    m = re.search(r"([A-Z][a-z]?)", str(surface_composition))
    if not m:
        return "unknown"
    candidate = m.group(1)
    # If the two-letter version exists, prefer it over the one-letter prefix.
    if len(candidate) == 1:
        two = surface_composition[m.start() : m.start() + 2]
        if len(two) == 2 and two[1].islower() and two in _TWO_LETTER_ELEMENTS:
            candidate = two
    return candidate


# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------

_RETRY_TOTAL = 3
_RETRY_BACKOFF_FACTOR = 2  # waits 2s, 4s, 8s between attempts
_RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def _make_session() -> requests.Session:
    """Return a :class:`requests.Session` configured with exponential backoff.

    Retries up to :data:`_RETRY_TOTAL` times on transient HTTP errors
    (codes in :data:`_RETRY_STATUS_FORCELIST`) with a backoff factor of
    :data:`_RETRY_BACKOFF_FACTOR` seconds.
    """
    session = requests.Session()
    retry = Retry(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUS_FORCELIST,
        allowed_methods=["POST"],
        # NOTE: do NOT set raise_on_status=False here.
        # When raise_on_status=False, urllib3 skips the status_forcelist check
        # and returns the bad response directly, so no retry ever happens.
        # Leaving it at the default (True) means urllib3 retries on any code
        # in status_forcelist and only raises MaxRetryError after all attempts
        # are exhausted — which requests then converts to an HTTPError.
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Pagination / HTTP
# ---------------------------------------------------------------------------


def fetch_cathub_reactions(
    api_url: str = _DEFAULT_API_URL,
    query_filter: dict[str, Any] | None = None,
    dft_functional_filter: str | None = None,
    page_delay: float = 2.0,
) -> list[dict[str, Any]]:
    """Fetch all matching reactions from the Catalysis-Hub GraphQL API.

    Handles cursor-based pagination automatically.  All pages are fetched
    and merged into a single flat list of reaction node dicts.

    Args:
        api_url: GraphQL endpoint URL.
        query_filter: Optional dict with keys ``surface_composition``,
            ``reactants``, ``first`` (page size), and ``after`` (start cursor).
        dft_functional_filter: If set, only reactions where ``dftFunctional``
            exactly matches this string are returned.  E.g. ``"BEEF-vdW"``.
            ``None`` disables filtering.
        page_delay: Seconds to wait **between** consecutive page requests.
            Helps prevent 503 / rate-limit errors on the CatHub API.
            Set to 0 to disable.  Default is 2 seconds.

    Returns:
        List of raw reaction dicts as returned by the API (one per edge node).

    Raises:
        requests.HTTPError: If any HTTP request fails with a non-2xx status.
        ValueError: If the API response does not contain expected structure.
    """
    if query_filter is None:
        query_filter = {}

    first: int = int(query_filter.get("first", 100))
    after: str | None = query_filter.get("after") or None
    surface_composition: str = str(query_filter.get("surface_composition", "Cu"))
    reactants: str = str(query_filter.get("reactants", "CO"))

    all_reactions: list[dict[str, Any]] = []
    page = 0
    session = _make_session()

    while True:
        page += 1

        # Polite inter-page delay: skip on the first request, apply from page 2
        # onward so we don't hammer the API and trigger rate-limiting / 503s.
        if page > 1 and page_delay > 0:
            logger.debug("Sleeping %.1fs before page %d (page_delay)", page_delay, page)
            time.sleep(page_delay)

        variables = _build_graphql_variables(first, after, surface_composition, reactants)
        payload = {"query": _REACTION_QUERY_TEMPLATE, "variables": variables}

        logger.info("Fetching page %d from %s (first=%d, after=%s)", page, api_url, first, after)
        response = session.post(api_url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        if "errors" in data:
            logger.warning("GraphQL errors: %s", data["errors"])

        try:
            reactions_block = data["data"]["reactions"]
            edges = reactions_block["edges"]
            page_info = reactions_block["pageInfo"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected API response structure on page {page}: {exc}") from exc

        nodes = [edge["node"] for edge in edges if edge.get("node")]

        # Apply DFT functional filter before accumulating.
        if dft_functional_filter is not None:
            before = len(nodes)
            nodes = [
                n
                for n in nodes
                if str(n.get("dftFunctional") or "").strip() == dft_functional_filter
            ]
            filtered = before - len(nodes)
            if filtered:
                logger.debug(
                    "Page %d: dropped %d/%d nodes (dftFunctional != %r)",
                    page,
                    filtered,
                    before,
                    dft_functional_filter,
                )

        all_reactions.extend(nodes)
        logger.info(
            "Page %d: fetched %d reactions (total so far: %d)",
            page,
            len(nodes),
            len(all_reactions),
        )

        if not page_info.get("hasNextPage") or not nodes:
            break
        after = page_info.get("endCursor")

    logger.info("Total reactions fetched: %d", len(all_reactions))
    return all_reactions


# ---------------------------------------------------------------------------
# Adsorbate detection
# ---------------------------------------------------------------------------

# Known adsorbates in roughly descending priority order.
_KNOWN_ADSORBATES = [
    "CO",
    "CO2",
    "OH",
    "O",
    "H",
    "N",
    "NO",
    "CH4",
    "CH3",
    "CH2",
    "CH",
    "C",
    "NH3",
    "NH2",
    "NH",
    "N2",
    "H2O",
    "H2",
]


def _derive_adsorbate(
    reactants: str | dict | None,
    products: str | dict | None,
) -> str:
    """Extract the adsorbate label from reactants/products fields.

    The API returns reactants/products either as a JSON-serialised string
    of a dict or as a plain string.  This function searches for known
    adsorbate tags and returns the first match.  Falls back to ``"CO"``
    with a warning when no known adsorbate is found.

    The Catalysis-Hub API typically encodes adsorbates as e.g. ``"COgas"``,
    ``"OHstar"``, ``"Hgas"`` — uppercase adsorbate name followed by a
    lowercase suffix.  The search is done in the original mixed-case string
    so that ``"OHgas"`` matches ``OH``, not ``CO``.

    Args:
        reactants: Raw ``reactants`` field from the API response.
        products: Raw ``products`` field from the API response.

    Returns:
        Adsorbate label string (e.g. ``"CO"``).
    """
    combined = f"{reactants} {products}" if reactants or products else ""

    for ads in _KNOWN_ADSORBATES:
        # Match the adsorbate tag when followed by a lowercase letter,
        # digit, non-word character, or end-of-string — but NOT another
        # uppercase letter (which would mean it is embedded in a longer symbol).
        if re.search(r"(?<![A-Z])" + re.escape(ads) + r"(?![A-Z])", combined):
            return ads

    logger.warning(
        "Could not detect adsorbate from reactants=%r products=%r; defaulting to 'CO'",
        reactants,
        products,
    )
    return "CO"


# ---------------------------------------------------------------------------
# ID / provenance helpers
# ---------------------------------------------------------------------------


def _make_catalyst_id(pub_id: str | None, reaction_id: str | None) -> str:
    """Build a deterministic, URL-safe catalyst identifier.

    Args:
        pub_id: Publication identifier from the API (``pubId``).
        reaction_id: Reaction node identifier from the API (``id``).

    Returns:
        String of the form ``"<pub_id>_<reaction_id>"`` with non-word
        characters replaced by underscores.
    """
    pid = re.sub(r"[^\w]", "_", str(pub_id or "unknown"))
    rid = re.sub(r"[^\w]", "_", str(reaction_id or "unknown"))
    return f"{pid}_{rid}"


def _build_provenance(
    pub_id: str | None,
    doi: str | None,
    year: int | str | None,
) -> str:
    """Build a concise provenance string.

    Args:
        pub_id: Publication identifier.
        doi:    DOI string (may be empty / None).
        year:   Publication year (integer or string).

    Returns:
        Pipe-separated string ``"<pub_id>|<doi>|<year>"``.
    """
    parts = [
        str(pub_id or "unknown"),
        str(doi or ""),
        str(year or ""),
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Structural feature computation
# ---------------------------------------------------------------------------


def _compute_structural_features(
    systems: list[dict[str, Any]] | None,
) -> tuple[float | None, float | None]:
    """Compute coordination number and avg neighbour distance from structure data.

    Uses a simple distance-cutoff approach (no ASE dependency) with a
    conservative 3.5 Å cutoff.  Only the first system in the list is
    used (typically the slab+adsorbate system).

    Args:
        systems: List of system dicts from the API response.  Each may
            contain ``positions``, ``numbers``, and ``cell`` fields.

    Returns:
        ``(coordination_number, avg_neighbor_distance)`` as floats, or
        ``(None, None)`` if structure data is absent or cannot be parsed.
    """
    if not systems:
        return None, None

    system = systems[0]
    positions_raw = system.get("positions")
    if not positions_raw:
        return None, None

    try:
        pos = np.array(positions_raw, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 2:
            return None, None

        # Compute pairwise distances (no PBC for simplicity in v1).
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dists = np.sqrt((diff**2).sum(axis=-1))  # (N, N)

        # Exclude self-distances.
        np.fill_diagonal(dists, np.inf)

        # Neighbours within cutoff.
        neighbour_mask = dists < _BOND_CUTOFF_ANGSTROM

        coord_nums = neighbour_mask.sum(axis=1).astype(float)  # per atom
        # Only consider atoms that have at least one neighbour.
        bonded = coord_nums > 0
        if not bonded.any():
            return None, None

        avg_coord = float(coord_nums[bonded].mean())

        # Average distance to all neighbours.
        neighbour_dists = np.where(neighbour_mask, dists, np.nan)
        avg_dist = float(np.nanmean(neighbour_dists[bonded]))

        return round(avg_coord, 4), round(avg_dist, 4)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Structural feature computation failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Schema columns (canonical order)
# ---------------------------------------------------------------------------

_CANONICAL_COLUMNS = [
    "catalyst_id",
    "element",
    "facet",
    "adsorbate",
    "coordination_number",
    "avg_neighbor_distance",
    "electronegativity",
    "d_band_center",
    "surface_energy",
    "adsorption_energy",
    "provenance",
    "unit_adsorption_energy",
    "target_definition",
    "dft_functional",  # DFT functional used (e.g. PBE, BEEF-vdW); one-hot encoded as a feature
]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_cathub_response(
    raw_reactions: list[dict[str, Any]],
    target_definition: str = "co_adsorption_energy_ev_v1",
) -> pd.DataFrame:
    """Convert a list of raw Catalysis-Hub reaction dicts to the project schema.

    Args:
        raw_reactions: List of reaction node dicts as returned by
            :func:`fetch_cathub_reactions`.
        target_definition: Name of the registered target definition to
            inject as the ``target_definition`` column.

    Returns:
        ``pd.DataFrame`` with canonical project columns.
        Fields that cannot be obtained from the API (``d_band_center``,
        ``surface_energy``) are left as ``NaN``; they are **never faked**.
    """
    if not raw_reactions:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)

    rows: list[dict[str, Any]] = []
    for rxn in raw_reactions:
        coord_num, avg_dist = _compute_structural_features(rxn.get("systems"))

        adsorbate = _derive_adsorbate(rxn.get("reactants"), rxn.get("products"))
        # surfaceComposition is now included in the GraphQL query, so it is
        # available directly on the node.  Fall back to inference from the
        # string when the field is present but non-trivial (alloy formulas).
        surface_comp = rxn.get("surfaceComposition")
        element = _infer_element(surface_comp)

        pub = rxn.get("publication") or {}
        row: dict[str, Any] = {
            "catalyst_id": _make_catalyst_id(rxn.get("pubId"), rxn.get("id")),
            "element": element,
            "facet": str(rxn.get("facet") or ""),
            "adsorbate": adsorbate,
            "coordination_number": coord_num,
            "avg_neighbor_distance": avg_dist,
            "electronegativity": float("nan"),  # filled later by enrich_with_element_features
            "d_band_center": float("nan"),  # filled later by enrich_with_element_features
            "surface_energy": float("nan"),  # not available from API
            "adsorption_energy": rxn.get("reactionEnergy"),
            "provenance": _build_provenance(
                rxn.get("pubId"),
                pub.get("doi"),
                pub.get("year"),
            ),
            "unit_adsorption_energy": "eV",
            "target_definition": target_definition,
            # Preserve DFT functional for feature encoding (helps model learn systematic
            # energy offsets between PBE, BEEF-vdW, etc.).
            "dft_functional": str(rxn.get("dftFunctional") or "unknown"),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS)

    # Ensure numeric dtypes for energy columns.
    df["adsorption_energy"] = pd.to_numeric(df["adsorption_energy"], errors="coerce")

    logger.info(
        "Parsed %d cathub reactions. coord_num present: %d, avg_dist present: %d",
        len(df),
        df["coordination_number"].notna().sum(),
        df["avg_neighbor_distance"].notna().sum(),
    )
    return df
