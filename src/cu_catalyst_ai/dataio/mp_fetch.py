"""Data ingestion entry-point.

Supported sources
-----------------
demo
    Generates a synthetic Cu catalyst dataset using a fixed random seed.
    Used as the default baseline source.

table
    Reads a local CSV or Parquet file, renames columns according to a
    ``column_mapping`` dict, fills missing metadata via ``defaults``, injects a
    ``target_definition`` column, and writes the standardised output.

cathub
    Fetches reaction data from the Catalysis-Hub GraphQL API.  Results are
    standardised to the project raw-table schema and written to disk before
    entering the cleaning pipeline.  Structural features
    (``coordination_number``, ``avg_neighbor_distance``) are derived from
    structure data when available.  ``d_band_center`` and ``surface_energy``
    are left as NaN — they are never faked.

mp
    Materials Project API — intentionally left as :class:`NotImplementedError`.
    Implement ``_fetch_from_mp()`` in a project-specific extension when needed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cu_catalyst_ai.utils.io import read_table, write_table
from cu_catalyst_ai.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_demo_dataset(n_samples: int = 240, seed: int = 42) -> pd.DataFrame:
    """Return a fully synthetic Cu catalyst dataset.

    All values are generated from fixed distributions calibrated to
    physically plausible ranges for Cu surface DFT calculations.
    """
    rng = np.random.default_rng(seed)
    facets = rng.choice(["111", "100", "110", "211"], size=n_samples, p=[0.35, 0.25, 0.2, 0.2])
    coord = rng.normal(8.0, 1.0, size=n_samples).clip(5.5, 10.5)
    neighbor = rng.normal(2.55, 0.10, size=n_samples).clip(2.2, 2.9)
    d_band = rng.normal(-1.6, 0.25, size=n_samples).clip(-2.4, -0.8)
    surface_energy = rng.normal(1.55, 0.18, size=n_samples).clip(1.1, 2.0)
    electroneg = np.full(n_samples, 1.90)

    facet_bonus = {"111": -0.08, "100": 0.02, "110": 0.10, "211": 0.16}
    adsorption = (
        -0.65
        + 0.11 * (coord - 8.0)
        + 0.70 * (neighbor - 2.55)
        + 0.42 * (d_band + 1.6)
        + 0.28 * (surface_energy - 1.55)
        + np.vectorize(facet_bonus.get)(facets)
        + rng.normal(0.0, 0.08, size=n_samples)
    )

    df = pd.DataFrame(
        {
            "catalyst_id": [f"cu_{i:04d}" for i in range(n_samples)],
            "element": "Cu",
            "facet": facets,
            "adsorbate": "CO",
            "coordination_number": coord.round(4),
            "avg_neighbor_distance": neighbor.round(4),
            "electronegativity": electroneg.round(4),
            "d_band_center": d_band.round(4),
            "surface_energy": surface_energy.round(4),
            "adsorption_energy": adsorption.round(4),
            "provenance": "demo",
            "unit_adsorption_energy": "eV",
        }
    )
    return df


def _fetch_from_table(
    input_path: str,
    output_path: str,
    column_mapping: dict[str, str] | None = None,
    defaults: dict[str, str] | None = None,
    target_definition: str | None = None,
) -> pd.DataFrame:
    """Read a local CSV/Parquet, apply column mapping, fill defaults, write output.

    Args:
        input_path: Absolute or project-relative path to the source file.
        output_path: Where to write the standardised raw table.
        column_mapping: Mapping of source column name → canonical column name.
            Only columns that need renaming must be listed.
        defaults: Constant values to fill for metadata columns that are absent
            or null in the source file.  Keys are standard column names.
        target_definition: Name of the registered target definition.  Written
            as a constant column ``target_definition`` in the output.

    Returns:
        Standardised ``pd.DataFrame`` with canonical column names.
    """
    df = read_table(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    # --- Column renaming ---------------------------------------------------
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info("Applied column_mapping: %s", column_mapping)

    # --- Fill defaults for absent/null metadata columns -------------------
    if defaults:
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value
                logger.info("Injected default column '%s' = %r", col, value)
            else:
                # Fill only nulls (don't overwrite real values)
                null_mask = df[col].isna()
                if null_mask.any():
                    df.loc[null_mask, col] = value
                    logger.info(
                        "Filled %d nulls in column '%s' with default %r",
                        null_mask.sum(),
                        col,
                        value,
                    )

    # --- Inject target_definition -----------------------------------------
    if target_definition is not None:
        df["target_definition"] = target_definition
        logger.info("Injected target_definition = %r", target_definition)

    write_table(df, output_path)
    logger.info("Saved standardised raw table (%d rows) to %s", len(df), output_path)
    return df


def fetch_data(
    source_name: str,
    output_path: str,
    n_samples: int = 240,
    seed: int = 42,
    *,
    input_path: str | None = None,
    column_mapping: dict[str, str] | None = None,
    defaults: dict[str, str] | None = None,
    target_definition: str | None = None,
    raw_output: str | None = None,
    cathub_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Unified data ingestion entry-point.

    Args:
        source_name: One of ``"demo"``, ``"table"``, ``"cathub"``, or ``"mp"``.
        output_path: Primary output path (used by demo; other sources use
            *raw_output* when provided, falling back to *output_path*).
        n_samples: Number of samples to generate (demo source only).
        seed: Random seed (demo source only).
        input_path: Path to the local file (table source only).
        column_mapping: Column rename mapping (table source only).
        defaults: Default values for missing metadata columns (table source only).
        target_definition: Registered target definition name (table/cathub).
        raw_output: Explicit output path for the standardised raw table.
            Falls back to *output_path* if not provided.
        cathub_kwargs: Keyword arguments for the cathub source.  Recognised
            keys: ``api_url`` (str), ``query_filter`` (dict).

    Returns:
        The ingested/generated ``pd.DataFrame``.

    Raises:
        ValueError: For unknown *source_name*.
        RuntimeError: If required parameters for a source are missing.
        NotImplementedError: For the ``"mp"`` source (not yet implemented).
    """
    if source_name == "demo":
        df = generate_demo_dataset(n_samples=n_samples, seed=seed)
        write_table(df, output_path)
        logger.info("Saved demo dataset to %s", output_path)
        return df

    if source_name == "table":
        if not input_path:
            raise RuntimeError(
                "data.input_path must be set when source_name='table'. "
                "Pass it via: data.input_path=/path/to/your/file.csv"
            )
        effective_output = raw_output if raw_output else output_path
        return _fetch_from_table(
            input_path=input_path,
            output_path=effective_output,
            column_mapping=column_mapping,
            defaults=defaults,
            target_definition=target_definition,
        )

    if source_name == "cathub":
        # Lazy import to avoid hard dependency when cathub is not used.
        from cu_catalyst_ai.dataio.cathub_fetch import (  # noqa: PLC0415
            fetch_cathub_reactions,
            parse_cathub_response,
        )

        kw = cathub_kwargs or {}
        api_url: str = str(kw.get("api_url", "https://api.catalysis-hub.org/graphql"))
        query_filter: dict = dict(kw.get("query_filter") or {})
        target_def: str = str(target_definition or "co_adsorption_energy_ev_v1")
        dft_functional_filter: str | None = kw.get("dft_functional_filter") or None
        page_delay: float = float(kw.get("page_delay", 2.0))

        # Support multi-element fetching: iterate over each element separately
        # so that the API surface_composition filter works correctly.
        target_elements: list[str] = list(kw.get("target_elements") or [])
        if not target_elements:
            # Fallback: single fetch using surface_composition from query_filter.
            target_elements = [str(query_filter.get("surface_composition", "Cu"))]

        import requests as _requests  # noqa: PLC0415

        all_frames: list[pd.DataFrame] = []
        failed_elements: list[str] = []
        for element in target_elements:
            elem_filter = dict(query_filter)
            elem_filter["surface_composition"] = element
            logger.info("Fetching CatHub data for element: %s", element)
            try:
                raw = fetch_cathub_reactions(
                    api_url=api_url,
                    query_filter=elem_filter,
                    dft_functional_filter=dft_functional_filter,
                    page_delay=page_delay,
                )
            except _requests.exceptions.HTTPError as exc:
                logger.warning("HTTP error fetching element %s (skipping): %s", element, exc)
                failed_elements.append(element)
                continue
            except _requests.exceptions.ConnectionError as exc:
                logger.warning("Connection error fetching element %s (skipping): %s", element, exc)
                failed_elements.append(element)
                continue
            if raw:
                all_frames.append(parse_cathub_response(raw, target_definition=target_def))
            else:
                logger.warning("No reactions returned for element: %s", element)

        if failed_elements:
            logger.warning(
                "CatHub fetch failed for %d/%d elements (skipped): %s",
                len(failed_elements),
                len(target_elements),
                failed_elements,
            )

        if all_frames:
            df = pd.concat(all_frames, ignore_index=True)
        else:
            # No data at all — return empty frame with canonical columns.
            from cu_catalyst_ai.dataio.cathub_fetch import _CANONICAL_COLUMNS  # noqa: PLC0415

            df = pd.DataFrame(columns=_CANONICAL_COLUMNS)
            logger.warning("CatHub fetch returned no data for any element in: %s", target_elements)

        effective_output = raw_output if raw_output else output_path
        write_table(df, effective_output)
        logger.info(
            "Saved cathub raw data (%d rows, %d elements) to %s",
            len(df),
            df["element"].nunique() if not df.empty else 0,
            effective_output,
        )
        return df

    if source_name == "mp":
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            raise RuntimeError("MP_API_KEY is not set. Use data=demo or export MP_API_KEY first.")
        raise NotImplementedError(
            "Materials Project fetching is intentionally left as a project-specific extension. "
            "Use the demo source now, then adapt this function to your query schema."
        )

    raise ValueError(f"Unknown data source: {source_name}")
