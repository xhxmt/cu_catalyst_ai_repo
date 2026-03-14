from __future__ import annotations

import os

import numpy as np
import pandas as pd

from cu_catalyst_ai.utils.io import write_table
from cu_catalyst_ai.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_demo_dataset(n_samples: int = 240, seed: int = 42) -> pd.DataFrame:
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


def fetch_data(
    source_name: str, output_path: str, n_samples: int = 240, seed: int = 42
) -> pd.DataFrame:
    if source_name == "demo":
        df = generate_demo_dataset(n_samples=n_samples, seed=seed)
        write_table(df, output_path)
        logger.info("Saved demo dataset to %s", output_path)
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
