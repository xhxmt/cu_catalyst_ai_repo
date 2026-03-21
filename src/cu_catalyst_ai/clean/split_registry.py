from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def assign_splits(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    max_samples_per_element: int | None = 200,
) -> pd.DataFrame:
    """Assign train/test splits with optional per-element cap and stratification.

    Caps each element's row count first to balance the distribution (e.g. Cu
    dominates at 87 % of records), then stratifies the train/test split by
    element so every metal appears in both subsets.

    Args:
        df: Input DataFrame. Must contain ``element`` and ``catalyst_id`` columns.
        seed: Random state for reproducibility of both sampling and splitting.
        test_size: Fraction held out for the test set.
        max_samples_per_element: If set, downsample each element to at most
            this many rows before splitting.  ``None`` disables downsampling
            (original behaviour, random split without stratification).

    Returns:
        Copy of *df* (possibly downsampled) with a ``split`` column set to
        ``"train"`` or ``"test"``.
    """
    out = df.copy()

    # --- Per-element downsampling (breaks "Cu dominance") ---
    if max_samples_per_element is not None and "element" in out.columns:
        out = (
            out.groupby("element", group_keys=False)
            .apply(
                lambda g: g.sample(n=min(len(g), max_samples_per_element), random_state=seed),
                include_groups=True,
            )
            .reset_index(drop=True)
        )

    # --- Stratified train/test split ---
    stratify_col = out["element"] if "element" in out.columns else None
    train_ids, test_ids = train_test_split(
        out["catalyst_id"],
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col,
    )
    out["split"] = "train"
    out.loc[out["catalyst_id"].isin(test_ids), "split"] = "test"
    return out
