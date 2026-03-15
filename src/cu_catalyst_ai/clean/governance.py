"""Governance helpers: split accepted rows from rows flagged for review.

Usage pattern::

    from cu_catalyst_ai.clean.governance import flag_rows, split_good_review

    df = flag_rows(df, mask_bad, reason="missing provenance", stage="provenance")
    clean_df, review_df = split_good_review(df)

Rows that accumulate multiple flags retain the *first* reason assigned.
``review_stage`` records which governance stage caught the row first.
"""

from __future__ import annotations

import pandas as pd

REVIEW_REASON_COL = "review_reason"
REVIEW_STAGE_COL = "review_stage"


def flag_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    stage: str,
) -> pd.DataFrame:
    """Mark rows selected by *mask* with a review reason and stage.

    Only rows that are not already flagged receive the reason/stage, so the
    *first* governance check that catches a row wins the attribution.

    Args:
        df: DataFrame that may already have ``review_reason`` / ``review_stage``
            columns from earlier calls.
        mask: Boolean Series aligned with *df* index, True → flag this row.
        reason: Human-readable description of why the row is isolated.
        stage: Short identifier for the governance stage (e.g. ``"provenance"``).

    Returns:
        A copy of *df* with ``review_reason`` and ``review_stage`` columns
        updated for the flagged rows.
    """
    out = df.copy()
    # Ensure columns exist
    if REVIEW_REASON_COL not in out.columns:
        out[REVIEW_REASON_COL] = pd.NA
    if REVIEW_STAGE_COL not in out.columns:
        out[REVIEW_STAGE_COL] = pd.NA

    # Only stamp rows not already flagged (first check wins)
    unflagged = mask & out[REVIEW_REASON_COL].isna()
    out.loc[unflagged, REVIEW_REASON_COL] = reason
    out.loc[unflagged, REVIEW_STAGE_COL] = stage
    return out


def split_good_review(
    df: pd.DataFrame,
    reason_col: str = REVIEW_REASON_COL,
    stage_col: str = REVIEW_STAGE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into clean rows and review-flagged rows.

    Args:
        df: DataFrame that may contain ``review_reason`` / ``review_stage``
            columns produced by :func:`flag_rows`.
        reason_col: Name of the review-reason column.
        stage_col: Name of the review-stage column.

    Returns:
        ``(clean_df, review_df)`` where *clean_df* has no ``review_reason``
        column and *review_df* retains both annotation columns.
    """
    if reason_col not in df.columns:
        # No flags at all — everything is clean
        return df.copy(), pd.DataFrame(columns=df.columns)

    flagged = df[reason_col].notna()
    clean_df = (
        df[~flagged].drop(columns=[reason_col, stage_col], errors="ignore").reset_index(drop=True)
    )
    review_df = df[flagged].reset_index(drop=True)
    return clean_df, review_df
