from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate

logger = logging.getLogger(__name__)


def run_cv(
    model,
    X,
    y,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Run cross-validation and return a one-row summary DataFrame.

    When *groups* is provided (and contains ≥2 unique values), uses
    :class:`~sklearn.model_selection.GroupKFold` so that all rows belonging
    to the same element stay in the same fold (Left-One-Element-Out style).
    This prevents optimistic R² caused by different facets of the same metal
    appearing in both train and test.

    Falls back to plain :class:`~sklearn.model_selection.KFold` when:
    - *groups* is ``None`` (no element column in dataset), or
    - fewer than 2 unique groups exist (e.g. single-metal demo datasets).

    Args:
        model: Scikit-learn estimator or Pipeline.
        X: Feature matrix (array-like or DataFrame).
        y: Target vector.
        n_splits: Number of CV folds.
        shuffle: Whether to shuffle before splitting (KFold only; GroupKFold
            has no ``shuffle`` parameter).
        random_state: Random seed for KFold shuffle (ignored for GroupKFold).
        groups: Group labels for each sample; typically the ``element`` column.
            Must have the same length as *y*.

    Returns:
        One-row :class:`~pandas.DataFrame` with columns
        ``mae_mean``, ``mae_std``, ``rmse_mean``, ``rmse_std``,
        ``r2_mean``, ``r2_std``.
    """
    use_group = False
    if groups is not None:
        n_unique = groups.nunique()
        if n_unique < 2:
            logger.warning(
                "GroupKFold skipped: only %d unique group(s) found in 'element'; "
                "falling back to KFold (shuffle=%s, random_state=%d).",
                n_unique,
                shuffle,
                random_state,
            )
        else:
            actual_splits = min(n_splits, n_unique)
            if actual_splits < n_splits:
                logger.warning(
                    "GroupKFold requested n_splits=%d but only %d unique groups found; "
                    "using n_splits=%d instead.",
                    n_splits,
                    n_unique,
                    actual_splits,
                )
            cv = GroupKFold(n_splits=actual_splits)
            use_group = True
            # Log fold composition for transparency
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
                test_groups = groups.iloc[test_idx].unique().tolist()
                logger.info(
                    "GroupKFold fold %d/%d — test elements: %s  (train=%d, test=%d)",
                    fold_idx + 1,
                    actual_splits,
                    sorted(test_groups),
                    len(train_idx),
                    len(test_idx),
                )

    if not use_group:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    cv_kwargs = {"groups": groups} if use_group else {}

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
        **cv_kwargs,
    )
    df = pd.DataFrame(scores)
    for col in ["test_mae", "test_rmse"]:
        df[col] = -df[col]
    summary = {
        "mae_mean": float(df["test_mae"].mean()),
        "mae_std": float(df["test_mae"].std(ddof=0)),
        "rmse_mean": float(df["test_rmse"].mean()),
        "rmse_std": float(df["test_rmse"].std(ddof=0)),
        "r2_mean": float(df["test_r2"].mean()),
        "r2_std": float(df["test_r2"].std(ddof=0)),
    }
    return pd.DataFrame([summary])


def run_cv_with_centering(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    adsorbates: pd.Series,
    n_splits: int = 5,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """GroupKFold CV with per-fold per-adsorbate target centering.

    Each fold computes adsorbate means **only from that fold's training rows**,
    subtracts them before fitting, and adds them back before evaluation.
    This prevents O's absolute energy scale (−6 to −2 eV) from dominating the
    MSE landscape and allows the model to focus on relative trends within each
    adsorbate class.

    Returns a one-row summary DataFrame with both restored (original-unit) and
    centered-space metrics.  The centered metrics serve as diagnostics: large
    gaps between centered and restored R² indicate unstable train_means
    estimates (too few samples for some adsorbate in that fold).

    Args:
        model: Scikit-learn estimator.  Cloned fresh per fold.
        X: Feature matrix aligned with *y*.
        y: Raw (un-centered) target vector.
        adsorbates: Adsorbate label per row (e.g. ``"CO"``, ``"O"``, ``"OH"``).
            Must be the same length as *y*.
        n_splits: Number of folds for :class:`sklearn.model_selection.GroupKFold`.
            Capped at the number of unique groups when *groups* is provided.
        groups: Element labels for GroupKFold. When ``None`` falls back to
            standard KFold (useful for single-element datasets).

    Returns:
        One-row :class:`~pandas.DataFrame` with columns:
        ``mae_mean``, ``mae_std``, ``rmse_mean``, ``rmse_std``,
        ``r2_mean``, ``r2_std`` (all in original eV units, after restoring),
        plus ``centered_r2_mean``, ``centered_r2_std``,
        ``centered_mae_mean``, ``centered_mae_std`` (diagnostic).
    """
    # Determine splitter
    if groups is not None and groups.nunique() >= 2:
        actual_splits = min(n_splits, groups.nunique())
        if actual_splits < n_splits:
            logger.warning(
                "run_cv_with_centering: n_splits capped at %d (only %d unique elements).",
                actual_splits,
                groups.nunique(),
            )
        splitter = GroupKFold(n_splits=actual_splits)
        split_kwargs: dict = {"groups": groups}
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_kwargs = {}

    fold_results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, **split_kwargs)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_raw, y_te_raw = y.iloc[train_idx], y.iloc[test_idx]
        ads_tr = adsorbates.iloc[train_idx]
        ads_te = adsorbates.iloc[test_idx]

        # Compute per-adsorbate means from *training* rows only
        train_means: dict[str, float] = y_tr_raw.groupby(ads_tr).mean().to_dict()

        # Warn if any adsorbate has very few training samples
        for ads_label, count in ads_tr.value_counts().items():
            if count < 30:
                logger.warning(
                    "run_cv_with_centering fold %d: adsorbate '%s' has only %d "
                    "train samples; centering mean may be unstable.",
                    fold_idx,
                    ads_label,
                    count,
                )

        # Center training targets
        y_tr_c = y_tr_raw - ads_tr.map(train_means)

        # Center test targets using *train* means (no leakage)
        # Unknown adsorbates fall back to global train mean
        global_mean = float(np.mean(list(train_means.values())))
        y_te_c = y_te_raw - ads_te.map(train_means).fillna(global_mean)

        # Clone + fit on centered targets
        mdl = clone(model)
        mdl.fit(X_tr, y_tr_c)

        # Predict in centered space, then restore
        y_pred_c = pd.Series(mdl.predict(X_te), index=y_te_raw.index)
        restore_shift = ads_te.map(train_means).fillna(global_mean)
        y_pred_restored = y_pred_c + restore_shift

        # Restored-space metrics (primary)
        r2_restored = float(r2_score(y_te_raw, y_pred_restored))
        mae_restored = float(mean_absolute_error(y_te_raw, y_pred_restored))
        rmse_restored = float(np.sqrt(mean_squared_error(y_te_raw, y_pred_restored)))

        # Centered-space metrics (diagnostic)
        r2_centered = float(r2_score(y_te_c, y_pred_c))
        mae_centered = float(mean_absolute_error(y_te_c, y_pred_c))

        test_els = sorted(groups.iloc[test_idx].unique().tolist()) if groups is not None else []
        logger.info(
            "Centering CV fold %d: test_elements=%s  R2_restored=%.3f  "
            "MAE_restored=%.4f  R2_centered=%.3f  MAE_centered=%.4f",
            fold_idx,
            test_els,
            r2_restored,
            mae_restored,
            r2_centered,
            mae_centered,
        )

        fold_results.append(
            {
                "r2": r2_restored,
                "mae": mae_restored,
                "rmse": rmse_restored,
                "r2_centered": r2_centered,
                "mae_centered": mae_centered,
            }
        )

    fdf = pd.DataFrame(fold_results)
    summary = {
        "mae_mean": float(fdf["mae"].mean()),
        "mae_std": float(fdf["mae"].std(ddof=0)),
        "rmse_mean": float(fdf["rmse"].mean()),
        "rmse_std": float(fdf["rmse"].std(ddof=0)),
        "r2_mean": float(fdf["r2"].mean()),
        "r2_std": float(fdf["r2"].std(ddof=0)),
        "centered_r2_mean": float(fdf["r2_centered"].mean()),
        "centered_r2_std": float(fdf["r2_centered"].std(ddof=0)),
        "centered_mae_mean": float(fdf["mae_centered"].mean()),
        "centered_mae_std": float(fdf["mae_centered"].std(ddof=0)),
    }
    return pd.DataFrame([summary])
