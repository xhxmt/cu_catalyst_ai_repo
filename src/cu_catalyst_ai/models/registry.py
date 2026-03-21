from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_name: str, random_state: int = 42, params: dict | None = None):
    """Build and return a model instance by name.

    For GPR the model is wrapped in a ``Pipeline([scaler, gpr])`` so that
    features are always standardised before kernel optimisation.  RF and
    XGBoost are returned unwrapped (tree models are scale-invariant).

    Parameters
    ----------
    model_name:
        One of ``"linear"``, ``"rf"``, ``"xgb"``, ``"gpr"``.
    random_state:
        Integer seed forwarded to RF and XGBoost.
    params:
        Extra keyword arguments merged into the base model constructor.
        For GPR, params are passed to *GaussianProcessRegressor* directly.
    """
    params = params or {}
    if model_name == "linear":
        return LinearRegression(**params)
    if model_name == "rf":
        return RandomForestRegressor(random_state=random_state, **params)
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc
        return XGBRegressor(random_state=random_state, objective="reg:squarederror", **params)
    if model_name == "gpr":
        # RBF + WhiteKernel: smooth signal kernel + independent noise.
        # n_restarts_optimizer=5 reduces the risk of converging to a local
        # optimum during log-marginal-likelihood optimisation (L-BFGS).
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=random_state,
            **params,
        )
        # Wrap in a Pipeline so that StandardScaler is applied automatically.
        # GPR is sensitive to feature magnitude; RF/XGBoost are not.
        return Pipeline([("scaler", StandardScaler()), ("gpr", gpr)])
    raise ValueError(f"Unknown model: {model_name}")
