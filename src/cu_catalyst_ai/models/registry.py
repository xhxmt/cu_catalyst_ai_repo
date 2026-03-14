from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression


def build_model(model_name: str, random_state: int = 42, params: dict | None = None):
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
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    raise ValueError(f"Unknown model: {model_name}")
