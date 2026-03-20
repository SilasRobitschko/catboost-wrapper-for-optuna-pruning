# optuna_catboost/__init__.py

from .classifier import OptunaCatBoostClassifier
# from .regressor import OptunaCatBoostRegressor  <-- Uncomment when ready

__all__ = [
    "OptunaCatBoostClassifier",
    # "OptunaCatBoostRegressor",
]