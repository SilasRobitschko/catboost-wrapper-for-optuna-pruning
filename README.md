# catboost-wrapper-for-optuna-pruning

A lightweight Python wrapper for CatBoost that enables seamless Optuna pruning on the GPU.

## The Problem
CatBoost is incredibly fast on the GPU, but it natively struggles with Optuna's `PruningCallback` during GPU training. Standard workarounds break CatBoost's internal logging, ruin the `best_iteration_` attribute, and mess up early stopping.

## The Solution
`OptunaCatBoostClassifier` wraps the native CatBoost model and trains it in discrete chunks. It seamlessly handles:
- **GPU Pruning:** Reports back to Optuna at regular intervals.
- **Continuous Logging:** Preserves the full `evals_result_` history across batches so your loss curves plot perfectly.
- **Accurate Best Iterations:** Manually tracks and overrides `best_iteration_` and `best_score_` so they represent the absolute best step across the entire training run.
- **Cross-Batch Early Stopping:** Implements custom early stopping that works accurately across batch boundaries.

## Installation
*(Add installation instructions here once published to PyPI, or instructions on how to install via `pip install git+https...`)*

## Quick Start

The wrapper functions exactly like the native `CatBoostClassifier`. Just pass your Optuna `trial` and the number of batches you want to wait between pruning checks (`pruning_batches`).

```python
import optuna
from optuna_catboost import OptunaCatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Dummy Data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

def objective(trial):
    param = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "task_type": "GPU",
        "eval_metric": "AUC",
        "random_seed": 42
    }

    # Initialize our wrapper
    model = OptunaCatBoostClassifier(
        trial=trial, 
        pruning_batches=100, # Checks pruning every 100 steps
        **param
    )

    # Fit normally
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        early_stopping_rounds=50,
        verbose=0
    )

    # best_score_ and best_iteration_ work seamlessly!
    return model.best_score_["validation"]["AUC"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)