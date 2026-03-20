# optuna_catboost/classifier.py

from catboost import CatBoostClassifier
import optuna
import warnings
from optuna_catboost import MAXIMIZE_METRICS

class OptunaCatBoostClassifier(CatBoostClassifier):
    def __init__(self, trial=None, pruning_batches=100, **kwargs):
        """
        CatBoostClassifier wrapper enabling Optuna pruning on the GPU 
        by training in discrete batches while preserving full logging.
        """
        self.trial = trial
        self.pruning_batches = pruning_batches
        
        # Initialize custom trackers to fix broken chunking metrics
        self._custom_best_iteration = None
        self._custom_best_score = None
        self._absolute_iterations_trained = 0
        self._custom_evals_result = {} # Stores full metric history across batches
        
        super().__init__(**kwargs)

    @property
    def best_iteration_(self):
        """Override native best_iteration_ to report the absolute best step."""
        if self._custom_best_iteration is not None:
            return self._custom_best_iteration
        return super().best_iteration_

    @property
    def best_score_(self):
        """Override native best_score_ to report the best score across all batches."""
        if self._custom_best_score is not None:
            return self._custom_best_score
        return super().best_score_

    @property
    def evals_result_(self):
        """Override native evals_result_ to return the accumulated full history."""
        if self._custom_evals_result:
            return self._custom_evals_result
        # Fallback to native if not batching
        try:
            return super().evals_result_
        except AttributeError:
            return {}

    def fit(self, X, y=None, eval_set=None, early_stopping_rounds=None, **kwargs):
        params = self.get_params()
        task_type = params.get("task_type", "CPU")
        
        # 1. Fallback to native behavior if pruning isn't needed/possible
        if self.trial is None or task_type != "GPU" or eval_set is None:
            if self.trial and eval_set is None:
                warnings.warn("Optuna pruning requires an eval_set. Falling back to standard continuous fit.")
            return super().fit(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, **kwargs)

        # 2. Setup Batching Variables
        total_iterations = params.get("iterations", 1000)
        
        # Determine metric direction (minimize vs maximize)
        raw_metric_name = params.get('eval_metric', 'Logloss') 
        base_metric_name = raw_metric_name.split(':')[0] # Strip params like :alpha=0.5
        
        is_maximize = base_metric_name in MAXIMIZE_METRICS
        best_eval_metric = float('-inf') if is_maximize else float('inf')
        patience_counter = 0
        val_name = None 
        
        # Reset custom history for a new fit call
        self._custom_evals_result = {}
        self._absolute_iterations_trained = 0
        
        # 3. The Batching Loop
        for start_idx in range(0, total_iterations, self.pruning_batches):
            current_batch_size = min(self.pruning_batches, total_iterations - start_idx)
            
            # Temporarily set iterations for this chunk
            self.set_params(iterations=current_batch_size)
            
            # Pass init_model if it's not the first batch
            init_model = self if start_idx > 0 else None
            
            # Suppress native early stopping; handle manually across chunks
            super().fit(X, y, eval_set=eval_set, init_model=init_model, early_stopping_rounds=None, **kwargs)

            # --- 4. Accumulate ALL metrics (Train and Eval) ---
            for dataset_name, metrics in super().evals_result_.items():
                if dataset_name not in self._custom_evals_result:
                    self._custom_evals_result[dataset_name] = {}
                    
                for metric_name, metric_values in metrics.items():
                    if metric_name not in self._custom_evals_result[dataset_name]:
                        self._custom_evals_result[dataset_name][metric_name] = []
                    self._custom_evals_result[dataset_name][metric_name].extend(metric_values)

            # Determine validation dictionary key dynamically on the first run
            if val_name is None:
                eval_keys = [k for k in super().evals_result_.keys() if k != 'learn']
                val_name = eval_keys[0] if eval_keys else "validation"

            # 5. Extract Metrics for Optuna
            metric_dict = super().evals_result_.get(val_name, {})
            if not metric_dict or raw_metric_name not in metric_dict:
                warnings.warn(f"Metric {raw_metric_name} not found in evals_result_. Stopping batching.")
                break 
                
            chunk_metrics = metric_dict[raw_metric_name]
            
            # 6. Update Custom Trackers & Early Stopping
            for i, score in enumerate(chunk_metrics):
                absolute_step = start_idx + i
                improved = (score > best_eval_metric) if is_maximize else (score < best_eval_metric)
                
                if improved: 
                    best_eval_metric = score
                    self._custom_best_score = {val_name: {raw_metric_name: score}}
                    self._custom_best_iteration = absolute_step
                    patience_counter = 0
                else:
                    patience_counter += 1

            last_score = chunk_metrics[-1]
            self._absolute_iterations_trained += current_batch_size

            # 7. Report to Optuna
            self.trial.report(last_score, self._absolute_iterations_trained)

            # 8. Pruning Check
            if self.trial.should_prune():
                self.set_params(iterations=total_iterations) # Restore state
                raise optuna.TrialPruned()

            # 9. Custom Early Stopping Check
            if early_stopping_rounds and patience_counter >= early_stopping_rounds:
                print(f"Stopped early at iteration {self._absolute_iterations_trained}")
                break

        # Restore original iterations parameter before returning to the user
        self.set_params(iterations=total_iterations)
        return self