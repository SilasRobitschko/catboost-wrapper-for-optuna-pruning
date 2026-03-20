# optuna_catboost/utils.py

# Metrics where HIGHER is better (Maximization)
# CatBoost assumes most metrics (Logloss, CrossEntropy) are minimized.
MAXIMIZE_METRICS = {
    'AUC', 
    'Accuracy', 
    'Precision', 
    'Recall', 
    'F1', 
    'TotalF1', 
    'MCC',      
    'R2',       
    'NDCG', 
    'PFound', 
    'MAP', 
    'MRR', 
    'Hits'
}