"""Run an additional reproducible 48-tree experiment (seed 42)."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from models.random_forest import RandomForest
from run_sentiment_trees import stratified_split, metrics

def main():
    df = pd.read_csv('data/sentiment_train.csv')
    columns = [f'feature_{i}' for i in range(768)]
    X = df[columns].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy(dtype=np.int32)
    train, val = stratified_split(y, seed=42)
    model = RandomForest(n_estimators=48, max_depth=12, min_samples_leaf=20,
                         max_features='sqrt', max_bins=24, max_samples=.7,
                         random_state=42)
    start = time.perf_counter()
    model.fit(X[train], y[train])
    summary = metrics(y[val], model.predict(X[val]))
    summary['training_accuracy'] = float(np.mean(model.predict(X[train]) == y[train]))
    summary['elapsed_seconds'] = round(time.perf_counter() - start, 2)
    summary['n_train'] = len(train)
    summary['n_validation'] = len(val)
    summary['parameters'] = {'n_estimators':48,'max_depth':12,'min_samples_leaf':20,
        'max_features':'sqrt','max_bins':24,'max_samples':.7,'random_state':42}
    target = Path('results/sentiment_trees/optimized_search_reproduced.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary,indent=2))
    print(f"Validation accuracy {summary['accuracy']:.4%}; macro F1 {summary['macro_f1']:.4f}")
if __name__ == '__main__': main()
