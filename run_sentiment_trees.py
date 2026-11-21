"""Reproducible Decision Tree / Random Forest experiments for sentiment features.

Usage: python run_sentiment_trees.py --data data/sentiment_train.csv
Optional: --quick to check the pipeline on a stratified subset before full runs.
Only provided training data are used; no pretrained classifier is imported.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest

SEED = 42


def stratified_split(y, validation_fraction=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for label in np.unique(y):
        indices = rng.permutation(np.where(y == label)[0])
        cut = max(1, int(round(len(indices) * validation_fraction)))
        va.extend(indices[:cut]); tr.extend(indices[cut:])
    return np.array(tr), np.array(va)


def metrics(y_true, pred):
    labels = np.unique(np.concatenate((y_true, pred)))
    conf = np.zeros((len(labels), len(labels)), dtype=int)
    mapping = {v:i for i,v in enumerate(labels)}
    for a,b in zip(y_true,pred):
        conf[mapping[a],mapping[b]] += 1
    per_class = {}
    for i,label in enumerate(labels):
        tp = conf[i,i]
        precision = tp / conf[:,i].sum() if conf[:,i].sum() else 0.0
        recall = tp / conf[i,:].sum() if conf[i,:].sum() else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[str(int(label))] = dict(precision=round(precision,5),
                                         recall=round(recall,5), f1=round(f1,5),
                                         support=int(conf[i,:].sum()))
    return {"accuracy":float(np.mean(pred==y_true)),
            "macro_f1":float(np.mean([v['f1'] for v in per_class.values()])),
            "classes":labels.astype(int).tolist(),
            "confusion_matrix":conf.tolist(),"per_class":per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='data/sentiment_train.csv')
    parser.add_argument('--output',default='results/sentiment_trees')
    parser.add_argument('--quick',action='store_true',help='Run fast smoke benchmark on 20% stratified samples')
    parser.add_argument('--final-fit',action='store_true',help='Retrain winning model on full dataset and write test predictions')
    parser.add_argument('--test',default='data/sentiment_test.csv')
    args=parser.parse_args()
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data)
    features=[f'feature_{i}' for i in range(768)]
    missing=set(features+['id','label'])-set(df.columns)
    if missing: raise ValueError(f'missing columns: {sorted(missing)}')
    X=df[features].to_numpy(dtype=np.float32)
    y=df.label.to_numpy(dtype=np.int32)
    if args.quick:
        keep,_=stratified_split(y,validation_fraction=0.8)
        X,y=X[keep],y[keep]
    train_idx,val_idx=stratified_split(y)
    Xtrain,ytrain=X[train_idx],y[train_idx]
    Xval,yval=X[val_idx],y[val_idx]
    print(f'Training {len(train_idx)}, validation {len(val_idx)}, seed {SEED}',flush=True)
    specs=[('tree_depth_8', DecisionTree(max_depth=8,min_samples_leaf=12,max_features=64,random_state=SEED)),
           ('tree_depth_16',DecisionTree(max_depth=16,min_samples_leaf=8,max_features=64,random_state=SEED)),
           ('forest_12',RandomForest(n_estimators=12,max_depth=10,min_samples_leaf=30,max_samples=0.7,max_bins=24,random_state=SEED)),
           ('forest_24',RandomForest(n_estimators=24,max_depth=12,min_samples_leaf=25,max_samples=0.7,max_bins=24,random_state=SEED)),
           ('forest_48_tuned',RandomForest(n_estimators=48,max_depth=12,min_samples_leaf=20,max_features='sqrt',max_samples=0.7,max_bins=24,random_state=SEED))]
    if args.quick: specs=specs[:3]
    results=[]; best=None; best_accuracy=-1
    for name,model in specs:
        start=time.perf_counter()
        model.fit(Xtrain,ytrain)
        elapsed=time.perf_counter()-start
        train_pred=model.predict(Xtrain)
        val_pred=model.predict(Xval)
        info=metrics(yval,val_pred)
        info.update(name=name,train_accuracy=float(np.mean(train_pred==ytrain)),
                    fit_seconds=round(elapsed,2),hyperparameters={k:v for k,v in model.__dict__.items()
                    if k in ('max_depth','max_features','min_samples_leaf','min_samples_split','criterion','max_bins','n_estimators')})
        results.append(info)
        print(f'{name}: validation={info["accuracy"]:.4f}, training={info["train_accuracy"]:.4f}, '
              f'macro_F1={info["macro_f1"]:.4f}, time={elapsed:.1f}s',flush=True)
        if info['accuracy']>best_accuracy:
            best_accuracy=info['accuracy']; best=(name,model)
    (output/'validation_results.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame([{k:r[k] for k in ('name','accuracy','macro_f1','train_accuracy','fit_seconds')}
                  for r in results]).to_csv(output/'comparison.csv',index=False)
    print('Selected:', best[0],flush=True)
    if args.final_fit:
        test=pd.read_csv(args.test)
        if test.id.duplicated().any():raise ValueError('duplicate test ids')
        chosen=best[1]
        chosen.fit(df[features].to_numpy(dtype=np.float32),df.label.to_numpy(dtype=np.int32))
        pred=chosen.predict(test[features].to_numpy(dtype=np.float32))
        pd.DataFrame({'id':test.id,'label':pred.astype(int)}).to_csv(output/'sentiment_predictions.csv',index=False)
        print('Wrote predictions keyed by sentiment test IDs; NOT a full Kaggle submission.')

if __name__=='__main__':main()
