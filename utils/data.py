
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42


def load_train(task: str):
    """Load the labeled training dataset."""

    if task not in ("image", "sentiment"):
        raise ValueError("task must be 'image' or 'sentiment'")

    df = pd.read_csv(DATA_DIR / f"{task}_train.csv")

    feature_cols = [col for col in df.columns if col.startswith("feature_")]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    return X, y


def load_split(task: str, val_size=0.2, random_state=SEED):
    """Create a reproducible stratified train/validation split."""

    X, y = load_train(task)

    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )


def load_test(task: str):
    """Load unlabeled Kaggle test features and their IDs."""

    if task not in ("image", "sentiment"):
        raise ValueError("task must be 'image' or 'sentiment'")

    df = pd.read_csv(DATA_DIR / f"{task}_test.csv")

    feature_cols = [col for col in df.columns if col.startswith("feature_")]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    ids = df["id"].to_numpy()

    return X, ids