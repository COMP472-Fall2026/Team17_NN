from pathlib import Path

import numpy as np
import pandas as pd

from models.knn import KNN
from models.neural_network import NeuralNetwork
from utils.data import DATA_DIR, load_train, load_test
from utils.preprocessing import make_preprocessor


def predict_task(task):

    # Train on all labeled examples before predicting the test labels
    x_train, y_train = load_train(task)
    x_test, test_ids = load_test(task)

    if task == "image":
        print(
            "Images: k-NN, k=1, cosine, sqrt_l2 scaling, density penalty=0.5",
            flush=True,
        )
        preprocessor = make_preprocessor("sqrt_l2")
        model = KNN(
            k=1,
            distance="cosine",
            density_penalty=0.5,
            density_neighbors=20,
        )
    else:
        print(
            "Sentiment: neural network, 256 hidden, dropout=0.7, "
            "weight decay=0.01, lr=0.0003, 40 epochs",
            flush=True,
        )
        preprocessor = make_preprocessor("standard")
        model = NeuralNetwork(
            epochs=40,
            lr=0.0003,
            batch_size=128,
            hidden=(256,),
            dropout=0.7,
            input_dropout=0.0,
            weight_decay=0.01
        )

    # Learn scaling from training data only
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    # Copies keep the arrays writable for PyTorch
    model.fit(x_train.copy(), y_train)
    predictions = model.predict(x_test.copy())

    if len(test_ids) != 5000 or len(predictions) != 5000:
        raise ValueError(f"Expected 5,000 {task} predictions")
    if pd.isna(test_ids).any() or len(np.unique(test_ids)) != 5000:
        raise ValueError(f"Invalid or duplicate {task} test IDs")

    max_label = 49 if task == "image" else 2
    if not np.isin(predictions, np.arange(max_label + 1)).all():
        raise ValueError(f"Invalid {task} labels")

    return pd.DataFrame({"id": test_ids, "label": predictions.astype(int)})


def generate_submission():

    image_predictions = predict_task("image")
    sentiment_predictions = predict_task("sentiment")
    predictions = pd.concat(
        [image_predictions, sentiment_predictions], ignore_index=True
    )

    # Match predictions by ID, not by the order of the test CSVs
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if (
        len(sample) != 10000
        or sample["id"].isna().any()
        or sample["id"].duplicated().any()
    ):
        raise ValueError("Expected 10,000 unique template IDs")
    if predictions["id"].duplicated().any() or set(sample["id"]) != set(
        predictions["id"]
    ):
        raise ValueError("Prediction IDs do not match the template")

    submission = sample[["id"]].merge(
        predictions, on="id", how="left", validate="one_to_one", sort=False
    )
    if submission["label"].isna().any():
        raise ValueError("Some predictions are missing")
    submission["label"] = submission["label"].astype(int)

    output_path = Path(__file__).resolve().parent / "submissions" / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to: {output_path}")


if __name__ == "__main__":
    generate_submission()
