# COMP 472 NN | Team 17

## Project Overview

Compare four classification algorithms on two datasets of pre-extracted numerical features:

- **Image Recognition:** 500 features, 50 classes
- **Sentiment Analysis:** 768 features, 3 classes

**Models:** k-NN, Decision Tree, Random Forest (from scratch), and Neural Network (PyTorch).

**Goal:** Compare model performance, tune hyperparameters, analyze errors, and maximize Kaggle accuracy. One clustering experiment is also required.

## Project Structure

- `data/` - Kaggle CSV datasets (local only).
- `notebooks/` - One notebook per model and dataset, evaluation for each, showing how its hyperparameters were chosen - Also contains the best hyperparams found with the model and an evaluation
- `models/` - Reusable algorithm implementations for both datasets.
- `utils/` - Data loading, preprocessing, and evaluation helpers.
- `run_experiment.py` - Train, validate, and compare models.
- `clustering.py` - Unsupervised learning experiment.
- `results/` - Experimental results and plots - refer to the notebooks they contain the complete results for our experiments in the ipynb
- `generate_submission.py` - Generate final Kaggle predictions.
- `submissions/` - Final prediction CSV.
- `report/` and `presentation/` - Course deliverables.

## Setup

**Python 3.13 required.**

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   py -3.13 -m venv .venv
   ```

3. Install dependencies:

   ```bash
   .venv/Scripts/python -m pip install -r requirements.txt
   ```

4. Download the five Kaggle CSV files into `data/`.

## Running and Viewing Our Work

### Reproduce our Kaggle submission

From the repository root, run:

```bash
.venv/Scripts/python generate_submission.py
```

This trains our two best models on all labeled data and writes `submissions/submission.csv`, the file we submitted. It contains all 10,000 test predictions.

### Our submitted models

| Dataset | Submitted model | Implementation | Development notebook |
|---|---|---|---|
| Image | k-NN | `models/knn.py` | `notebooks/image_knn.ipynb` |
| Sentiment | Neural Network | `models/neural_network.py` | `notebooks/sentiment_nn.ipynb` |

- To see how our **Image** model was developed, view `notebooks/image_knn.ipynb`.
- To see how our **Sentiment** model was developed, view `notebooks/sentiment_nn.ipynb`.

The final hyperparameters from these two notebooks are the ones hard-coded in `generate_submission.py`.

### All notebooks

| To see our implementation of... | View |
|---|---|
| k-NN on the Image dataset (submitted) | `notebooks/image_knn.ipynb` |
| Neural Network on the Image dataset | `notebooks/image_nn.ipynb` |
| Decision Tree on the Image dataset | `notebooks/image_decision_tree.ipynb` |
| Random Forest on the Image dataset | `notebooks/image_random_forest.ipynb` |
| Neural Network on the Sentiment dataset (submitted) | `notebooks/sentiment_nn.ipynb` |
| k-NN on the Sentiment dataset | `notebooks/sentiment_knn.ipynb` |
| Decision Tree on the Sentiment dataset | `notebooks/sentiment_decision_tree.ipynb` |
| Clustering on the Sentiment dataset | `notebooks/sentiment_clustering.ipynb` |

Random Forest results on the Sentiment dataset are in `report/SENTIMENT_TREE_FINDINGS.md`.

### Reading the notebooks

- **The notebooks are already run.** Their outputs are saved, so you can read the results without running anything.
- Each model notebook changes one hyperparameter at a time and carries the best value into the next step.
- Each model notebook ends with the final model's results on the validation set:
  - a **confusion matrix**;
  - **precision, recall, F1-score and support** for every class;
  - overall **accuracy**, **macro average** and **weighted average**.

## Workflow

1. Explore datasets in `notebooks/`.
2. Implement models with a common `fit()` / `predict()` interface.
3. Run experiments with an 80/20 stratified train/validation split (seed 42).
4. Compare validation accuracy and tune hyperparameters.
5. Retrain selected models on full labeled data.
6. Generate `submissions/submission.csv` containing all 10,000 test predictions.

## Collaboration

Work on separate feature branches and submit pull requests to `main`. Do not commit datasets, virtual environments, or generated predictions.