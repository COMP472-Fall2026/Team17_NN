
# COMP 472 NN | Team 17

## Project Overview

Compare four classification algorithms on two datasets of pre-extracted numerical features:

- **Image Recognition:** 500 features, 50 classes
- **Sentiment Analysis:** 768 features, 3 classes

**Models:** k-NN, Decision Tree, Random Forest (from scratch), and Neural Network (PyTorch).

**Goal:** Compare model performance, tune hyperparameters, analyze errors, and maximize Kaggle accuracy. One clustering experiment is also required.

## Project Structure

- `data/` — Kaggle CSV datasets (local only).
- `notebooks/` — EDA and visualizations for each dataset.
- `models/` — Reusable algorithm implementations for both datasets.
- `utils/` — Data loading, preprocessing, and evaluation helpers.
- `run_experiment.py` — Train, validate, and compare models.
- `clustering.py` — Unsupervised learning experiment.
- `results/` — Experimental results and plots.
- `generate_submission.py` — Generate final Kaggle predictions.
- `submissions/` — Final prediction CSV.
- `report/` and `presentation/` — Course deliverables.

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

## Workflow

1. Explore datasets in `notebooks/`.
2. Implement models with a common `fit()` / `predict()` interface.
3. Run experiments with an 80/20 stratified train/validation split (seed 42).
4. Compare validation accuracy and tune hyperparameters.
5. Retrain selected models on full labeled data.
6. Generate `submissions/submission.csv` containing all 10,000 test predictions.

**Note:** Training and submission scripts are initially placeholders.

## Collaboration

Work on separate feature branches and submit pull requests to `main`. Do not commit datasets, virtual environments, or generated predictions.
