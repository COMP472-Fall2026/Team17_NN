# Sentiment classification: Decision Tree and Random Forest

## What we tried

For this part of the project, we compared a single Decision Tree with a Random Forest on the sentiment dataset. The dataset contains 12,000 labelled samples, with 768 numerical features per sample (precomputed BERT embeddings). The labels are 0 (negative), 1 (neutral), and 2 (positive).

We kept the same 80/20 stratified split for every run (seed 42): 9,600 training samples and 2,400 validation samples. This makes the comparisons easier to interpret. The validation data was kept out of training, and the split thresholds were calculated from the training data only.

Both classifiers are implemented in NumPy. The tree selects feature thresholds using Gini impurity (entropy is also supported); to keep CPU training manageable, it considers thresholds based on quantile bins instead of checking every unique feature value. The forest trains trees on bootstrap samples, uses random feature subsets at splits, and averages class probabilities for prediction. We did not use a prebuilt tree or forest classifier.

## Results

These numbers are from the local validation set, **not** from the Kaggle leaderboard.

| Model settings | Training accuracy | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|
| Decision Tree: depth 8, min leaf 12, 64 features/node | 61.12% | 46.42% | 46.22% |
| Decision Tree: depth 16, min leaf 8, 64 features/node | 77.90% | 44.42% | 44.47% |
| Random Forest: 12 trees, depth 10, min leaf 30 | 73.73% | 55.00% | 54.68% |
| Random Forest: 24 trees, depth 12, min leaf 25 | 80.15% | 56.54% | 56.28% |
| Random Forest: 48 trees, depth 12, min leaf 20, sqrt features/node | 85.80% | **59.04%** | **58.77%** |

The 12- and 24-tree experiments used a bootstrap fraction of 0.7. The 48-tree run also used a 0.7 bootstrap fraction and 24 quantile bins. Other settings differ between these runs, so the table should not be read as an isolated experiment on the number of trees.

Full metrics from the initial four runs are saved in `results/sentiment_trees/validation_results.json`. The additional forest run is documented in `results/sentiment_trees/optimized_search.json`.

## What stood out

The deeper single tree fitted the training data much better, but it actually did worse on validation data. That's a sign that the extra depth was learning details that didn't carry over to unseen examples.

The forests did better than either single tree. Our best completed run reached 59.04% validation accuracy with 48 trees. It's an improvement over the first 24-tree result (56.54%), but the training score is still far above the validation score. Overfitting remains an issue.

One possible reason is the feature representation. Each sentiment sample has 768 dense embedding features, while our trees make one-feature-at-a-time splits. A different classifier could make better use of relationships between those features. We would need to compare against the team's k-NN and neural network results before drawing a firm conclusion.

## Limitations and next steps

- The tree checks quantile-based split candidates rather than every possible threshold. This saves training time but may miss some useful splits.
- The results are from one stratified split. Running additional seeds would give a better sense of how stable the rankings are.
- The experiments changed more than one parameter between some configurations. A more controlled sweep would vary one setting at a time.
- A longer forest search was attempted but did not finish within the available runtime, so it has no reported result.
- We cannot know private Kaggle accuracy from validation results. The team's final submission should use whichever **permitted** model performs best on independent validation.

## Running the experiments

Place `sentiment_train.csv` (and, for prediction, `sentiment_test.csv`) in `data/`.

```bash
python -m unittest discover -s tests -v
python run_sentiment_trees.py --data data/sentiment_train.csv
python benchmark_improve.py
```

To fit the selected model to the full labelled sentiment dataset and generate sentiment-only predictions:

```bash
python run_sentiment_trees.py --final-fit --data data/sentiment_train.csv --test data/sentiment_test.csv
```

This output is **not** the complete Kaggle submission. The sentiment predictions still need to be combined by sample ID with image predictions and the official submission template.

