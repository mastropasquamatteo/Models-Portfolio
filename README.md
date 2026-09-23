# ML Notebooks (2023)

Two end-to-end machine learning notebooks built in early 2023 on public Kaggle datasets: one classification, one regression. Each notebook goes from raw data to a saved scikit-learn pipeline, with the reasoning written next to the code.

| Project | Task | Final model | Result on held-out data |
|---|---|---|---|
| [Fetal Health Classification](Classification/Fetal%20Health%20Classification) | 3-class classification | Gradient Boosting (tuned) | weighted F1 0.918, accuracy 0.92 |
| [House Price Prediction](Regression/Houses%20Price%20Prediction) | Regression | XGBoost (tuned) | MAPE 9.3%, MAE about $16,400 |

## Fetal Health Classification

**Data.** 2,126 cardiotocography records, 21 numeric features, 3 classes (normal, suspect, pathological). Source: [Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

**Approach.**

- Classes are unbalanced, so every model is compared on weighted F1.
- A validation set is held out at the start and used only for the final check.
- 29 scikit-learn classifiers compared with default settings. Gradient Boosting and Histogram Gradient Boosting come out on top (F1 0.95 and 0.96 on the internal test split).
- 20-split cross-validation to choose between the two, then grid search on Gradient Boosting (best: 400 trees, learning rate 0.1, max depth 2; cross-validated F1 0.957).
- Learning curve to check whether more data would help.

**Result on the validation set (426 records).**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.94 | 0.98 | 0.96 |
| Suspect | 0.88 | 0.61 | 0.72 |
| Pathological | 0.84 | 0.89 | 0.86 |
| **Weighted avg** | **0.92** | **0.92** | **0.92** |

The weak spot is the *suspect* class: about 4 in 10 suspect cases are classified as something else.

## House Price Prediction

**Data.** Kaggle House Prices dataset (Ames, Iowa): 1,460 sales, 79 features. After dropping 7 mostly-empty columns and the rows with missing values, 1,338 records remain. Source: [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Approach.**

- Split into train (900), test (199) and validation (237) sets.
- Five models compared with default settings on the test set.
- Feature analysis: correlation with price, distributions, category ordering by mean price. Tried ordinal encoding against one-hot encoding: no real gain, so one-hot encoding was kept.
- Grid search on XGBoost and Random Forest, then a voting ensemble of the two. The ensemble did worse than XGBoost alone and was dropped.

**Results.**

| Model | MAPE (test) | MAE (test) |
|---|---|---|
| Decision Tree | 14.3% | $27,705 |
| Linear Regression | 12.7% | $22,979 |
| K-Nearest Neighbors | 12.6% | $24,394 |
| Random Forest (tuned) | 10.7% | $20,097 |
| XGBoost, default settings | 9.2% | $16,791 |
| Ensemble RF + XGBoost | 9.6% | $18,315 |
| **XGBoost (tuned)** | **9.0%** | **$17,095** |

On the validation set the tuned XGBoost scores MAPE 9.3% and MAE about $16,400, in line with the test set.

## What I would do differently today

I am leaving the notebooks as they were written, but a few choices would not pass my own review now:

- **Scaling (classification).** The scaler is re-fitted on the test and validation data instead of being fitted on the training data only and reused. The numbers above are probably a little optimistic because of this.
- **Tuning gain (regression).** Tuning improved MAPE by about 0.2 points over default XGBoost on a 199-row test set. That is within the noise; the honest conclusion is that default XGBoost was already about as good as this setup gets.
- **Split (regression).** Train, test and validation are taken by row position without shuffling or cross-validation. Fine as a first pass, weak as evidence.
- **Missing values (regression).** Dropping every row with a missing value is simple, but it throws away about 8% of the data. Imputation would be the better choice.

## Repository contents

Each folder contains the notebook and the exported pipeline (`.joblib`). The datasets are not included: download them from the Kaggle links above and place the CSV files in the same folder as the notebook.

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter.
