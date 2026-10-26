# Titanic Survival Prediction

## Problem

Binary classification: predict whether a passenger survived the Titanic disaster, using the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data) (`train.csv`, 891 rows).

## Workflow

1. **Load & inspect** — `df.info()`, `df.describe()`, `df.isnull().sum()` to check shape, types, and missing values.
2. **EDA** — count plots of `Survived` against `Sex`, `Pclass`, and `Embarked`; an age histogram; a fare boxplot by survival; and a correlation heatmap of numeric features.
3. **Feature engineering:**
   - `Title` extracted from `Name` (grouped into Mr/Mrs/Miss/Master/Rare)
   - `FamilySize` = `SibSp + Parch + 1`
   - `IsAlone` — binary flag when `FamilySize == 1`
   - `HasCabin` — binary flag for whether `Cabin` was recorded, instead of dropping the column
   - `Age` missing values filled with the median **per (Pclass, Title) group** rather than a single global median
   - `Embarked` filled with the mode; `Fare` filled with the median
   - `AgeBand` / `FareBand` — binned versions of Age and Fare (created but not currently used in the feature set — see Notes)
4. **Encoding** — `LabelEncoder` applied to `Sex`, `Embarked`, `Title`.
5. **Features used for modeling:** `Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title, FamilySize, IsAlone, HasCabin`
6. **Split** — 80/20 train/test, stratified on `Survived`.
7. **Model comparison** — 5-fold cross-validation on the training set across five models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN.
8. **Tuning** — `GridSearchCV` on `RandomForestClassifier`, searching `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
9. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, classification report, and confusion matrix on the held-out test set.
10. **Feature importance** — bar plot from the tuned Random Forest's `feature_importances_`.

## Results

*(Run the notebook and paste your actual output here — this section is intentionally blank rather than filled with placeholder numbers, since these weren't in the executed notebook.)*

| Model | CV Accuracy |
|---|---|
| Logistic Regression |0.8048 (+/- 0.0248)|
| Decision Tree |0.7614 (+/- 0.0454)|
| Random Forest |0.8063 (+/- 0.0337)|
| SVM |0.6939 (+/- 0.0179)|
| KNN |0.7290 (+/- 0.0354)|


**Test set metrics:**

| Metric | Score |
|---|---|
| Accuracy |0.8044692737430168|
| Precision |0.765625|
| Recall |0.7101449275362319|
| F1 |0.7368421052631579|
| ROC-AUC |0.8555335968379447|

**Top features by importance:** (from the `feature_importances_` bar plot)

## Notes / Next Steps

- `AgeBand` and `FareBand` are engineered in the notebook but not included in the final `features` list — worth testing whether adding them helps tree-based models or hurts Logistic Regression/SVM, as noted in the code comment.
- No comparison against boosting models (XGBoost/LightGBM) yet — a natural next step given Random Forest was the only model tuned.
- `Ticket` and `Name` (beyond `Title`) are unused — grouping shared ticket numbers could capture travel groups not reflected in `SibSp`/`Parch`.
- Consider a stacking/voting classifier combining the top 2–3 models from the comparison step.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Place `train.csv` in the same directory as the notebook, then run all cells top to bottom.
