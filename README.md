# 🦍 Machine Learning for Next-Year Survival in Mountain Gorillas

**A synthetic, end-to-end machine-learning project exploring longitudinal ecological data, mortality-risk classification, model interpretation, and group-level risk screening.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Jupyter](https://img.shields.io/badge/Analysis-Jupyter-F37626)
![Data](https://img.shields.io/badge/Data-100%25%20synthetic-purple)

> **Important:** This is a synthetic research simulation, not a study of real mountain gorillas. All records, survival labels, model outputs, group comparisons, and priority rankings are simulated.

## Project overview

This repository demonstrates how SQL, feature engineering, a Random Forest classifier, model evaluation, and feature-importance analysis can be combined to study a one-year survival outcome in a simulated mountain-gorilla-inspired population.

The target is `survived_next_year` (1 = survived, 0 = did not survive). This is a **binary next-year classification task**, not a time-to-event survival model such as a Cox proportional-hazards model.

### ⚠️ Data Transparency

The datasets used in this project are **synthetic** and were created strictly for **educational, analytical, and portfolio purposes**. They do not represent real individual mountain gorillas, verified survival records, or field observations.

The variables, ecological context, and assumptions were informed by published knowledge of mountain-gorilla ecology and population dynamics. However, **all survival outcomes, model results, group-risk scores, and findings presented in this repository are generated from synthetic data and must not be interpreted as real-world findings**.

This project demonstrates how machine learning could be used to explore questions about survival patterns. If a comparable study were conducted with appropriate real-world data obtained from an authorized institution, a similar analytical workflow could be adapted to the study design and data quality.

Any future research involving real institutional or field data would require appropriate **data-access permission, research authorization, and applicable ethical or regulatory approvals**.

## Dataset and modelling design

| Component | Reported value |
|---|---:|
| Individual-year records | 4,080 |
| Engineered predictor variables | 28 |
| Simulated records labelled survived | 4,007 (98.2%) |
| Simulated records labelled did not survive | 73 (1.8%) |
| Training period | 2010–2020 |
| Training records | 2,895 (98.1% survived) |
| Held-out test period | 2021–2023 |
| Test records | 1,185 (98.5% survived; 18 did not survive) |
| Random Forest trees | 500 |
| Minimum samples per leaf | 4 |
| Feature-importance permutation repeats | 30 |
| Random seed used in model notebook | 42 |

The 4,080 rows are longitudinal **individual-year records**, not 4,080 unique gorillas. The model uses demographic, group-composition, body-condition, behaviour, and rainfall predictors.

## Results — simulated findings

### 1. Class imbalance is the central evaluation challenge

The complete feature matrix contains 4,007 survival labels and 73 non-survival labels. In the held-out 2021–2023 test set, only **18 of 1,185 records (1.52%)** are labelled as non-survival.

This imbalance makes raw accuracy misleading: a classifier that predicts survival for every test record would already achieve approximately **98.48% accuracy**.

### 2. Held-out test performance: high accuracy, no detected deaths

The saved modelling notebook reports the following test-set metrics:

| Metric | Reported result | Interpretation |
|---|---:|---|
| Accuracy | 98% (98.48% from the displayed class counts) | Same as the all-survive baseline at the shown counts |
| ROC-AUC | 0.9037 | Probability ranking separates the two labels, but does not ensure useful threshold-based death detection |
| Average precision | 0.9984 | Computed with **survival** as the positive class; the survival prevalence itself is 98.5% |
| Death-class precision | 0.00 | No correct death predictions were made |
| Death-class recall | 0.00 | **0 of 18 simulated deaths were detected** |
| Death-class F1 | 0.00 | No useful death-class precision/recall balance at the default threshold |
| Macro-average recall | 0.50 | Reflects failure on the death class and near-perfect recall for survival |
| Out-of-bag accuracy | 0.9796 | Also dominated by the majority class |

The classification report shows 18 non-survival records and 1,167 survival records in the test set. At the default 0.5 classification threshold, the model labels all 1,185 records as survived. Consequently, the reported accuracy does **not** show that the classifier identifies mortality risk effectively.

Five-fold stratified cross-validation on the training period reports:

| Metric | Mean ± standard deviation |
|---|---:|
| ROC-AUC | 0.8939 ± 0.0233 |
| Average precision | 0.9977 ± 0.0005 |
| F1 | 0.9901 ± 0.0004 |
| Balanced accuracy | 0.4996 ± 0.0004 |

The high F1 and average-precision values refer to the dominant **survival** class in the current setup. The cross-validation balanced accuracy of approximately 0.50 is consistent with near-zero ability to classify both classes at the chosen threshold. Metrics focused on the rare non-survival outcome, using non-survival as the positive class, should be reported after the modelling pipeline is corrected.

### 3. Permutation feature importance

Permutation importance was calculated using the decrease in ROC-AUC after permuting each feature, with 30 repeats and `random_state=42`.

| Rank | Feature | Mean decrease in ROC-AUC | Standard deviation |
|---:|---|---:|---:|
| 1 | `age` | 0.095988 | 0.012884 |
| 2 | `age_class_ord` | 0.049297 | 0.012030 |
| 3 | `prop_infants` | 0.011839 | 0.002803 |
| 4 | `group_size` | 0.011465 | 0.005180 |
| 5 | `prop_silverbacks` | 0.005367 | 0.001824 |
| 6 | `mean_agonistic` | 0.004721 | 0.002231 |
| 7 | `prop_adult_females` | 0.003958 | 0.001132 |
| 8 | `habitat_quality_score` | 0.002794 | 0.001252 |
| 9 | `dependency_ratio` | 0.002569 | 0.002041 |
| 10 | `annual_rainfall_mm` | 0.001853 | 0.000951 |

**Interpretation:** Age and ordinal age class lead the saved ranking. The mean importance for `age` (0.095988) is nearly twice that of `age_class_ord` (0.049297). `prop_infants` and `group_size` are the next strongest listed predictors, but their reported importance values are much smaller. Annual rainfall ranks tenth in this table and has a comparatively small importance estimate.

Age and age class encode overlapping information, so their individual scores should not be interpreted as independent or causal effects. Permutation importance indicates how a model score changes when a feature is disrupted; it does not establish an ecological mechanism.

### 4. Random Forest impurity importance

The same saved notebook reports these top features using mean decrease in impurity (MDI):

| Rank | Feature | MDI importance |
|---:|---|---:|
| 1 | `age` | 0.236076 |
| 2 | `age_class_ord` | 0.202030 |
| 3 | `group_size` | 0.054269 |
| 4 | `foraging_efficiency` | 0.033124 |
| 5 | `mean_affiliation` | 0.028818 |
| 6 | `prop_infants` | 0.027854 |
| 7 | `body_condition_score` | 0.027506 |
| 8 | `rest_pct` | 0.027099 |
| 9 | `prop_silverbacks` | 0.025684 |
| 10 | `dom_rank_norm` | 0.024791 |

MDI and permutation importance answer different questions and need not have identical rankings. For interpretation, permutation importance is generally easier to connect to a chosen held-out scoring metric, but both methods are affected by correlated predictors and the model's failure to detect the minority class.

### 5. Simulated survival by age class and sex

The SQL exploratory notebook reports these survival percentages across the full synthetic table:

| Age class | Female records | Female survival | Male records | Male survival |
|---|---:|---:|---:|---:|
| Infant | 424 | 94.34% | 455 | 91.87% |
| Juvenile | 628 | 100.00% | 524 | 100.00% |
| Subadult | 371 | 100.00% | 290 | 100.00% |
| Prime adult | 484 | 100.00% | 268 | 100.00% |
| Senior | 359 | 98.33% | 277 | 97.83% |

These rates reflect the synthetic labels and their generation assumptions. They should not be presented as actual age-specific survival rates in mountain gorillas. The perfect survival values for several classes also indicate that the simulation's event distribution is sparse and may limit what a classifier can learn.

### 6. Group-level screening scores

The visualization notebook computes a heuristic score from standardized mean group size and mortality rate. The saved output ranks four groups as High, one as Medium, and seven as Low:

| Simulated group | Mean group size | Mortality rate | Heuristic risk score | Priority label |
|---|---:|---:|---:|---|
| Agashya | 4.385 | 5.2% | 2.936 | High |
| Bwenge | 6.387 | 5.4% | 2.685 | High |
| Ugenda | 8.682 | 5.3% | 2.241 | High |
| Amahoro | 6.283 | 4.3% | 2.165 | High |
| Umubano | 14.279 | 2.9% | 0.026 | Medium |
| Pablo | 13.630 | 1.5% | -0.591 | Low |
| Susa | 16.507 | 2.1% | -0.793 | Low |
| Hirwa | 18.555 | 1.9% | -1.291 | Low |
| Kwitonda | 17.225 | 1.3% | -1.332 | Low |
| Sabyinyo | 17.470 | 0.7% | -1.724 | Low |
| Muhoza | 18.054 | 0.4% | -1.972 | Low |
| Ntambara | 21.457 | 0.8% | -2.350 | Low |

**Interpretation:** Agashya has the highest heuristic score in the saved table (2.936); Ugenda is third among the groups marked High, not the highest by this score. The score is a standardized portfolio heuristic, and the Low/Medium/High labels are created by dividing the score range into three bins. They are not calibrated probabilities, validated demographic-risk categories, or recommendations for real conservation interventions.

## Core formulas

The equations below use GitHub's supported `math` code fences.

### 1. Target definition

Let the survival label for individual-year record `i,t` be 1 if the simulated individual survives through the next year and 0 otherwise.

```math
Y_{i,t} = \begin{cases}1, & \text{survived through } t+1 \\ 0, & \text{did not survive through } t+1\end{cases}
```

### 2. Random Forest probability

The fitted Random Forest contains 500 trees. The predicted probability is the average tree-level probability for the survival class.

```math
\hat{p}_{i,t} = \frac{1}{B}\sum_{b=1}^{B}\hat{p}_{i,t}^{(b)}, \qquad B=500
```

The complementary probability assigned to non-survival is `1 - p_hat`.

### 3. Precision, recall, F1, and balanced accuracy for mortality

For clinically or ecologically relevant mortality evaluation, define death as the positive class: `D = 1 - Y`.

```math
\text{Precision}_{D} = \frac{TP_D}{TP_D+FP_D}
```

```math
\text{Recall}_{D} = \frac{TP_D}{TP_D+FN_D}
```

```math
F1_D = 2\cdot\frac{\text{Precision}_{D}\cdot\text{Recall}_{D}}{\text{Precision}_{D}+\text{Recall}_{D}}
```

```math
\text{Balanced Accuracy} = \frac{\text{Recall}_{D}+\text{Specificity}_{D}}{2}
```

The current test predictions yield death recall and death F1 of 0.00. That is why the reported accuracy of 98.48% is insufficient evidence of a useful mortality classifier.

### 4. ROC-AUC-based permutation importance

For feature `j`, the score is the baseline ROC-AUC minus the average ROC-AUC after randomly permuting that feature. The current notebook uses `R = 30` permutations per feature.

```math
I_j = AUC(X,Y) - \frac{1}{R}\sum_{r=1}^{R} AUC(\pi_r(X_j),Y), \qquad R=30
```

Here, `pi_r(X_j)` denotes the feature matrix with column `j` permuted in repeat `r`. A positive value means that disrupting the feature lowers ROC-AUC; a near-zero or negative value means the model's score does not improve when the feature is preserved under this procedure.

### 5. Group-level heuristic risk score

The visualization notebook standardizes each group's mean size and mortality rate, then combines them with a negative sign for size and a positive sign for mortality.

```math
R_g = -\frac{\bar{S}_g-\mu_S}{\sigma_S} + \frac{M_g-\mu_M}{\sigma_M}
```

`S_g` is the mean group size, `M_g` is the synthetic mortality rate, and `mu` and `sigma` are the across-group mean and standard deviation. This is a hand-built ranking score, not a fitted or calibrated probability of extinction, mortality, or conservation need.

## Workflow and tools

1. **SQL and EDA** — load CSV tables into SQLite, join annual individual, group, and rainfall records, and summarize survival by group, age class, and sex.
2. **Feature engineering** — create age-class encoding, group-composition ratios, normalized dominance rank, rainfall features, behaviour aggregates, and the final 28-feature matrix.
3. **Model training** — fit a 500-tree Random Forest with `class_weight='balanced'`, `min_samples_leaf=4`, `max_features='sqrt'`, and `random_state=42`.
4. **Evaluation and interpretation** — use a temporal split (2010–2020 train; 2021–2023 test), stratified cross-validation on the training subset, ROC-AUC, average precision, class metrics, MDI importance, and permutation importance.
5. **Visualization and exploratory group screening** — plot results and produce heuristic group-level risk rankings.

### Repository layout

```text
simulated-gorilla-survival-ml/
├── README.md
├── requirements.txt
├── data/
│   ├── gorilla_individuals.csv
│   ├── gorilla_groups.csv
│   ├── gorilla_observations.csv
│   ├── rainfall_monthly.csv
│   ├── analytical_dataset.csv
│   └── feature_matrix.csv
├── notebooks/
│   ├── 01_eda_sql.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_visualization.ipynb
├── src/
│   ├── data_utils.py
│   └── model_utils.py
└── outputs/
```

Several notebooks currently save generated plots and model artifacts to `data/`, despite the repository also containing an `outputs/` directory. Standardizing output paths would make the project easier to reproduce and maintain.

## Reproducibility

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the notebooks in numerical order from `notebooks/` so that intermediate CSV files exist before downstream notebooks load them. Some notebook paths are relative to the notebook working directory, so execute each notebook from its expected location or standardize paths before running the workflow in a different environment.

## Limitations and next improvements

1. **Severe class imbalance:** only 18 of 1,185 test records are non-survival cases; the saved model misses all 18 at the default threshold.
2. **Preprocessing leakage:** the feature-engineering notebook fits median imputation on the full dataset before the 2010–2020 / 2021–2023 split. The imputer must be fitted on training data only, preferably inside a scikit-learn `Pipeline`.
3. **Cross-validation design:** stratified folds do not preserve temporal order or group repeated records by individual. Add forward-chaining evaluation and, where the research question requires generalization to unseen individuals or groups, group-aware splits.
4. **Metric orientation:** average precision and F1 are reported for the majority survival class. Recompute precision-recall metrics with non-survival as the positive event and report death recall, specificity, precision, PR-AUC, balanced accuracy, and a confusion matrix.
5. **Decision threshold:** evaluate thresholds using training-only validation or nested temporal validation; do not tune the threshold against the held-out test period.
6. **Interpretability:** `age` and `age_class_ord` are correlated encodings. Importance values are not causal effects.
7. **Group priority is exploratory:** the standardized score is not validated for management decisions and should not be presented as operational conservation advice.
8. **Simulation reproducibility:** keep the data-generation assumptions and random seeds documented, and provide a dedicated generator if the synthetic CSVs are to be regenerated from source.
9. **Scope:** the current workflow predicts a one-year binary label. Time-to-event analysis, censoring, competing risks, calibration, confidence intervals, and external validation are not implemented in this classifier workflow.

## Selected methodological references

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
3. Fisher, A., Rudin, C., & Dominici, F. (2019). All Models are Wrong, but Many are Useful: Variable Importance for Black-Box, Proprietary, or Misspecified Prediction Models. *Journal of Machine Learning Research*, 20(177), 1–81. https://jmlr.org/papers/v20/18-760.html
4. Scikit-learn documentation: [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html) and [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).

These references describe methods used in the project; they do not validate the synthetic data or its simulated ecological findings.

## Author

**Bruno Rudahongerwa Rwarahoze**  
Former Data Entry Intern with research experience at the Dian Fossey Gorilla Fund (DFGF).

---

*All reported values describe the current synthetic dataset and saved notebook outputs. Re-run the complete workflow after any code or data changes, then update this README from the regenerated results.*
