# 🦍 Machine Learning for Survival Prediction in Mountain Gorilla Populations (A Simulated Project)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BrunoElDaroux/gorilla-survival-ml/main)


## Project Overview
A full end-to-end machine learning pipeline predicting **individual survival outcomes** in mountain gorilla (*Gorilla beringei beringei*) populations using behavioral, social, and environmental variables collected longitudinally across gorilla groups.


## Research Questions

1. Can individual survival outcomes in mountain gorillas be predicted from behavioral, social, and environmental variables collected longitudinally?
2. Which variables are the strongest predictors of survival; and are those findings consistent with ecological theory on group-living in social mammals?
3. Which social groups are demographically at-risk and should be prioritised for conservation intervention?

---

## ⚠️ Data Transparency

The datasets used in this project is **synthetic** and was created strictly for **educational, analytical, and portfolio purposes**. It does not represent real individual mountain gorillas, verified survival records, or field observations.

The variables, biological context, and assumptions were informed by published knowledge of mountain gorilla ecology and population dynamics. However, **all survival outcomes, model results, and findings presented in this repository are generated from synthetic data and should not be interpreted as real-world findings**.

This project is intended to demonstrate how machine learning can be applied to questions related to mountain gorilla survival. **If a comparable study were conducted using appropriate real-world data from an authorized institution, the analysis could follow a similar approach and produce outputs structured like those demonstrated here.**

Any future research using real institutional or field data would require the appropriate **data-access permission, research authorization, and applicable ethical or regulatory approvals**.

---

## Key Findings Summary (Simulated Results)

1. `Age is the dominant predictor — not group size.`
Permutation feature importance places age at the top of the model (importance = 0.096 ± 0.013), nearly twice the signal of its ordinal encoding (0.049 ± 0.012). Together, these two age-related features account for the strongest survival signal in the data, consistent with established mountain gorilla mortality curves: infants (mean predicted survival probability = 0.82, range 0.46 – 0.97) and seniors (range 0.82 – 1.0) are the highest-risk classes. Of the 18 observed deaths in the test window (2021 – 2023), 12 (67%) were infants and 6 (33%) were seniors; zero prime adults died.

2. `Group composition matters, but group size ranks 4th.`
Proportion of infants in the group (0.012 ± 0.003) and raw group size (0.011 ± 0.005) ranked 3rd and 4th respectively, close together and well below age. This refines the original narrative: it is not simply "larger groups → higher survival"; rather, it is the age structure and silverback ratio (0.005 ± 0.002) of the group that modulate risk alongside size, consistent with dilution-effect theory (Harcourt & Fossey, 1981) operating through compositional pathways.

3. `Rainfall signal is weaker than expected.`
Annual rainfall shows low positive importance (0.002 ± 0.001), but lagged rainfall (prior-year mm) yields negative permutation importance (−0.001), suggesting it adds noise rather than robust signal in this dataset. The expected indirect pathway; rainfall → food availability → survival, was not recovered cleanly by the Random Forest, though elevated infant mortality during the anomalously wet 2015–2016 El Niño years (annual rainfall: 1,342 mm and 1,384 mm vs. 970–1,163 mm baseline) is visible in the raw data.

4. `Sex is not a meaningful predictor.`
`sex_binary` has negative permutation importance (−0.001 ± 0.003); below-chance performance when permuted, indicating that while males may face higher real-world injury risk from dominance contests, the Random Forest finds no clean sex-based survival signal in this dataset. foraging_efficiency similarly underperforms at −0.006 ± 0.003, the most negative feature in the model.

5. `Class imbalance is the central modelling challenge.`
With survival rates near 98%, the model predicts survival = 1 for virtually every individual, yielding deceptively high raw accuracy while missing most actual deaths (18 observed deaths are all predicted as surviving except one). Permutation importance scores must therefore be interpreted cautiously: near-zero or slightly negative values do not necessarily mean a feature is ecologically irrelevant, they reflect signal diluted by an overwhelmingly one-sided target. Balanced metrics (ROC-AUC, balanced accuracy) are the appropriate evaluation lens here.

6. `Conservation output: Ugenda is the highest-priority group.`
Ugenda recorded the most deaths in the test set (4 events: 2 senior deaths in 2023, 2 infant deaths across 2021–2023); more than any other group. Bwenge and Hirwa each recorded 2 deaths. Groups with high infant proportions, smaller sizes, and declining senior cohorts were flagged as priority candidates for intervention.
Also, Small, and isolated groups with declining group size trajectories were flagged as highest priority for intervention.

---

## Tools & Stack
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas / numpy | Data manipulation |
| sqlite3 / pandasql | SQL queries in Python |
| scikit-learn | Random Forest classifier |
| plotnine | ggplot2-style visualizations in Python |
| matplotlib / seaborn | Supplementary plots |
| Jupyter Notebook | All analysis |

---

## Project Structure
```
gorilla_survival_ml/
├── README.md
├── requirements.txt
├── data/
│   └── gorilla_individuals.csv       
│       gorilla_groups.csv           
│       gorilla_observations.csv      
│       rainfall_monthly.csv         
│       analytical_dataset.csv        ← Final ML-ready data
├── notebooks/
│   ├── 01_eda_sql.ipynb              ← SQL joins + Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  ← Build ML-ready feature matrix
│   ├── 03_modeling.ipynb             ← Random Forest + evaluation
│   └── 04_visualization.ipynb        ← ggplot2-style results & plots
└── src/
    ├── data_utils.py                 ← Reusable data functions
    └── model_utils.py                ← Reusable modeling functions
```

---

## Ecological Background
Parameters are grounded in published literature:
- **Survival rates**: Adults ~95%/yr, Infants ~82%/yr (Robbins et al., 2011)
- **Group size**: Ranges 2–38, mean ~10 (Harcourt & Fossey, 1981)
- **Rainfall**: Virunga Massif bimodal pattern ~1800mm/yr
- **Key predictors**: Group size (predator dilution, cooperative defense), rainfall (food availability lag), dominance rank (resource access)

---


## Author

*A Former Data Entry Intern with research experience at the Dian Fossey Gorilla Fund.*
