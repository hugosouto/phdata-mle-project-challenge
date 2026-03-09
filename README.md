# Data Analysis & Model Evaluation — KC House Price Prediction

Model evaluation, error analysis, and improvement recommendations for the King County housing price prediction model.

## Purpose

This branch addresses the challenge requirements:
- **Evaluate the model** — accuracy, overfitting/underfitting, error patterns
- **Suggest improvements** — features, hyperparameters, alternative approaches
- **Responsible AI** — interpretability (SHAP), error analysis, causal analysis, counterfactual what-if

## Contents

```
.
├── analysis/
│   ├── rai_analysis_kc_house.ipynb   # Full RAI analysis notebook
│   └── sweetviz_kc_house.html        # Automated EDA profiling report
├── data/
│   ├── kc_house_data.csv             # 21,613 home sales (King County)
│   ├── zipcode_demographics.csv      # US Census demographics per zipcode
│   └── future_unseen_examples.csv    # Unseen homes for prediction testing
├── create_model.py                   # Original baseline model (KNN) for reference
├── conda_environment.yml             # Conda environment (Python 3.11)
└── README.md
```

## Notebook — Responsible AI Analysis

Self-contained notebook (`analysis/rai_analysis_kc_house.ipynb`) that:

1. **Loads data** directly from `data/` and builds train/test splits
2. **Trains both models** — baseline KNN (same as `create_model.py`) and improved GradientBoosting
3. **Compares performance** — side-by-side metrics table, predicted-vs-actual plots
4. **Error Analysis** — Microsoft [erroranalysis](https://erroranalysis.ai/) tree + heatmaps + cohort breakdowns
5. **SHAP Interpretability** — global importance, beeswarm, dependence plots, waterfall local explanations
6. **Counterfactual What-If** — via `responsibleai` + manual scenario analysis
7. **Causal Analysis** — EconML LinearDML treatment effects with heterogeneity
8. **Full Interactive RAI Dashboard** — launches the complete Microsoft Responsible AI Dashboard

## Running

```sh
# Create conda environment
conda env create -f conda_environment.yml
conda activate rai-analysis

# Open the notebook
jupyter notebook analysis/rai_analysis_kc_house.ipynb
```

The notebook is fully self-contained — no need to run `create_model.py` first.
