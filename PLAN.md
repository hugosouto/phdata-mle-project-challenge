# PLAN.md

phData MLE Project Challenge — Sound Realty housing price prediction.

## Branches

| Branch | Purpose | Status |
|---|---|---|
| `main` | Clean repo for GitHub submission | active |
| `baseline-api` | Frozen simple FastAPI + KNN + dashboard | frozen |
| `enhanced-api` | 4 endpoints, LightGBM, monitoring, retraining | frozen |
| `data-analysis` | Notebooks, RAI analysis, context docs | active |

Worktrees at `worktrees/{branch-name}/` (gitignored).

---

## Deliverable 1 — REST API + Docker (Required)

> Deploy the model as a RESTful service, JSON POST, return prediction + metadata.

| Requirement | Status | Branch |
|---|---|---|
| Endpoint accepting `future_unseen_examples.csv` columns | done | baseline-api, enhanced-api |
| Return JSON with prediction + metadata | done | both (enhanced adds R², MAE, MAPE, RMSE, n_features, uses_demographics) |
| Backend join of `zipcode_demographics.csv` | done | both |
| Docker containerized | done | both |
| Discuss autoscaling (presentation only) | pending | presentation |
| Zero-downtime model updates | done | enhanced-api (hot-swap retraining) |
| Bonus: minimal-features endpoint | done | both |

**What exists:**
- `baseline-api`: 2 endpoints (full + minimal), KNN only, monitoring dashboard
- `enhanced-api`: 4 endpoints (basic, minimal, enhanced, optimized), 3 models (KNN + 2 LightGBM), monitoring dashboard, retraining module, stress testing

---

## Deliverable 2 — Test Script (Required)

> Script that sends `future_unseen_examples.csv` to the endpoint.

| Requirement | Status | Branch |
|---|---|---|
| Send examples, show predictions work | done | both |
| baseline: `test_api.py` (7 checks) | done | baseline-api |
| enhanced: `test_api.py` (16 checks, all 4 endpoints + edge cases) | done | enhanced-api |

---

## Deliverable 3 — Model Evaluation (Required)

> Evaluate how well the model generalizes. Overfitting? Underfitting?

| Requirement | Status | Where |
|---|---|---|
| Baseline KNN evaluation (R², MAE, MAPE, RMSE) | done | enhanced-api `create_model.py` outputs metrics |
| RAI / Responsible AI analysis notebook | done | data-analysis `analysis/rai_analysis_kc_house.ipynb` |
| Overfitting/underfitting analysis | **pending** | needs notebook with learning curves, cross-validation |
| Residual analysis | **pending** | needs residual plots, error by price range |
| Feature importance / what's missing | **pending** | needs SHAP or feature importance analysis |

**What's needed:** A proper model evaluation notebook that tells the story:
1. Baseline KNN: what features it uses, what it ignores, R²=0.73
2. Cross-validation to check overfitting (KNN k=5 on 21K rows)
3. Residual plots — where does the model fail? (high-price homes, waterfront)
4. Feature importance — which dropped features matter most (grade, waterfront, lat/long)

---

## Deliverable 4 — Model Improvement (Required)

> Apply basic ML principles. Not Kaggle — explain your decisions.

| Requirement | Status | Where |
|---|---|---|
| Improved model trained | done | enhanced-api `create_model.py` (LightGBM, R²=0.87) |
| Before/after comparison | partially done | metrics exist, needs visual in notebook/presentation |
| Feature engineering rationale | **pending** | needs documentation in notebook |
| Explain decisions made | **pending** | presentation |

**What's needed:** A notebook showing the improvement journey:
1. Why LightGBM over KNN? (handles mixed features, no scaling needed, feature importance built-in)
2. Which features added and why? (grade = biggest impact, waterfront, lat/long for spatial patterns)
3. Before/after metrics comparison (R² 0.73 → 0.87, MAE $102K → $74K)
4. Why NOT deeper tuning? (80% solution philosophy, diminishing returns)

---

## Deliverable 5 — Presentation (Required)

> 15 min business + 15 min technical. Expect lots of Q&A.

| Requirement | Status |
|---|---|
| Business half (non-technical audience) | **pending** |
| Technical half (engineers/scientists) | **pending** |
| Prepare for Q&A | **pending** |

### Business Half (15 min) — "Pretend we're real estate professionals"
- Problem: CMA takes 2-4 hours manually, ML does it in seconds
- Demo: show API predicting prices (live curl or dashboard)
- Value: consistency, speed, data-driven (21,613 transactions vs. gut feeling)
- Feature impacts: SHAP waterfall showing dollar values per feature
- Trust: explain what the model considers (grade, sqft, location, waterfront)
- Sound Realty fit: how this plugs into their workflow

### Technical Half (15 min) — "Now let's dig into the details"
- API architecture: FastAPI + uvicorn + Docker
- Model progression: KNN baseline → LightGBM (explain why, show metrics)
- Scaling discussion: Azure (or AWS) — container orchestration, load balancer, autoscaling
- Zero-downtime deployment: retraining module, hot-swap, versioning
- Monitoring: dashboard, metrics middleware, latency tracking
- MLOps awareness: model registry, CI/CD, feature stores, data drift detection
- Bug found in provided code (`create_model.py` line 14)

---

## Remaining Work — Priority Order

### P0: Must Have (core requirements not yet fulfilled)

1. **Model evaluation notebook** — `data-analysis` branch
   - Learning curves, cross-validation, residual analysis
   - Feature importance (what baseline ignores)
   - This IS a deliverable, not just presentation material

2. **Model improvement notebook** — `data-analysis` branch
   - Before/after comparison with charts
   - Feature engineering rationale
   - Decision explanations (why LightGBM, why these features)

3. **Presentation deck** — can be Gamma, Google Slides, or similar
   - Business half + Technical half
   - SHAP waterfall plots for business storytelling

### P1: Should Have (strong differentiators)

4. **SHAP analysis** — feature importance with dollar impacts
   - Business-friendly visualization
   - Goes into both notebook and presentation

5. **README cleanup on main** — what reviewers see on GitHub
   - Clear project structure explanation
   - How to run everything
   - Which branch has what

### P2: Nice to Have (wow factor)

6. Before/after model comparison visual
7. Architecture diagram (laptop → production scaling)
8. Live API demo prep (curl commands ready)
9. Interactive price map with Folium

---

## Context Files (data-analysis branch)

| File | Purpose |
|---|---|
| `context/ML_Project_Challenge_Instructions.pdf` | Original assignment |
| `context/ML_Project_Challenge_Prep_Video_Transcript.txt` | Andy's video guidance |
| `context/Sound Realty Multifamily - Competitive intelligence and AI strategy.md` | Domain research |
| `context/sound-realty-briefing-dashboard.html` | Client briefing |
| `context/HugoSouto_AISolutionsArchitect_Presentation.html` | Hugo's portfolio reference |
| `analysis/rai_analysis_kc_house.ipynb` | RAI analysis notebook |
| `analysis/sweetviz_kc_house.html` | Sweetviz EDA report |
