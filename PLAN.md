# PLAN.md

BMAD workflow plan for the phData MLE Project Challenge.

## Phase 1 — Analysis

### 1. Brainstorm Project (BP) — `/bmad-brainstorming`
- **Purpose:** Expert-guided ideation to explore solution approaches
- **Status:** done
- **Output:** `_bmad-output/brainstorming/brainstorming-session-2026-03-08-0200.md`
- **Key decisions:**
  - Two-tier strategy: simple baseline FIRST, wow factor in presentation only
  - API code just needs to work cleanly (FastAPI + Docker)
  - Wow factor lives in presentation and system thinking (Hugo's strength)
  - Chatbot vision = 30-second teaser slide, not a deliverable
  - Minimum delivery via BMAD quick spec + quick dev

### 2. Domain & Enrichment Research (DR) — Claude Max Deep Research
- **Purpose:** Seattle housing market domain knowledge + open data enrichment sources
- **Status:** done
- **Output:**
  - `_bmad-output/brainstorming/research-results/Prompt 1 - Seattle Housing Market Domain Research.md`
  - `_bmad-output/brainstorming/research-results/Prompt 2 - Open Data Enrichment Sources for King County Housing Model.md`

### ~~3. Technical Research~~ — SKIPPED
### ~~4. Create Brief~~ — SKIPPED
### ~~5. Create PRD~~ — SKIPPED
_Brainstorming covered strategic decisions. Full BMAD planning workflow is overkill for this project. Going straight to quick spec + quick dev._

## Phase 2 — Build (Quick Path)

### 3. Quick Spec (QS) — `/bmad-bmm-quick-spec`
- **Purpose:** Implementation-ready spec for minimum viable delivery
- **Status:** done
- **Output:** `_bmad-output/implementation-artifacts/tech-spec-sound-realty-api.md`
- **Scope:** FastAPI + Docker + test script (6 tasks, 8 ACs, 6 files)
- **Reviews completed:**
  - Party Mode: 4 improvements (path resolution, fail-fast startup, zipcode dtype, edge case test)
  - Adversarial Review: 13 findings, 7 fixes applied (NaN flow, negative price guard, test assertions, requests out of prod, logging, non-root Docker, security notes)

### 4. Quick Dev (QD) — `/bmad-bmm-quick-dev`
- **Purpose:** Implement the quick spec
- **Status:** done
- **Output:** 6 files created (`requirements.txt`, `app/__init__.py`, `app/main.py`, `Dockerfile`, `.dockerignore`, `test_api.py`)
- **Test results:** 7/7 checks passed (health, 5 predictions, edge case)
- **ACs verified:** 7/8 (AC 7 Docker build pending manual verification)
- **Note:** Fixed `str | None` → `Optional[str]` for Python 3.9 compatibility

## Phase 3 — Core Deliverables (Model)

### 5. Model Evaluation
- **Purpose:** Assess generalization, overfitting/underfitting of the provided KNN baseline
- **Status:** pending
- **Deliverable:** Evaluation notebook/script with metrics (R², RMSE, MAE), learning curves, residual analysis
- **Key questions:** Does the model generalize? Is it overfitting (KNN with k=5 on 21K rows)? Where does it fail?

### 6. Model Improvement
- **Purpose:** Apply basic ML principles to improve on the baseline (not a Kaggle competition — 80% solution)
- **Status:** pending
- **Approach:** Traditional ML — feature engineering, algorithm selection, hyperparameter tuning
- **Candidates:** Add ignored features (grade, waterfront, yr_built, lat/long), try Ridge/Lasso/RandomForest/GradientBoosting, cross-validation
- **Output:** Updated `model/model.pkl` + before/after comparison

## Phase 4 — Wow Factor (If Time Allows)

### 7. Wow factor enhancements (ranked by impact/effort)
- [x] Bug discovery mention in presentation (0 min) — `create_model.py` line 14: `DEMOGRAPHICS_PATH` points to wrong file
- [x] Sound Realty branding in API response (5 min) — `"provider": "Sound Realty AI"` in every response
- [ ] Before/After model comparison visual (30 min)
- [ ] SHAP waterfall for business storytelling (1-2 hrs)
- [ ] Live API demo with curl in presentation (0 min)
- [ ] Architecture diagram that grows (1 hr)
- [ ] Interactive Price Map with Folium (1-2 hrs)
- [ ] "What If" scenario re-prediction endpoint (2 hrs)
- [ ] Chatbot teaser slide with market data (slides only)
- [ ] EPA Smart Location Database enrichment (1-3 hrs)

## Phase 5 — Presentation

### 8. Build presentation
- **Status:** pending
- **Format:** Business half (15 min) + Technical half (15 min)
- **Key assets:** Domain research, model evaluation results, before/after metrics, SHAP plots, live demo, architecture diagram, chatbot teaser
