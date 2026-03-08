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
- **Status:** pending
- **Scope:** FastAPI + Docker + test script + model evaluation + model improvement

### 4. Quick Dev (QD) — `/bmad-bmm-quick-dev`
- **Purpose:** Implement the quick spec
- **Status:** pending

## Phase 3 — Wow Factor (If Time Allows)

### 5. Wow factor enhancements (ranked by impact/effort)
- [ ] Bug discovery mention in presentation (0 min)
- [ ] Sound Realty branding in API response (5 min)
- [ ] Before/After model comparison visual (30 min)
- [ ] SHAP waterfall for business storytelling (1-2 hrs)
- [ ] Live API demo with curl in presentation (0 min)
- [ ] Architecture diagram that grows (1 hr)
- [ ] Interactive Price Map with Folium (1-2 hrs)
- [ ] "What If" scenario re-prediction endpoint (2 hrs)
- [ ] Chatbot teaser slide with market data (slides only)
- [ ] EPA Smart Location Database enrichment (1-3 hrs)

## Phase 4 — Presentation

### 6. Build presentation
- **Status:** pending
- **Format:** Business half (15 min) + Technical half (15 min)
- **Key assets:** Domain research, SHAP plots, live demo, architecture diagram, chatbot teaser
