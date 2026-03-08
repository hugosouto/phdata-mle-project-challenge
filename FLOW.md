# FLOW.md

Decision flow for the phData MLE challenge. Updated as decisions are made.

## Brainstorming Strategy Decisions

```mermaid
graph LR
    BS([Brainstorming Session]) --> STRATEGY{Strategy approach?}
    STRATEGY -->|decided| TWO_TIER[Two-tier: Simple baseline + Wow factor]

    TWO_TIER --> SKILL{Where are Hugo's strengths?}
    SKILL -->|ML training/eval| ML_STRONG[ML is the confident anchor]
    SKILL -->|Presentation| PRES_STRONG[Elite presenter - ministers weekly]
    SKILL -->|API/Docker| API_GAP[API is the critical gap]

    ML_STRONG --> REFRAME[Strategic Reframe]
    PRES_STRONG --> REFRAME
    API_GAP --> REFRAME
    REFRAME --> R1[API code just needs to WORK]
    REFRAME --> R2[Wow factor lives in PRESENTATION]

    TWO_TIER --> WOW{Wow factor approach?}
    WOW -->|decided| WOW_MENU[Presentation + System Thinking]
    WOW_MENU --> W1[Chatbot teaser - 30sec vision slide]
    WOW_MENU --> W2[Domain expertise - Seattle housing research]
    WOW_MENU --> W3[SHAP waterfall for business storytelling]
    WOW_MENU --> W4[Live API demo with curl]
    WOW_MENU --> W5[Architecture diagram that grows]
    WOW_MENU --> W6[Bug discovery mention - code review signal]

    TWO_TIER --> EXEC{Execution approach?}
    EXEC -->|decided| BMAD_QUICK[BMAD Quick Spec + Quick Dev]

    TWO_TIER --> RESEARCH{Domain research?}
    RESEARCH -->|done| DR1[Seattle Housing Market - Deep Research]
    RESEARCH -->|done| DR2[Open Data Enrichment Sources - Deep Research]

    style BS fill:#e1f5fe
    style STRATEGY fill:#fff3e0
    style SKILL fill:#fff3e0
    style WOW fill:#fff3e0
    style EXEC fill:#fff3e0
    style RESEARCH fill:#fff3e0
    style REFRAME fill:#e8f5e9
    style TWO_TIER fill:#e8f5e9
    style BMAD_QUICK fill:#e8f5e9
    style DR1 fill:#e8f5e9
    style DR2 fill:#e8f5e9
```

## Simple API Implementation Decisions

```mermaid
graph LR
    SIMPLE([Simple Baseline Build]) --> FW{API Framework?}
    FW -->|FastAPI| FW_DECISION[Auto-docs, Pydantic validation, modern ML standard]

    SIMPLE --> MODEL{Model artifacts?}
    MODEL -->|Option A| MODEL_DECISION[Run create_model.py manually, bake into Docker]
    MODEL_DECISION --> MODEL_WHY[Simpler build. Talk about retraining in presentation]

    SIMPLE --> DOCKER{Docker approach?}
    DOCKER -->|Plain Dockerfile| DOCKER_DECISION[Single Dockerfile, no docker-compose]
    DOCKER_DECISION --> DOCKER_WHY[docker-compose mentioned in presentation as scaling stepping stone]

    SIMPLE --> ENDPOINT{Endpoints?}
    ENDPOINT --> EP1[POST /predict - accepts 18 columns]
    EP1 --> JOIN[Backend joins demographics on zipcode]
    JOIN --> RESPONSE[Returns predicted_price + metadata]

    SIMPLE --> TEST{Test script?}
    TEST --> TEST_DECISION[Send examples from future_unseen_examples.csv]

    SIMPLE --> CODE{Code quality?}
    CODE --> CODE_DECISION[Self-documenting: detailed docstrings on every function]
    CODE_DECISION --> CODE_WHY[Must explain every line in interview]

    SIMPLE --> FLASK{Why not Flask?}
    FLASK --> FLASK_ANS[No auto-docs, manual validation, older. Learning either way - learn the better tool]

    style SIMPLE fill:#e1f5fe
    style FW fill:#fff3e0
    style MODEL fill:#fff3e0
    style DOCKER fill:#fff3e0
    style ENDPOINT fill:#fff3e0
    style TEST fill:#fff3e0
    style CODE fill:#fff3e0
    style FLASK fill:#fff3e0
    style FW_DECISION fill:#e8f5e9
    style MODEL_DECISION fill:#e8f5e9
    style DOCKER_DECISION fill:#e8f5e9
    style CODE_DECISION fill:#e8f5e9
```

## Technical Implementation Flow

```mermaid
graph LR
    START([Start: phData MLE Challenge]) --> EVAL[Phase 1A: Evaluate Baseline Model]

    EVAL --> EVAL_Q{How does baseline perform?}
    EVAL_Q -->|Overfitting| IMPROVE_OVERFIT[Reduce complexity / regularize]
    EVAL_Q -->|Underfitting| IMPROVE_UNDERFIT[Add features / try better algorithm]
    EVAL_Q -->|Reasonable| IMPROVE_TUNE[Light hyperparameter tuning]

    IMPROVE_OVERFIT --> MODEL[Phase 1B: Improved Model]
    IMPROVE_UNDERFIT --> MODEL
    IMPROVE_TUNE --> MODEL

    MODEL --> ALG_Q{Which algorithm?}
    ALG_Q -->|pending| ALG_DECIDE[Compare RF vs GBR vs KNN improved]

    MODEL --> FEAT_Q{Which features to add?}
    FEAT_Q -->|pending| FEAT_DECIDE[Test ignored columns from sales data]

    ALG_DECIDE --> SAVE_MODEL[Save model.pkl + model_features.json]
    FEAT_DECIDE --> SAVE_MODEL

    SAVE_MODEL --> API[Phase 2: REST API]

    API --> FRAMEWORK_Q{API Framework?}
    FRAMEWORK_Q -->|pending| FRAMEWORK_DECIDE[FastAPI vs Flask]

    FRAMEWORK_DECIDE --> ENDPOINTS[Design Endpoints]
    ENDPOINTS --> PREDICT[POST /predict - full features]
    ENDPOINTS --> MINIMAL[POST /predict/minimal - model features only]
    ENDPOINTS --> HEALTH[GET /health]

    PREDICT --> DOCKER[Phase 3: Containerize]
    MINIMAL --> DOCKER
    HEALTH --> DOCKER

    DOCKER --> DOCKER_Q{Docker approach?}
    DOCKER_Q -->|pending| DOCKER_DECIDE[Dockerfile + docker-compose]

    DOCKER_DECIDE --> TEST[Phase 4: Test Script]
    TEST --> PRESENT[Phase 5: Presentation]

    PRESENT --> BIZ[Business Audience: Value + Outcomes]
    PRESENT --> TECH[Technical Audience: Architecture + MLOps]

    style START fill:#e1f5fe
    style EVAL_Q fill:#fff3e0
    style ALG_Q fill:#fff3e0
    style FEAT_Q fill:#fff3e0
    style FRAMEWORK_Q fill:#fff3e0
    style DOCKER_Q fill:#fff3e0
    style PRESENT fill:#e8f5e9
```

## Decision Log

| # | Decision Point | Status | Choice | Why |
|---|---------------|--------|--------|-----|
| 1 | Overall strategy | **decided** | Two-tier: simple baseline + wow factor in presentation | API just needs to work; presentation is Hugo's elite zone |
| 2 | Execution approach | **decided** | BMAD Quick Spec + Quick Dev | Full BMAD workflow is overkill; fastest path to working baseline |
| 3 | Wow factor delivery | **decided** | Presentation teaser, not built product | Chatbot/vision as 30-sec closing hook, not scope creep |
| 4 | API framework | **decided** | FastAPI | Auto-docs, Pydantic validation, modern ML standard. Learning either way — learn the better tool |
| 5 | Model artifacts | **decided** | Run create_model.py manually, bake into Docker | Simpler. Talk about retraining pipelines in presentation |
| 6 | Docker approach | **decided** | Plain Dockerfile, no docker-compose | Single service, no added complexity. Discuss compose/ECS in presentation |
| 7 | Baseline model performance | pending | — | — |
| 8 | Algorithm for improved model | pending | — | — |
| 9 | Features to add | pending | — | — |
| 10 | Scaling strategy (discussion only) | pending | — | Blue-green / ECS / ALB narrative planned |
| 11 | Model update strategy (discussion only) | pending | — | Blue-green deployment narrative planned |
