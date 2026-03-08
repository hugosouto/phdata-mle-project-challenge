# PRINCIPLES.md

Key principles extracted from the phData challenge instructions and prep video transcript. Use these as a checklist when making any decision.

## Evaluation Criteria (What They're Really Looking For)

1. **How you think** — not perfection, not the "right" answer
2. **How you structure solutions** — tradeoffs, design decisions, and the "why"
3. **How you communicate** — to both business and technical audiences
4. **Can you deploy and test an ML model properly?**
5. **Can you adapt, research, and make reasonable tradeoffs?**

## Build Philosophy

- **Simplest working solution first** — do not overcomplicate
- **80% solution** — this is not a Kaggle competition
- **Don't get stuck** — move on, ask questions, use the internet
- **Focus on core strengths** — play to what you know
- **No overengineering** — build something that works, not something impressive-looking

## API Requirements (Hard)

- REST endpoint, JSON POST
- Inputs = columns from `future_unseen_examples.csv` (no demographic data from client)
- Backend joins demographic data automatically
- Returns prediction + metadata
- Bonus: minimal endpoint with only required model features

## Technical Depth Expected

- **Explain the "why"** for every design decision and tradeoff
- **Scaling**: how would you scale up/down without stopping the service?
- **Model updates**: how to deploy new model versions with zero downtime?
- **Docker**: why you used it (or didn't)
- **Performance measurement**: how accurate is the model? overfitting? underfitting?

## MLOps Awareness (Breadth + Depth)

Touch on all of these, go deep in at least one:
- Model registries
- Feature stores
- CI/CD pipelines
- Monitoring (data drift, prediction drift)
- Versioning and deployment
- **AWS preferred**, but use what you're comfortable with

## Presentation Rules

- **Business half**: pretend they're real estate professionals, not engineers. Focus on value, outcomes, time savings, how it fits their process.
- **Technical half**: dive into architecture, Docker, scaling, model updates, performance metrics, design decisions, tradeoffs.
- **Expect lots of questions** after presenting — be ready to defend every choice.
- **AI usage is fine** but you must be able to explain anything AI helped create in full detail.

## Model Evaluation Expectations

- Start from `create_model.py`
- Figure out how well the model generalizes to new data
- Has it appropriately fit the dataset?
- Improve using basic ML principles — understanding matters more than performance numbers

## Anti-Patterns (What NOT to Do)

- Don't spend too much time on model tuning
- Don't deploy to cloud — laptop + Docker Desktop is sufficient
- Don't aim for a specific end result — choose your own adventure
- Don't present without explaining the "why"
- Don't use AI as a crutch — augment, don't replace understanding
