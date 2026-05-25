# Project 2 Complete Guide — MLflow + GitHub Actions CI/CD
## Adding Experiment Tracking and Automated Testing to Churn Radar

> This document covers every step added in Project 2. It assumes you have
> completed Project 1 (GUIDE.md) and have the churn-radar API running.

---

## Table of Contents

1. [What We Added in Project 2](#1-what-we-added-in-project-2)
2. [MLflow — Experiment Tracking](#2-mlflow--experiment-tracking)
3. [GitHub Actions — CI/CD Pipeline](#3-github-actions--cicd-pipeline)
4. [Files Changed or Created](#4-files-changed-or-created)
5. [How to Use MLflow Day-to-Day](#5-how-to-use-mlflow-day-to-day)
6. [How CI/CD Works in Practice](#6-how-cicd-works-in-practice)
7. [Troubleshooting](#7-troubleshooting)
8. [Key Concepts Glossary](#8-key-concepts-glossary)

---

## 1. What We Added in Project 2

```
Project 1 (foundation)          Project 2 (added on top)
──────────────────────          ──────────────────────────────────
data preprocessing         →   unchanged
train.py                   →   + MLflow logging throughout
predict.py                 →   unchanged
app/main.py (FastAPI)      →   unchanged
Dockerfile                 →   unchanged
requirements.txt           →   + mlflow[skinny], pyarrow
.gitignore                 →   + mlruns/ excluded
                                + .github/workflows/ci.yml  (NEW)
```

**Why these two things?**

Without MLflow: you run train.py, get some numbers, forget what settings
you used, run it again with different settings, can't remember which was better.

Without CI/CD: you push code, it might break the tests, nobody knows until
someone manually runs pytest — which often doesn't happen.

Together they solve the two most common failures in early ML engineering:
- "Which model version is this, and how was it trained?" (MLflow)
- "Is the code actually passing its tests right now?" (GitHub Actions)

---

## 2. MLflow — Experiment Tracking

### What is MLflow?

MLflow is an open-source platform for managing the entire ML lifecycle.
We use its **Tracking** component, which automatically records:

| What it logs | Example |
|-------------|---------|
| Parameters | n_estimators=200, max_depth=10, learning_rate=0.1 |
| Metrics | accuracy=0.77, f1=0.63, roc_auc=0.84 |
| Artifacts | best_model.pkl, metrics.json, features.json |
| Tags | best_model=RandomForest, churn_rate=26.54%, python=3.13 |
| Run metadata | start time, duration, run ID |

Every time you run `python src/train.py`, a new row is added to
the MLflow experiment table. After 10 runs you can compare them all
in a browser dashboard.

### Core MLflow concepts

**Experiment** = a named group of related runs.
Ours is called "churn-radar". Think of it as a folder.

**Run** = one execution of train.py. Has its own parameters, metrics,
artifacts, and a unique run ID like `22af9d2f0ddc430c9e52e6eb46c2a4ec`.

**Artifact** = any file attached to a run. We attach:
- `best_model.pkl` — the saved model
- `metrics.json` — training results
- `features.json` — which columns were used

**Nested runs** = runs inside runs. Our parent run is "churn-radar-training".
Inside it, "RandomForest" and "XGBoost" are nested runs with their own
parameters and metrics.

```
Run: "churn-radar-training"         ← parent run (one per train.py execution)
  Tags: best_model=RandomForest
  Metrics: best_accuracy=0.77, best_roc_auc=0.84
  Artifacts: best_model.pkl, metrics.json
  │
  ├── Nested Run: "RandomForest"    ← RF-specific params + metrics
  │     Params: n_estimators=200, max_depth=10
  │     Metrics: accuracy=0.77, roc_auc=0.84
  │
  └── Nested Run: "XGBoost"        ← XGB-specific params + metrics
        Params: n_estimators=200, learning_rate=0.1
        Metrics: accuracy=0.76, roc_auc=0.83
```

### Where MLflow stores data

```
mlruns/                         ← created automatically when you run train.py
  0/                            ← default experiment
  1/                            ← "churn-radar" experiment
    22af9d2f0ddc430c9e52e6eb46c2a4ec/   ← one folder per run
      artifacts/
        best_model.pkl
        metrics.json
      metrics/                  ← one file per metric, one value per line
        accuracy
        f1
        roc_auc
      params/                   ← one file per parameter
        n_estimators
        max_depth
      tags/
        best_model
        mlflow.runName
```

This is all local to your machine. In a real team, you'd point
`MLFLOW_TRACKING_URI` at a central server so everyone shares runs.

### MLflow code changes in train.py

**1. Setup (runs once before training)**
```python
mlflow.set_tracking_uri("mlruns")         # where to save data
mlflow.set_experiment("churn-radar")      # which experiment bucket
```

**2. Parent run (wraps the whole training session)**
```python
with mlflow.start_run(run_name="churn-radar-training") as parent_run:
    run_id = parent_run.info.run_id

    # Log dataset info as tags
    mlflow.set_tags({
        "dataset": "Telco Customer Churn",
        "dataset_rows": 7043,
        "churn_rate": "26.54%",
    })

    # ... train models ...

    # Log best model's metrics on the parent run
    mlflow.log_metrics({"best_accuracy": 0.77, "best_roc_auc": 0.84})
    mlflow.set_tag("best_model", "Random Forest")

    # Save model file as artifact
    mlflow.log_artifact("model/best_model.pkl")
```

**3. Nested run (one per model)**
```python
with mlflow.start_run(run_name="RandomForest", nested=True):
    # Log all hyperparameters
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced",
    })

    model = RandomForestClassifier(...)
    model.fit(X_train, y_train)

    # Log evaluation metrics
    mlflow.log_metrics({
        "accuracy": 0.77,
        "f1": 0.63,
        "roc_auc": 0.84,
    })

    # Log the model itself (queryable, loadable, versioned)
    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")
```

### Running the MLflow UI

```powershell
cd churn-radar
mlflow ui
```

Open **http://localhost:5000** in your browser.

You'll see:
- A table with one row per training run
- Click any run to see all its parameters, metrics, and artifacts
- Use the checkboxes to select multiple runs and click "Compare"
  to see a side-by-side comparison table and charts

**Try this:** Run `python src/train.py` twice. Then open the MLflow UI.
You'll have two rows in the table — same results since we didn't change
anything, but it demonstrates how the history builds up over time.

---

## 3. GitHub Actions — CI/CD Pipeline

### What is GitHub Actions?

GitHub Actions is automation built into GitHub. When you push code,
GitHub reads your `.github/workflows/ci.yml` file and runs it on their
servers automatically.

You get:
- A green checkmark on commits where tests pass
- A red X on commits where tests fail
- Email notifications when something breaks
- A full log of what ran and what failed

### The workflow file: `.github/workflows/ci.yml`

```yaml
name: CI - Test and Build

on:
  push:
    branches: [ main ]      # runs on every push to main
  pull_request:
    branches: [ main ]      # runs on every PR to main
```

**Two jobs run in sequence:**

#### Job 1: test
```
1. Checkout your code (clone the repo onto a GitHub runner)
2. Install Python 3.11
3. pip install -r requirements.txt
4. Create placeholder model files (so tests don't fail on missing .pkl)
5. pytest tests/ -v
```

Why placeholder models?
Our real model needs the Kaggle dataset to train. The CI server doesn't
have it. But our tests mock the predictor anyway — they don't actually
call the model. We just need loadable .pkl files so the imports don't crash.

```python
# The CI step creates a tiny 2-tree Random Forest as a stand-in
clf = RandomForestClassifier(n_estimators=2)
clf.fit(X_dummy, y_dummy)
joblib.dump(clf, "model/best_model.pkl")
```

#### Job 2: docker-build
```
1. Only runs if Job 1 (test) passed — "needs: test"
2. Checkout code
3. Create placeholder model files again
4. docker build -t churn-radar:ci .
5. docker run -d -p 8000:8000 churn-radar:ci
6. sleep 20 (wait for startup)
7. curl http://localhost:8000/health  ← smoke test
8. docker stop
```

This confirms:
- The Dockerfile has no syntax errors
- All dependencies install correctly inside the container
- The app starts and responds to HTTP requests

### What happens when you push to GitHub

```
git push origin main
        │
        ▼
GitHub detects push
        │
        ▼
Reads .github/workflows/ci.yml
        │
        ├── Job: test
        │     └── pytest tests/ -v
        │           ├── PASS → green checkmark ✓
        │           └── FAIL → red X, email sent ✗
        │
        └── Job: docker-build (only if test passed)
              └── docker build + smoke test
                    ├── PASS → green checkmark ✓
                    └── FAIL → red X ✗
```

### Viewing CI results on GitHub

1. Go to your repo: **https://github.com/tusharaloorkar/churn-radar**
2. Click the **"Actions"** tab
3. You'll see a list of workflow runs — one per push
4. Click any run to see detailed logs for each step

Every commit in your repo history will show a green ✓ or red ✗
next to the commit message.

---

## 4. Files Changed or Created

### New file: `.github/workflows/ci.yml`
The GitHub Actions workflow. Never ran locally — GitHub runs it.

### Modified: `src/train.py`
Added MLflow tracking throughout:
- `mlflow.set_experiment()` at the top
- `with mlflow.start_run()` wrapping all training code
- `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.log_artifact()`
- Nested runs for RandomForest and XGBoost

### Modified: `requirements.txt`
Added:
```
mlflow[skinny]==3.12.0   ← [skinny] avoids heavy optional deps
pyarrow==23.0.1          ← required by mlflow, installed as binary wheel
```

### Modified: `.gitignore`
Added:
```
mlruns/     ← MLflow's local database (large, changes every run, not for Git)
mlflow.db   ← alternative SQLite backend
```

---

## 5. How to Use MLflow Day-to-Day

### Train and track
```powershell
python src/train.py
```
Every run is logged automatically. No extra commands needed.

### View the dashboard
```powershell
mlflow ui
# Open: http://localhost:5000
```

### Compare two runs
1. Open http://localhost:5000
2. Check the boxes next to two runs
3. Click **"Compare"**
4. See a table of params and metrics side by side

### Experiment: change a hyperparameter and compare

Try changing `n_estimators` from 200 to 100 in `train.py`:
```python
"n_estimators": 100,   # was 200
```
Run `python src/train.py` again. Now you have two runs in MLflow.
Compare them — you'll see the accuracy difference was tiny but training
was faster. That's a real engineering trade-off decision, made with data.

### Finding a run by ID

Every run has a unique ID printed at the end of training:
```
MLflow Run ID: 22af9d2f0ddc430c9e52e6eb46c2a4ec
```

You can load the exact model from any run:
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("runs:/22af9d2f0ddc430c9e52e6eb46c2a4ec/random_forest_model")
```
This is powerful for reproducing results — you can reload any model
from any point in time, even months later.

---

## 6. How CI/CD Works in Practice

### Normal workflow (everything passes)
```
# Make a change
edit app/main.py

# Stage and commit
git add app/main.py
git commit -m "Add new endpoint"
git push

# GitHub automatically runs CI (takes ~3 minutes)
# You get an email: "CI - Test and Build: success"
# Green checkmark appears on the commit
```

### When tests fail
```
# Make a breaking change (accidentally)
edit tests/test_api.py  # introduce a bug

git add .
git commit -m "Oops"
git push

# GitHub runs CI
# pytest fails
# You get an email: "CI - Test and Build: failure"
# Red X appears on the commit

# Fix the bug
edit tests/test_api.py
git add .
git commit -m "Fix broken test"
git push

# CI runs again, passes
# Green checkmark restored
```

### The key rule: never push broken code

In a team environment, broken CI blocks the pull request from being merged.
Even working alone, the discipline of "CI must pass" forces you to run
tests before pushing — a habit that prevents bugs from reaching production.

---

## 7. Troubleshooting

### "No module named 'mlflow'"
```powershell
pip install "mlflow[skinny]" pyarrow --prefer-binary
```

### "mlflow[skinny] fails to install — pyarrow won't build"
pyarrow needs a pre-built binary wheel. Install with:
```powershell
pip install pyarrow --prefer-binary
pip install "mlflow[skinny]"
```
If that still fails, your Python version may not have pyarrow wheels yet.
Check https://pypi.org/project/pyarrow/#files and look for your Python version.

### "Port 5000 already in use" (mlflow ui won't start)
```powershell
mlflow ui --port 5001
# Then open: http://localhost:5001
```

### GitHub Actions workflow not showing in Actions tab
- The file MUST be at exactly: `.github/workflows/ci.yml`
- The YAML indentation must be consistent (spaces only, no tabs)
- Push the file and wait ~30 seconds for GitHub to detect it

### CI fails with "model files not found"
The CI creates placeholder model files in the workflow. If you see import
errors about missing .pkl files, check that the "Create placeholder model
files" step ran successfully in the GitHub Actions log.

---

## 8. Key Concepts Glossary

| Term | Simple Explanation |
|------|--------------------|
| **MLflow** | A tool that automatically records every training run's settings and results |
| **Experiment** | A named group of runs in MLflow (like a folder) |
| **Run** | One execution of train.py — has its own params, metrics, artifacts |
| **Artifact** | Any file attached to an MLflow run (model, plots, CSVs) |
| **Nested run** | A run inside another run — used to track individual models within a session |
| **GitHub Actions** | Automation built into GitHub — runs workflows on every push |
| **Workflow** | A YAML file in .github/workflows/ that defines what to automate |
| **Job** | A group of steps inside a workflow that run on one virtual machine |
| **Runner** | The temporary GitHub-hosted virtual machine that executes your jobs |
| **CI** | Continuous Integration — automatically test on every push |
| **CD** | Continuous Delivery — automatically build/deploy after tests pass |
| **Smoke test** | The simplest possible check — does the app start and respond at all? |
| **Placeholder model** | A minimal trained model used in CI so tests run without real training data |
| **`needs: test`** | Declares that docker-build only runs if the test job passes first |
| **`mlflow.log_params()`** | Save hyperparameters (settings) to the current run |
| **`mlflow.log_metrics()`** | Save numeric results (accuracy, F1, etc.) to the current run |
| **`mlflow.log_artifact()`** | Attach a file to the current run |

---

## What's Next — Project 3

Project 3 adds **DVC (Data Version Control)**:

- Version your dataset alongside your code (like Git, but for data files)
- Version your model files without committing huge binaries to Git
- Build a retraining pipeline: one command updates data, retrains, and
  saves a new model version — all tracked and reproducible

This is the skill that gets you from "I know MLflow" to "I understand
the full MLOps data lifecycle" — a rare combination at the entry level.

---

*Guide written alongside the Project 2 build — May 2026*
*Stack: Python 3.13 · MLflow 3.12 · GitHub Actions · FastAPI · Docker*
