# Project 1 Complete Guide — Churn Radar
## End-to-End ML API with FastAPI + Docker + Render

> This document walks through every single step taken to build and deploy
> the Churn Radar project from scratch. Written for complete beginners —
> every command, every concept, and every "why" is explained.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [Day 1 — Setup & Dataset](#4-day-1--setup--dataset)
5. [Day 2 — Data Preprocessing](#5-day-2--data-preprocessing)
6. [Day 3 — Model Training](#6-day-3--model-training)
7. [Day 4 — FastAPI Application](#7-day-4--fastapi-application)
8. [Day 5 — Docker](#8-day-5--docker)
9. [Day 6 — GitHub](#9-day-6--github)
10. [Day 7 — Deploy to Render](#10-day-7--deploy-to-render)
11. [Testing the Live API](#11-testing-the-live-api)
12. [Troubleshooting](#12-troubleshooting)
13. [Key Concepts Glossary](#13-key-concepts-glossary)

---

## 1. What We Built

A **live REST API** that predicts whether a telecom customer will cancel
their service ("churn"), based on their account details.

**The full pipeline:**
```
Raw CSV data
    ↓
preprocess.py   → clean data, encode categories, scale numbers
    ↓
train.py        → train Random Forest + XGBoost, save best model
    ↓
predict.py      → load saved model, run inference on new data
    ↓
app/main.py     → FastAPI wraps predict.py into HTTP endpoints
    ↓
Dockerfile      → package everything into a container
    ↓
Render          → run the container on a cloud server (public URL)
```

**Live URLs:**
- API: https://churn-radar.onrender.com
- Interactive docs: https://churn-radar.onrender.com/docs
- GitHub: https://github.com/tusharaloorkar/churn-radar

**Model results:**
- Winner: Random Forest
- Accuracy: 77.0%
- F1 Score: 0.63
- ROC-AUC: 0.84

---

## 2. Prerequisites

Before starting, make sure you have:

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.11+ | Run the code | python.org |
| pip | Install Python libraries | Comes with Python |
| Git | Version control | git-scm.com |
| Docker Desktop | Containerize the app | docker.com |
| GitHub account | Host the code | github.com |
| Render account | Deploy the app | render.com |
| Kaggle account | Download the dataset | kaggle.com |

**Check everything is installed:**
```powershell
python --version    # should show 3.11 or higher
git --version       # should show 2.x
docker --version    # should show 27.x or higher
```

---

## 3. Project Structure

```
churn-radar/
│
├── data/
│   ├── raw/                          ← put the Kaggle CSV here
│   └── processed/                    ← auto-generated cleaned data
│
├── notebooks/
│   └── 01_eda.ipynb                  ← exploratory data analysis
│
├── src/
│   ├── __init__.py                   ← marks src as a Python package
│   ├── preprocess.py                 ← data cleaning + feature engineering
│   ├── train.py                      ← model training script
│   └── predict.py                    ← inference logic for the API
│
├── app/
│   ├── __init__.py                   ← marks app as a Python package
│   └── main.py                       ← FastAPI application
│
├── model/
│   ├── best_model.pkl                ← saved trained model (binary)
│   ├── encoders.pkl                  ← saved label encoders
│   ├── scaler.pkl                    ← saved feature scaler
│   └── metrics.json                  ← training results
│
├── tests/
│   ├── __init__.py
│   └── test_api.py                   ← automated API tests
│
├── .gitignore                        ← files Git should ignore
├── .dockerignore                     ← files Docker should ignore
├── Dockerfile                        ← container build recipe
├── requirements.txt                  ← Python dependencies
├── README.md                         ← project overview for GitHub
└── GUIDE.md                          ← this file
```

---

## 4. Day 1 — Setup & Dataset

### Step 1.1 — Create the folder structure

```powershell
mkdir churn-radar
cd churn-radar
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir src
mkdir app
mkdir model
mkdir tests
```

### Step 1.2 — Download the dataset

1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download**
3. Unzip and place the CSV at:
   ```
   data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

**About the dataset:**
- 7,043 rows — each row is one customer
- 21 columns — things like tenure, monthly charges, contract type
- Target column: `Churn` — "Yes" (left) or "No" (stayed)
- Churn rate: 26.5% — about 1 in 4 customers churns

### Step 1.3 — Create requirements.txt

```
fastapi==0.115.0
uvicorn==0.32.0
pydantic==2.9.2
scikit-learn==1.6.1
xgboost==2.1.4
pandas==2.2.3
numpy==2.2.0
joblib==1.4.2
pytest==8.3.3
httpx==0.27.2
python-multipart==0.0.12
```

> **Why pin versions?**
> Without pinning (e.g. `scikit-learn==1.6.1`), pip installs whatever
> is latest today — which may behave differently next month. Pinning
> ensures the same result on every machine, forever.

> **Why these specific versions?**
> We're on Python 3.13. Older versions like scikit-learn 1.5.x don't
> have pre-built wheels for Python 3.13 and would fail to install
> without a C compiler.

### Step 1.4 — Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 5. Day 2 — Data Preprocessing

**File:** `src/preprocess.py`

### What preprocessing does

Raw data from the real world has three problems ML models can't handle:
1. **Missing/bad values** — e.g. TotalCharges has blank spaces instead of 0
2. **Text categories** — e.g. "Month-to-month". Models only understand numbers
3. **Different scales** — tenure is 0-72, charges are 20-120. Large numbers
   dominate small ones during training

Preprocessing fixes all three before the data reaches the model.

### The data quality issue we fixed

`TotalCharges` contains spaces `" "` for brand-new customers who haven't
been billed yet. It looks numeric but is actually stored as text with blanks.

```python
# Fix: strip whitespace, replace empty string with "0", convert to float
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = df["TotalCharges"].replace("", "0")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
```

### Encoding categories

ML models only understand numbers. We convert text to numbers two ways:

**Binary columns** (only "Yes" or "No"):
```python
df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
# "Yes" → 1, "No" → 0
```

**Multi-class columns** (3+ unique values):
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Contract"] = le.fit_transform(df["Contract"])
# "Month-to-month" → 0, "One year" → 1, "Two year" → 2
```

### Why we SAVE the encoders to disk

```python
joblib.dump(encoders, "model/encoders.pkl")  # during training
encoders = joblib.load("model/encoders.pkl") # during inference
```

During training, LabelEncoder decides: "Month-to-month"=0, "One year"=1.
If we re-fit on new data later, it might assign different numbers!
"Month-to-month" could become 1 instead of 0 — completely breaking predictions.

Saving the encoder means the API always uses the same mapping as training.

### Scaling numbers

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(...)
```

StandardScaler converts each value to a Z-score:
```
scaled = (value - mean) / standard_deviation
```

After scaling, every numeric column has mean=0 and std=1. The model treats
all features equally regardless of their original range.

### fit=True vs fit=False

This is the most important concept in preprocessing:

```python
def encode(df, fit=True):
    if fit:
        # TRAINING: learn the mapping from data, save it
        le.fit_transform(df["Contract"])
        joblib.dump(le, "model/encoders.pkl")
    else:
        # INFERENCE: load saved mapping, apply it
        le = joblib.load("model/encoders.pkl")
        le.transform(df["Contract"])
```

- `fit=True` → called once during training
- `fit=False` → called for every prediction in the API

---

## 6. Day 3 — Model Training

**File:** `src/train.py`

### How to run

```powershell
cd churn-radar
python src/train.py
```

### What training does

```
Load data → Preprocess → Split 80/20 → Train RF → Train XGB → Compare → Save best
```

### Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% held out for testing
    random_state=42,     # same split every run
    stratify=y           # preserve 26.5% churn ratio in both splits
)
```

**Why stratify?** Without it, by random chance the test set might have very
few churners (say, 5% instead of 26%). That would make our evaluation
unreliable. `stratify=y` guarantees the ratio matches the full dataset.

### Model 1 — Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,        # build 200 decision trees
    max_depth=10,            # limit each tree's depth (prevents memorizing)
    class_weight="balanced", # give more importance to the rare churn class
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
```

**What is a Random Forest?**
Builds many Decision Trees, each trained on a random subset of the data.
Final prediction = majority vote across all trees.
- "Random" = each tree sees random rows + random features
- "Forest" = many trees

**Why `class_weight="balanced"`?**
73.5% of customers don't churn. Without this, the model can get 73%
accuracy by always predicting "No churn" — which is useless. `balanced`
automatically upweights the minority class (churners).

### Model 2 — XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,       # how fast the model learns
    scale_pos_weight=2.77,   # ratio of negatives/positives (handles imbalance)
    random_state=42
)
```

**What is XGBoost?**
Builds trees sequentially — each tree learns from the mistakes of the
previous one. This is called "gradient boosting." Usually more accurate
than Random Forest on tabular data.

### Evaluation metrics

```
Accuracy: 77.0%   ← % of all predictions that were correct
F1 Score: 0.63    ← balance of precision and recall for the churn class
ROC-AUC:  0.84    ← how well the model separates churners from non-churners
```

**Why not just use accuracy?**
73.5% of customers don't churn. A model that always predicts "No churn"
gets 73.5% accuracy — but catches exactly 0 churners. Useless!

F1 and ROC-AUC both penalize this behavior. ROC-AUC of 0.84 means our
model correctly ranks churners above non-churners 84% of the time.

**Confusion matrix explained:**
```
              Predicted: Stay   Predicted: Churn
Actual: Stay       TN=809            FP=226
Actual: Churn      FN=98             TP=276
```
- **TP (276):** correctly identified churners → retention team contacts them ✓
- **TN (809):** correctly identified loyal customers ✓
- **FP (226):** loyal customers flagged as churners → unnecessary contact
- **FN (98):** real churners we missed → they leave undetected ✗

### Saving the model

```python
import joblib
joblib.dump(best_model, "model/best_model.pkl")
```

`joblib.dump()` serializes (converts) the Python model object into a binary
file. `joblib.load()` reads it back. This means:
- Training happens once (takes minutes)
- The API loads the saved model at startup (takes milliseconds)
- No retraining needed for every prediction

### Feature importances (from the output)

```
tenure                    0.1939  #########
Contract                  0.1815  #########
MonthlyCharges            0.1718  ########
TotalCharges              0.1502  #######
InternetService           0.0908  ####
PaymentMethod             0.0561  ##
```

These confirm what EDA showed: contract type and tenure are the biggest
drivers of churn. This is a real business insight, not just a number.

---

## 7. Day 4 — FastAPI Application

**File:** `app/main.py`

### How to run

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://localhost:8000/docs

### What is FastAPI?

FastAPI is a Python library that turns Python functions into HTTP endpoints.
An HTTP endpoint is a URL that responds to requests.

Without FastAPI → model only works by running Python scripts manually.
With FastAPI → anyone with internet access can get predictions via HTTP.

### Pydantic schemas — input validation

```python
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=120)       # required, must be 0-120
    MonthlyCharges: float = Field(..., ge=0)     # required, must be >= 0
    Contract: str = Field(...)                   # required
    # ... more fields
```

**What Pydantic does:**
When a request comes in, Pydantic automatically:
1. Checks all required fields are present
2. Checks each value has the right type (int, float, str)
3. Checks constraints (ge=0 means "greater than or equal to 0")
4. Rejects bad requests with a 422 error before the model ever sees them

Without Pydantic: garbage in → garbage predictions (or crashes)
With Pydantic: garbage in → clean error message → user fixes their request

### The endpoints

```python
@app.get("/health")
def health_check():
    # Returns: {"status": "healthy", "model_loaded": true, "version": "1.0.0"}
    # Used by: Docker, Render, and monitoring tools to check if we're alive

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    # Input:  customer data as JSON
    # Output: {"churn": true, "churn_probability": 0.82, "risk_level": "High"}

@app.post("/predict/batch")
def predict_churn_batch(batch: BatchCustomerFeatures):
    # Input:  list of up to 100 customers
    # Output: list of predictions + summary counts
```

### Lifespan — load model once at startup

```python
@asynccontextmanager
async def lifespan(app):
    initialize_predictor()  # loads model from disk ONCE
    yield                   # server runs here, handling requests
    # shutdown code here
```

**Why load at startup, not per request?**
Loading `best_model.pkl` from disk takes ~100ms.
Making a prediction with a loaded model takes ~1ms.
If we loaded per request, a busy server handling 100 requests/second
would spend 10 seconds per second just reading files. That's impossible.
Loading once at startup means all requests share the same in-memory model.

### Testing the API locally

**Option 1 — Browser (easiest):**
Open http://localhost:8000/docs → click any endpoint → "Try it out"

**Option 2 — PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"tenure":2,"MonthlyCharges":89.5,"TotalCharges":179.0,"Contract":"Month-to-month","PaymentMethod":"Electronic check","InternetService":"Fiber optic"}'
```

**Option 3 — curl (if installed):**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure":2,"MonthlyCharges":89.5,"TotalCharges":179.0,"Contract":"Month-to-month","PaymentMethod":"Electronic check","InternetService":"Fiber optic"}'
```

**High-risk customer response:**
```json
{
  "churn": true,
  "churn_probability": 0.8202,
  "risk_level": "High",
  "message": "This customer has a high likelihood of churning. Immediate retention action recommended."
}
```

**Low-risk customer response:**
```json
{
  "churn": false,
  "churn_probability": 0.08,
  "risk_level": "Low",
  "message": "This customer is likely to stay. No immediate action needed."
}
```

---

## 8. Day 5 — Docker

**File:** `Dockerfile`

### What Docker solves

**The problem:** "It works on my machine."
- Your laptop: Python 3.13, scikit-learn 1.6.1, Windows
- Server: Python 3.9, scikit-learn 1.3.0, Linux
- Result: different behavior, crashes, wrong predictions

**The solution:** Package your app + Python version + all libraries into
one container that runs identically everywhere.

### Key concepts

| Term | Analogy | What it means |
|------|---------|---------------|
| Image | Recipe / Class | Blueprint for a container |
| Container | Dish / Instance | A running copy of an image |
| Dockerfile | Instructions | How to build the image |
| Layer | Steps in recipe | Each Dockerfile instruction creates a cached layer |

### The Dockerfile explained line by line

```dockerfile
FROM python:3.11-slim
```
Start from an official Python 3.11 image (pre-built Linux + Python).
`-slim` = minimal OS, no unnecessary tools. Keeps image small.

```dockerfile
WORKDIR /app
```
All subsequent commands run from the `/app` directory inside the container.

```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```
**Why copy requirements BEFORE code?**
Docker caches each step. If we copied all code first:
- Change one line of code → Docker re-runs pip install (slow, 2+ minutes)

By copying requirements.txt first:
- Change code only → Docker reuses cached pip install layer (fast, seconds)
- Change requirements → Docker re-runs pip install (unavoidable)

```dockerfile
COPY src/ ./src/
COPY app/ ./app/
COPY model/ ./model/
```
Copy only what the container needs to run. Tests, notebooks, and data
are excluded by `.dockerignore`.

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s CMD ...
```
Every 30 seconds, Docker checks if `/health` responds. If it fails 3 times,
Docker marks the container "unhealthy" and can restart it automatically.

```dockerfile
USER appuser
```
Running as root inside a container is a security risk. We create a
dedicated non-root user to run the app.

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```
The command that runs when the container starts.
- `--host 0.0.0.0` — listen on all interfaces (required to receive external traffic)
- `--workers 2` — handle 2 requests simultaneously

### Build the image

```powershell
docker build -t churn-radar .
```

- `docker build` — reads the Dockerfile and creates an image
- `-t churn-radar` — tag (name) the image "churn-radar"
- `.` — use the current directory as the build context

This takes 3-5 minutes the first time (downloads Python base image,
installs all packages). Subsequent builds are faster due to layer caching.

### Run the container

```powershell
docker run -d --name churn-api -p 8000:8000 churn-radar
```

- `-d` — detached mode (runs in background)
- `--name churn-api` — give the container a name
- `-p 8000:8000` — map port 8000 on your machine to port 8000 in container
- `churn-radar` — the image to run

### Useful Docker commands

```powershell
docker ps                    # list running containers
docker logs churn-api        # see container output/logs
docker stop churn-api        # stop the container
docker rm churn-api          # delete the container
docker images                # list all images on your machine
```

### Verify the container works

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
# Should return: {"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

---

## 9. Day 6 — GitHub

### What is Git and why use it?

Git tracks every change you make to your code over time.
- Each "commit" is a saved snapshot you can return to
- GitHub hosts your repo online so others (and recruiters) can see it

### Step 9.1 — Create a GitHub repo

1. Go to https://github.com/new
2. Name: `churn-radar`
3. Visibility: **Public** (portfolio projects must be public)
4. Do NOT check any boxes (no auto-generated README)
5. Click **Create repository**

### Step 9.2 — Initialize and push

```powershell
cd churn-radar

git init                            # create empty .git folder
git add .                           # stage all files
git status                          # verify what's staged (check .pkl files are excluded)
git commit -m "Initial commit: Churn Radar — FastAPI + Docker + scikit-learn MLOps project"
git branch -M main                  # rename branch to "main"
git remote add origin https://github.com/tusharaloorkar/churn-radar.git
git push -u origin main             # push to GitHub
```

### What .gitignore excludes and why

```
data/raw/*.csv      → datasets can be GBs, not for Git
model/*.pkl         → binary files, hard to diff, tracked with DVC later
venv/               → OS-specific, huge, auto-generated
.env                → NEVER commit secrets
__pycache__/        → auto-generated bytecode
```

Note: `model/metrics.json` IS committed — it's small text and useful for
showing training results on GitHub.

### Every time you make changes

```powershell
git add .
git commit -m "describe what you changed"
git push
```

---

## 10. Day 7 — Deploy to Render

### What is Render?

Render is a cloud platform that runs your Docker container on their servers.
The free tier gives you a public URL and is sufficient for portfolio projects.

**Free tier limitation:** The server "sleeps" after 15 minutes of inactivity.
The first request after sleeping takes ~50 seconds to wake up. This is normal
for free hosting — paid plans eliminate this.

### Step 10.1 — Sign up

Go to https://render.com → sign up with GitHub account.

### Step 10.2 — Create Web Service

1. Click **New +** → **Web Service**
2. Click **Connect a repository**
3. Select `tusharaloorkar/churn-radar`

### Step 10.3 — Configure settings

| Setting | Value |
|---------|-------|
| Name | `churn-radar` |
| Branch | `main` |
| Runtime | **Docker** |
| Instance Type | **Free** |

Everything else: leave as default.

### Step 10.4 — Deploy

Click **Create Web Service**. Render will:
1. Clone your GitHub repo
2. Run `docker build` using your Dockerfile
3. Start the container
4. Assign a URL: `https://churn-radar.onrender.com`

Wait 5-8 minutes. Watch the logs — you should see:
```
[predict] Model loaded: RandomForestClassifier
[predict] Ready to make predictions.
[startup] API is ready to serve predictions.
==> Your service is live
```

### Auto-deploys

Every time you push to GitHub (`git push`), Render automatically:
1. Detects the new commit
2. Rebuilds the Docker image
3. Deploys the new version

No manual steps needed after the initial setup.

---

## 11. Testing the Live API

### Health check
```powershell
Invoke-RestMethod -Uri "https://churn-radar.onrender.com/health"
```
Response:
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

### High-risk customer (month-to-month, fiber optic, new customer)
```powershell
Invoke-RestMethod -Uri "https://churn-radar.onrender.com/predict" `
    -Method POST -ContentType "application/json" `
    -Body '{"tenure":2,"MonthlyCharges":89.5,"TotalCharges":179.0,"Contract":"Month-to-month","PaymentMethod":"Electronic check","InternetService":"Fiber optic"}'
```
Response:
```json
{
  "churn": true,
  "churn_probability": 0.8202,
  "risk_level": "High",
  "message": "This customer has a high likelihood of churning. Immediate retention action recommended."
}
```

### Low-risk customer (two-year contract, long tenure)
```powershell
Invoke-RestMethod -Uri "https://churn-radar.onrender.com/predict" `
    -Method POST -ContentType "application/json" `
    -Body '{"tenure":60,"MonthlyCharges":45.0,"TotalCharges":2700.0,"Contract":"Two year","PaymentMethod":"Bank transfer (automatic)","InternetService":"DSL"}'
```
Response:
```json
{
  "churn": false,
  "churn_probability": 0.08,
  "risk_level": "Low",
  "message": "This customer is likely to stay. No immediate action needed."
}
```

### Interactive docs
Open in browser: https://churn-radar.onrender.com/docs

This is auto-generated by FastAPI — no extra work required. It lets anyone
explore and test all endpoints without writing any code.

---

## 12. Troubleshooting

### "ModuleNotFoundError: No module named 'joblib'"
**Cause:** Dependencies not installed.
**Fix:**
```powershell
pip install -r requirements.txt
```

### "scikit-learn fails to install"
**Cause:** Pinned version doesn't have a pre-built wheel for your Python version.
**Fix:** Use versions compatible with your Python:
- Python 3.13 requires scikit-learn >= 1.6.0 and numpy >= 2.0

### "UnicodeEncodeError: 'charmap' codec can't encode character"
**Cause:** Windows terminal (cp1252) can't print Unicode symbols like → or █.
**Fix:** Replace Unicode characters with ASCII equivalents:
- `→` → `->`
- `█` → `#`
- `—` → `-`

### "FileNotFoundError: Model not found at 'model/best_model.pkl'"
**Cause:** You haven't trained the model yet, or you're running from the wrong directory.
**Fix:**
```powershell
cd churn-radar          # must be in the project root
python src/train.py     # train first
```

### "Port 8000 already in use"
**Cause:** Another process (uvicorn or Docker) is already using port 8000.
**Fix:**
```powershell
docker stop churn-api   # stop Docker container if running
# or use a different port:
uvicorn app.main:app --port 8001
```

---

## 13. Key Concepts Glossary

| Term | Simple Explanation |
|------|--------------------|
| **API** | A way for software to talk to other software over the internet via HTTP |
| **REST** | A style of API design using standard HTTP methods (GET, POST, etc.) |
| **Endpoint** | A specific URL that your API responds to (e.g. `/predict`) |
| **JSON** | A text format for sending structured data over HTTP (like a Python dict) |
| **Inference** | Using a trained model to make predictions on new data |
| **Training** | Teaching a model by showing it thousands of labeled examples |
| **Serialization** | Saving a Python object (like a model) to a file so it can be loaded later |
| **Docker Image** | A packaged blueprint of your app + dependencies (like a template) |
| **Docker Container** | A running instance of an image (like an object from a class) |
| **Port mapping** | `-p 8000:8000` links port 8000 on your machine to port 8000 in the container |
| **Pydantic** | A Python library that validates data types and constraints automatically |
| **ROC-AUC** | A metric measuring how well a model separates two classes (0.5=random, 1.0=perfect) |
| **F1 Score** | A metric balancing precision and recall — better than accuracy for imbalanced data |
| **StandardScaler** | Transforms features to have mean=0 and std=1 so all features are equally weighted |
| **LabelEncoder** | Converts text categories ("Yes", "No") to numbers (1, 0) |
| **Overfitting** | When a model memorizes training data instead of learning patterns — performs badly on new data |
| **Class imbalance** | When one outcome (No churn: 73%) is much more common than another (Churn: 27%) |
| **Stratify** | Ensuring the train/test split preserves the same class ratio as the full dataset |
| **CI/CD** | Continuous Integration / Deployment — automatically test and deploy on every code push |
| **Git commit** | A saved snapshot of your code at a specific point in time |
| **pip** | Python's package manager — installs libraries listed in requirements.txt |

---

## What's Next — Project 2

Project 2 builds on this foundation by adding:

1. **MLflow experiment tracking**
   - Every `python src/train.py` run logs parameters, metrics, and the model
   - A web UI lets you compare all runs side by side
   - Promotes the best run to "production"

2. **GitHub Actions CI/CD**
   - Every `git push` triggers automatic tests (`pytest`)
   - If tests pass, automatically rebuilds the Docker image
   - If tests fail, the push is flagged — broken code never reaches production

These are the two things that turn a "data scientist who can code"
into an "ML engineer who can ship."

---

*Guide written alongside the project build — May 2026*
*Stack: Python 3.13 · FastAPI · scikit-learn · XGBoost · Docker · Render*
