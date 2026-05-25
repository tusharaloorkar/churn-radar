# =============================================================================
# Dockerfile — Container Build Instructions
# =============================================================================
#
# WHAT IS DOCKER?
#   Docker packages your application and ALL its dependencies (Python version,
#   libraries, config files) into a single portable unit called a "container".
#
#   Without Docker, "it works on my machine" is a constant problem:
#     - Your laptop runs Python 3.11, the server runs 3.9 → different behavior
#     - You have scikit-learn 1.5.0, a teammate has 1.3.0 → different results
#
#   With Docker, everyone runs the EXACT same environment everywhere.
#
# WHAT IS A Dockerfile?
#   A Dockerfile is a recipe that tells Docker HOW to build your container.
#   Each line is an instruction. Docker runs them top-to-bottom and creates
#   a layered "image". When you run the image, it becomes a "container".
#
# KEY CONCEPTS:
#   Image  = the recipe (like a template or class in OOP)
#   Container = a running instance of an image (like an object instance)
#
# HOW TO BUILD AND RUN:
#   docker build -t churn-predictor .
#   docker run -p 8000:8000 churn-predictor
#   Then open: http://localhost:8000/docs
# =============================================================================

# --- Base Image ---
# Every Dockerfile starts with a base image — a pre-built Linux environment.
# We're using the official Python 3.11 image with "-slim" (minimal OS,
# no unnecessary tools) to keep our final image small.
#
# Think of it like: instead of building a house from scratch, we start
# with a pre-built foundation (Python + Linux) and add our stuff on top.
FROM python:3.11-slim

# --- Metadata ---
# Labels are optional metadata about the image. Not functional, but good practice.
LABEL maintainer="your-email@example.com"
LABEL description="Customer Churn Prediction API"
LABEL version="1.0.0"

# --- Working Directory ---
# Set the working directory INSIDE the container.
# All subsequent commands run relative to this path.
# /app is a convention for web applications.
WORKDIR /app

# --- System Dependencies ---
# Install OS-level packages needed by our Python libraries.
# XGBoost and scikit-learn need these C/C++ libraries to compile.
#
# --no-install-recommends: don't install optional packages (saves space)
# rm -rf /var/lib/apt/lists/*: delete package index cache after install
#   (reduces image size — Docker layers are permanent, so we clean up
#   in the same RUN command to avoid bloating the layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python Dependencies ---
# WHY COPY requirements.txt BEFORE the rest of the code?
#   Docker caches each layer. If we copy all files first, then pip install,
#   ANY code change invalidates the pip install cache — even if requirements
#   didn't change. This is slow (pip install takes 1-3 minutes).
#
#   By copying requirements.txt first:
#     - If only code changed → Docker reuses the cached pip install layer ✓
#     - If requirements changed → Docker re-runs pip install (unavoidable)
#   This "dependency-first" pattern is a Docker best practice for fast rebuilds.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
#   --no-cache-dir: don't save downloaded packages to disk (reduces image size)
#   --upgrade pip: ensure we have the latest pip version

# --- Application Code ---
# Now copy the rest of the application. This layer is rebuilt on every code
# change, but it's fast because it's just file copies (no compilation).
COPY src/ ./src/
COPY app/ ./app/
COPY model/ ./model/
#   Note: we copy the pre-trained model into the image so the API can
#   load it immediately. Alternative: load from cloud storage at runtime.

# --- Create empty __init__.py files ---
# Python needs __init__.py in directories to treat them as packages
# (so 'from src.preprocess import ...' works inside the container).
RUN touch src/__init__.py app/__init__.py

# --- Port ---
# Document which port the application listens on.
# EXPOSE is documentation only — it doesn't actually open the port.
# The actual port binding happens in 'docker run -p 8000:8000'
EXPOSE 8000

# --- Health Check ---
# Docker periodically runs this command to verify the container is healthy.
# If it fails, Docker marks the container as unhealthy.
# Render and other cloud platforms use this to restart broken containers.
#
# --interval=30s: check every 30 seconds
# --timeout=10s: fail if no response within 10 seconds
# --start-period=30s: give the app 30 seconds to start before checking
# --retries=3: mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# --- Non-root User ---
# Running as root inside a container is a security risk — if an attacker
# escapes the container, they'd have root on the host system.
# Best practice: create a dedicated non-root user for running the app.
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# --- Start Command ---
# CMD defines the default command to run when the container starts.
# This runs our FastAPI app using Uvicorn as the server.
#
# --host 0.0.0.0: listen on ALL network interfaces inside the container
#   (necessary so requests from outside the container can reach it)
# --port 8000: the port to listen on (must match EXPOSE above)
# --workers 2: run 2 worker processes (handle 2 requests simultaneously)
#   In production, set this to (2 × CPU_cores + 1)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
