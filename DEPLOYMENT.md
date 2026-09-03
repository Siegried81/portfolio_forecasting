# Docker Deployment Guide — Portfolio Forecasting

## Quick Start

### Prerequisites
- Docker Desktop installed and running
- `.env` file configured (copy from `.env.example` and add your API keys)

### macOS / Linux
```bash
bash deploy.sh
```

### Windows (PowerShell)
```powershell
.\deploy.bat
```

Then open: **http://localhost:8501**

---

## What's Been Set Up

### Files Created / Modified

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-layer Python image, Streamlit on port 8501 with healthcheck |
| `docker-compose.yml` | Single-service compose with env passthrough, restart policy, healthcheck |
| `.dockerignore` | Excludes `.venv`, `__pycache__`, `.git` to speed up builds |
| `deploy.sh` | Bash script for macOS/Linux deployment |
| `deploy.bat` | Batch script for Windows deployment |

### Build Details

- **Base image**: `python:3.12-slim` (lightweight, 250MB)
- **Build time**: ~3–5 minutes (first build only; cached layers reuse on rebuilds)
- **Final image size**: ~1.2–1.5 GB (large due to scipy, statsmodels, cvxpy dependencies)
- **Layer caching**: `requirements.txt` copied separately so code changes don't invalidate pip cache

### Key Modifications from Your Code

Your modified `src/config.py` now includes:
- `DEFAULT_MAX_WEIGHT_PER_ASSET = 0.35` — global constraint on portfolio concentration
- Integration into sidebar UI (`app.py` line 126) — user-adjustable from 10% to 100%
- Passed to all three portfolio optimizations (historical, forecast, realized)

This prevents extreme single-name concentration in the efficient frontier and makes the three-portfolio comparison more interpretable (less likely for one window's realized-optimal to put 100% into a lucky winner).

---

## Running the Container

### With Docker Compose (Recommended)

```bash
docker compose up -d --pull always
```

- `-d`: Run in background
- `--pull always`: Always pull the latest base image (good for reproducibility)

**Check status:**
```bash
docker ps | grep portfolio-forecasting
docker compose logs -f  # Stream logs
```

**Stop:**
```bash
docker compose down
```

### With Docker Run (Direct)

```bash
docker run -d \
  --name portfolio-forecasting \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  portfolio-forecasting
```

---

## Environment Variables

All optional API keys go in `.env` (never hardcode or commit real keys):

```bash
# AI Analyst tab (Groq primary LLM)
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b  # Default; see .env.example for current

# Local fallback if Groq is rate-limited
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1

# News digest
NEWSAPI_KEY=

# Live Treasury yields for macro context
FRED_API_KEY=

# Market data fallback (Yahoo Finance → Twelve Data)
TWELVEDATA_API_KEY=
```

**Without keys:**
- App still runs fully (market data, optimization, forecasting work)
- AI Analyst tab shows "unavailable" until Ollama is running locally or Groq is configured
- News digest won't fetch; macro yields fall back to fixed 4% default

---

## Troubleshooting

### Build is very slow / times out

**Cause:** First build requires downloading and compiling scipy, statsmodels, cvxpy from source (35–50 MB download + build time).

**Fix:**
- Ensure you have at least 10 GB free disk space
- Check internet speed (build downloads ~100 MB)
- Run `docker builder prune` if you have many old builds cached
- Consider letting it run overnight if your connection is slow

Once built, layers are cached — rebuilds after code changes take <1 minute.

### Container exits immediately

**Check logs:**
```bash
docker compose logs portfolio-forecasting
```

**Common causes:**
- `ModuleNotFoundError`: Missing dependency — rebuild with `docker compose build --no-cache`
- `GROQ_API_KEY` invalid: App logs warning but continues (Ollama fallback kicks in)
- Port 8501 already in use: Change `ports: ["8502:8501"]` in `docker-compose.yml`

### Can't connect to http://localhost:8501

1. **Verify container is running:**
   ```bash
   docker ps | grep portfolio-forecasting
   ```

2. **Check port mapping:**
   ```bash
   docker port portfolio-forecasting
   ```
   Should show `8501/tcp -> 0.0.0.0:8501`

3. **Verify healthcheck:**
   ```bash
   docker inspect portfolio-forecasting | grep -A 5 '"State"'
   ```
   If `"Running": false`, see "Container exits immediately" above.

4. **Wait for startup:**
   Streamlit takes 10–15 seconds to initialize. Check logs:
   ```bash
   docker compose logs -f | grep -i "streamlit\|running"
   ```

### High memory/CPU usage during build

Normal for first build. Pip is resolving ~100 direct/transitive dependencies. Once built, the running container uses ~300–400 MB RAM at idle.

---

## Production Deployment

### Render (Recommended for this app)

`render.yaml` already exists in your repo:
1. Push to GitHub
2. Connect repo to Render
3. Render auto-deploys from `Dockerfile`
4. Set `GROQ_API_KEY`, `NEWSAPI_KEY` in Render dashboard → Environment

### Streamlit Community Cloud (Simpler but less control)

1. Push repo to GitHub (without `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in app Settings

### Docker Swarm / Kubernetes

For multi-container orchestration, deploy `docker-compose.yml` with:
```bash
docker stack deploy -c docker-compose.yml portfolio
```

---

## Performance Tips

### First-Time Users: Cold Build Optimization

If the initial build is taking too long and you're on a slow connection, you can pre-cache dependencies:

```bash
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t portfolio-forecasting .
```

Then push to a registry so others can pull the cached layers (skips the 3–5 minute build).

### Runtime Performance

- **Forecast + Compare tab**: ARIMA is slow on many windows. Switch to ETS or Naive for iteration.
- **Walk-forward backtest**: Limit to 5 windows unless you're overnight-running a full analysis.
- **Memory**: If running on a machine with <2GB free, the container may OOM. Add swap or increase Docker's memory limit.

---

## CI/CD Integration

### GitHub Actions Example (push → build → test)

```yaml
name: Build & Test

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: portfolio-forecasting:latest
```

---

## Questions?

See the main README.md for methodology, scope, and feature documentation.

For deployment issues, check:
1. Docker Desktop status (Settings → Resources)
2. Disk space: `docker system df`
3. Network: `docker pull python:3.12-slim` (basic connectivity test)
4. Logs: `docker compose logs -f`
