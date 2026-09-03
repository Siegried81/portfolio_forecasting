# Streamlit Cloud Deployment (Free)

## Overview

**Streamlit Cloud** is the fastest way to deploy this app for free. It handles all infrastructure; you just push to GitHub.

| Feature | Free Tier |
|---------|-----------|
| Apps | Unlimited |
| Compute | Shared VM (~1GB RAM, auto-sleep after 1 hour inactivity) |
| Public access | Yes |
| Custom domain | No (gets subdomain like `your-name-portfolio-forecasting.streamlit.app`) |
| Secrets management | Yes (environment variables) |
| Cost | **Free** |

---

## Step-by-Step Setup

### 1. Prepare Your Repository

**Commit your code to GitHub:**

```bash
git add .
git commit -m "Add Streamlit Cloud deployment config"
git push origin main
```

Ensure these files are present and committed:
- `app.py` ✓
- `requirements.txt` ✓
- `src/` directory ✓
- `.streamlit/secrets.toml` (optional locally, required in Streamlit Cloud)
- `.gitignore` should exclude: `.env`, `.venv`, `.streamlit/secrets.toml`

**Verify `.gitignore` includes secrets:**

```bash
echo ".streamlit/secrets.toml" >> .gitignore
git add .gitignore
git commit -m "Exclude Streamlit Cloud secrets from git"
git push
```

### 2. Sign Up & Connect Repository

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"** → **"From existing repo"**
3. Connect your GitHub account (authorize Streamlit)
4. Select your repo: `portfolio-forecasting`
5. Branch: `main` (or whatever your default is)
6. Main file path: `app.py`
7. Click **Deploy**

Streamlit will:
- Clone your repo
- Install `requirements.txt`
- Run `streamlit run app.py`
- Assign a public URL like: `https://<your-username>-portfolio-forecasting.streamlit.app`

**Deployment takes ~3–5 minutes** (first time installs all dependencies).

---

## 3. Add Secrets (API Keys)

**In Streamlit Cloud Dashboard:**

1. Go to your app's page → **Settings** (⚙️ icon, top right)
2. Click **Secrets** tab
3. Paste your API keys in TOML format:

```toml
GROQ_API_KEY = "gsk_..."
NEWSAPI_KEY = "abc123..."
FRED_API_KEY = "xyz789..."
TWELVEDATA_API_KEY = "..."
```

**Or use GitHub Secrets** (more secure for CI/CD):
- Add the secrets to your GitHub repo (Settings → Secrets and variables)
- Streamlit automatically reads them if named `STREAMLIT_*`

**Important:** Never commit real API keys to GitHub. The `.streamlit/secrets.toml` file in your repo should only have template placeholders (empty strings).

---

## Testing Locally Before Deployment

### Test with Streamlit CLI

```bash
# Activate your venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run the app locally (same as Streamlit Cloud will)
streamlit run app.py
```

This opens `http://localhost:8501` and hot-reloads on file changes.

### Test with Local `.env` File

Create `.env` with your real API keys (this file is in `.gitignore`, so it won't push):

```bash
GROQ_API_KEY=gsk_...
NEWSAPI_KEY=...
FRED_API_KEY=...
TWELVEDATA_API_KEY=...
```

Then run:
```bash
streamlit run app.py
```

The app will load these from `.env` via `python-dotenv` (see `src/config.py`).

### Verify All Tabs Load

- **Overview tab** — should load prices and metrics without API keys
- **Efficient Frontier tab** — should compute weights (no keys needed)
- **Forecast & Compare tab** — should run walk-forward (may be slow on shared VM)
- **AI Analyst tab** — will show "unavailable" without `GROQ_API_KEY`, but that's OK (fallback to Ollama, which won't be available on Streamlit Cloud)

---

## After Deployment

### Redeployment

Every time you push to GitHub (on the branch you selected), Streamlit automatically redeploys:

```bash
git add .
git commit -m "Feature: add max weight constraint"
git push origin main
```

Redeployment takes ~2 minutes. Secrets persist across deploys.

### Monitor Logs

Click your app name on the Streamlit Cloud dashboard → **Manage app** → **Logs**

Shows real-time Streamlit output, errors, and API calls.

### Troubleshooting on Streamlit Cloud

#### App won't deploy
- **Check logs**: Streamlit Cloud → Logs tab
- **Common causes:**
  - Missing `requirements.txt` — fix: `pip freeze > requirements.txt`
  - Import error in `app.py` — test locally first with `streamlit run app.py`
  - Outdated dependency pin — update `requirements.txt` to remove exact versions

#### API keys not loading
- **Verify secrets are set**: Dashboard → Settings → Secrets
- **Check syntax**: Must be TOML format (not Python dict)
- **App reload**: Streamlit Cloud caches; wait ~1 minute after adding secrets, then refresh browser

#### App runs slow
- **Forecast + Compare tab on Streamlit Cloud**: Uses shared 1GB VM; ARIMA on many windows can timeout
- **Workaround**: 
  - Use ETS or Naive forecasting model (much faster)
  - Limit walk-forward windows to 3–5
  - Use a shorter date range to reduce data

#### "ModuleNotFoundError"
- Likely an old cached build. In Streamlit Cloud dashboard:
  - Click your app → **Settings** → **Reboot app**
  - Or delete and redeploy

---

## Cost Comparison

| Platform | Cost | Build Time | Auto-reload | Secrets | Best For |
|----------|------|-----------|------------|---------|----------|
| **Streamlit Cloud** | Free | 3–5 min | On push | Yes | Rapid prototyping, demos, free tier |
| **Render** | Free tier + $7/mo | 2–3 min | Manual redeploy | Yes | Longer runtime, custom domain (paid) |
| **AWS/GCP/Azure** | ~$10–50/mo | <1 min | Yes (CI/CD) | Yes | Production, scaling, custom infra |
| **Docker Desktop** | Free (local) | 3–5 min | On rebuild | File-based | Development, testing |

---

## Advanced: GitHub Actions Auto-Deploy

For production, trigger Streamlit Cloud deploys from GitHub Actions:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger Streamlit Cloud deploy
        run: |
          curl -X POST https://share.streamlit.io/api/deploy \
            -H "Authorization: Bearer ${{ secrets.STREAMLIT_API_TOKEN }}" \
            -H "Content-Type: application/json" \
            -d '{"repo": "${{ github.repository }}","branch": "main"}'
```

(Requires `STREAMLIT_API_TOKEN` secret from Streamlit Cloud dashboard)

---

## Quick Summary

```bash
# 1. Commit and push to GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# 2. Go to https://share.streamlit.io → New app → Select your repo
# 3. Add secrets in Streamlit Cloud dashboard
# 4. Done! App is live at https://<username>-portfolio-forecasting.streamlit.app
```

---

## Next Steps

- **Monitor**: Check logs if something goes wrong
- **Iterate**: Push updates to GitHub, Streamlit auto-redeploys
- **Scale**: If you hit rate limits on free APIs, upgrade API keys or move to Render/AWS
- **Custom domain**: Upgrade to Streamlit Cloud Pro ($99/mo) or use Render with a paid tier

---

For full Streamlit Cloud docs: https://docs.streamlit.io/deploy/streamlit-cloud/
