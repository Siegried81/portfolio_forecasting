# Deployment Quick Reference — Docker + Streamlit Cloud

## ✅ Option 1: Docker (Recommandé pour développement/VPS)

### Démarrer le container
```bash
# Build (une seule fois)
docker build -t portfolio-forecasting .

# Lancer
docker run -d --name portfolio-forecasting -p 8501:8501 portfolio-forecasting

# Ou avec docker-compose
docker compose up -d
```

**Accès:**
- Local: http://localhost:8501
- Via réseau: http://<your-ip>:8501

**Commandes utiles:**
```bash
# Voir logs
docker logs portfolio-forecasting -f

# Arrêter
docker stop portfolio-forecasting

# Nettoyer
docker rm portfolio-forecasting
```

**Avantages:**
- ✓ Complément contrôle
- ✓ Peut tourner sur un VPS / machine locale
- ✓ Pas de limitation de ressources partagées

---

## ✅ Option 2: Streamlit Cloud (Gratuit, auto-redeploy)

### 1. Push à GitHub
```bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### 2. Streamlit Cloud Dashboard
- Allez à https://share.streamlit.io
- Cliquez "New app" → "From existing repo"
- Sélectionnez `portfolio-forecasting`
- Déploiement se fait automatiquement en 3-5 min

### 3. Ajouter les secrets (API keys)
- Dashboard → Settings → Secrets
- Ajouter en TOML:
```toml
GROQ_API_KEY = "gsk_..."
NEWSAPI_KEY = "..."
FRED_API_KEY = "..."
TWELVEDATA_API_KEY = "..."
```

**Accès:**
- https://<username>-portfolio-forecasting.streamlit.app

**Avantages:**
- ✓ Zéro setup infrastructure
- ✓ Auto-redeploy sur chaque `git push`
- ✓ Gratuit (app illimitées)
- ✓ URL publique

---

## Comparaison

| Aspect | Docker | Streamlit Cloud |
|--------|--------|-----------------|
| **Setup** | Simple (docker run) | 2-3 clics + git push |
| **Coût** | Gratuit (tu host) | Gratuit (Streamlit host) |
| **Performance** | Dépend de ta machine | Shared 1GB VM (assez pour ce projet) |
| **Redeploy** | Manuel (`docker build`) | Auto (sur git push) |
| **URL** | localhost:8501 ou ton IP | Streamlit subdomain |
| **Secrets** | `.env` file | Dashboard Streamlit |
| **Maintenance** | Tu gères | Streamlit gère |

---

## Test Local Avant Deployment

```bash
# Activate venv
source .venv/bin/activate

# Create .env avec tes API keys (not committed)
echo "GROQ_API_KEY=..." > .env
echo "NEWSAPI_KEY=..." >> .env

# Test
streamlit run app.py
```

Puis tester chaque tab:
- **Overview** — charge les prix (pas besoin d'API keys)
- **Efficient Frontier** — optimise (pas besoin d'API keys)
- **Forecast & Compare** — run backtest
- **AI Analyst** — affiche "unavailable" sans GROQ_API_KEY (c'est OK, fallback graceful)

---

## Toi en ce moment

✅ **Docker build réussi** — l'image est prête  
✅ **Container en cours sur port 8502** — http://localhost:8502  
✅ **docker-compose.yml configuré** — pour faciliter les redeploys  
✅ **Streamlit Cloud prêt** — juste besoin de `git push`  

---

## Prochaines étapes

1. **Docker local (immédiat):**
   ```bash
   docker ps  # Vérifie le container
   # Ouvre http://localhost:8502 dans un browser
   ```

2. **Streamlit Cloud (optionnel mais recommandé):**
   ```bash
   git push origin main
   # Va automatiquement déployer en 3-5 min
   ```

3. **Arrêter le Docker si besoin:**
   ```bash
   docker stop portfolio-forecasting
   ```

---

Done! Tu as maintenant les deux déployés. 🚀
