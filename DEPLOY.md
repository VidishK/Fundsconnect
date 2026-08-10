# Deploy Fundsconnect (free) — Render

Netlify cannot run this app’s Python API. Use **Render** (steps below). Railway works the same idea with the `Procfile` / Dockerfile.

## Speed / “data not loading”

Free Render **spins down** after ~15 min idle. The first request after sleep can take **30–60 seconds** while the container boots.

After that, rankings should appear quickly. Live NAV fills in a few seconds later (intentionally deferred so it doesn’t block the table).

If the page still shows “Local mode” / demo data, wait for the yellow **Waking server** banner to finish retries, then hard-refresh. Keep the tab open once awake — subsequent loads are fast.

Optional Render env vars:
- `NAV_WORKERS=4` (default) — lower if the service OOMs during NAV refresh

## What you need
- GitHub account
- This project pushed to a **public or private** GitHub repo
- Free [Render](https://render.com) account

## 0. One-time: put the project on GitHub

In Terminal, from the project folder:

```bash
cd ~/Desktop/Fundsconnect
git init
git add Dockerfile .dockerignore render.yaml Procfile requirements.txt \
  app.py fund_scorer.py metrics.py data_source.py nav_source.py \
  holdings_source.py holdings_fetch.py pe_enrich.py overlap_engine.py \
  static funds.xlsx ranked.csv holdings.csv holdings_upload_TEMPLATE.csv \
  DEPLOY.md .gitignore
git commit -m "Deploy-ready Fundsconnect for Render"
```

Create a new empty repo on GitHub (e.g. `Fundsconnect`), then:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USER/Fundsconnect.git
git push -u origin main
```

(Replace `YOUR_USER` with your GitHub username.)

## 1. Deploy on Render (recommended)

1. Open https://dashboard.render.com → **New** → **Web Service**
2. Connect the **Fundsconnect** GitHub repo
3. Settings:
   - **Runtime:** Docker (Render detects `Dockerfile`)
   - **Instance type:** Free
   - **Health check path:** `/api/health`
4. Click **Create Web Service**
5. Wait for the first build (~3–8 min)
6. Open the URL Render gives you, e.g. `https://fundsconnect-xxxx.onrender.com`

Optional: **New** → **Blueprint** and select the repo — uses `render.yaml`.

## 2. After it’s live — what to click

1. Open the site → **Data** → **Fetch all funds** (if you want fresh holdings)
2. Then **Fill P/E & P/B** (valuations)
3. **Overlap** → pick funds

Live **NAV** loads automatically from mfapi.in when the Rankings page runs.

## 3. Free-tier limits (important)

| Reality | What it means |
|---------|----------------|
| Sleeps after ~15 min idle | First visit after sleep can take **30–60s** to wake |
| **Ephemeral disk** | Fetch / Fill / uploads are **lost on redeploy or sometimes on sleep** unless you keep files in git and redeploy |
| Build minutes | Free plan has monthly build limits — don’t spam redeploys |

**Tip:** After a good Fetch + Fill locally, commit an updated `holdings.csv` and push so the deployed snapshot stays useful after restarts.

## 4. Railway (alternative)

1. https://railway.app → New Project → Deploy from GitHub
2. Select the repo
3. Railway uses `Dockerfile` or `Procfile`
4. Set start command if needed:  
   `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 5. Local Docker test (optional)

```bash
cd ~/Desktop/Fundsconnect
docker build -t fundsconnect .
docker run --rm -p 8000:8000 fundsconnect
# open http://127.0.0.1:8000
```

## 6. Checklist before you share the link

- [ ] GitHub repo pushed with `funds.xlsx` + `holdings.csv`
- [ ] Render service green / healthy
- [ ] `/api/health` returns OK
- [ ] Rankings page loads
- [ ] Overlap sidebar shows funds
- [ ] (Optional) Data → Fetch + Fill P/E on the live site

## Files added for deploy

| File | Purpose |
|------|---------|
| `Dockerfile` | How the cloud builds and runs the app |
| `.dockerignore` | Keeps image small (skips caches / vendor junk) |
| `render.yaml` | One-click Render Blueprint |
| `Procfile` | Railway / Heroku-style start command |
| `DEPLOY.md` | This guide |
