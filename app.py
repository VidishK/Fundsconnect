"""
app.py  --  FastAPI backend for the fund-research website
================================================================
Serves the quantitative engine over HTTP and hosts the single-page
frontend.

Endpoints
---------
GET  /                         -> the website (static index.html)
GET  /api/health               -> {"status": "ok", ...}
GET  /api/profiles             -> list of risk profiles (incl. dial info)
GET  /api/glossary             -> every metric: formula, meaning, direction
GET  /api/funds                -> all funds with all metrics + VaR (no score)
GET  /api/rank?profile=balanced&category=...&min_history=r5&n=...
                               -> funds ranked under a named profile
GET  /api/rank/dial?t=0.6&category=...&n=...
                               -> funds ranked on the continuous risk dial
                                  (blended balanced -> aggressive)
GET  /api/fund/{name}          -> one fund: all metrics + score in every profile
GET  /api/overlap/status       -> holdings file present + counts
GET  /api/overlap/funds        -> funds available in holdings (+ AUM join)
GET  /api/overlap?funds=A,B&min_funds=2  -> ranked overlapping companies
GET  /api/overlap/company/{name} -> which funds hold a company + weights
POST /api/data/upload            -> upload Excel/CSV/JSON (preview or save)
GET  /api/data/status            -> current holdings store status
GET  /api/nav/live               -> start/refresh live NAV + progress + values
GET  /api/nav/status             -> progress only

Run:  uvicorn app:app --reload --port 8000   (or: python3 app.py)
"""
from __future__ import annotations
import math
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from urllib.parse import unquote

import fund_scorer as fs
from metrics import enrich_metrics, GLOSSARY
from data_source import load_funds, EXCEL_FILE, CSV_FILE
from holdings_source import (
    load_holdings, holdings_status, parse_upload, save_holdings_df, UPLOAD_META,
)
from overlap_engine import compute_overlap, company_detail, prepare_holdings, attach_aum
import nav_source as nav
import holdings_fetch as hfetch
import pe_enrich as peen

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"

app = FastAPI(title="Fundsconnect — Quantitative Fund Research API", version="1.0")

# ----------------------------------------------------------------------
# data loading  (single category for now; tag every fund so the engine
# treats them as one peer group. Replace with a real 'category' column
# when you load multiple fund types.)
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Re-read the live data source on every call so Excel edits show up
    on the next page refresh (no caching)."""
    df = load_funds()              # reads funds.xlsx if present, else ranked.csv
    df = enrich_metrics(df)        # add VaR / CVaR
    return df


def _data_status() -> dict:
    """Which file is live and when it was last modified on disk."""
    import datetime as _dt
    active = EXCEL_FILE if EXCEL_FILE.exists() else CSV_FILE
    mtime = active.stat().st_mtime if active.exists() else 0
    return {"source": active.name,
            "updated_at": _dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "updated_ts": mtime}


def _clean(records):
    """Replace NaN/inf with None so the JSON is valid."""
    out = []
    for r in records:
        out.append({k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                    for k, v in r.items()})
    return out


DISPLAY_COLS = [
    "name", "amc", "category", "aum", "ter", "cagr",
    "r1", "r2", "r3", "r5", "r7", "r10", "r15",
    "sharpe", "sortino", "infratio", "upcap", "downcap", "beta", "sd", "pe", "pb",
    "var", "var95_annual", "var99_annual", "cvar95_annual",
]


@app.get("/api/health")
def health():
    df = load_data()
    return {"status": "ok", "funds": int(len(df)),
            "categories": sorted(df["category"].unique().tolist()),
            **_data_status()}


@app.get("/api/data-status")
def data_status():
    """Lightweight endpoint the frontend polls to detect Excel changes."""
    return _data_status()


@app.get("/api/profiles")
def profiles():
    out = []
    for key in fs.ORDER:
        p = fs.RISK_PROFILES[key]
        out.append({"key": key, "name": p.name, "weights": p.weights,
                    "direction_overrides": p.direction_overrides})
    return {"profiles": out, "order": fs.ORDER,
            "dial": {"low": "balanced", "high": "aggressive",
                     "note": "t=0 balanced ... t=1 aggressive; beta flips to a positive at t>=0.5"}}


@app.get("/api/glossary")
def glossary():
    return {"metrics": GLOSSARY}


@app.get("/api/funds")
def funds(category: str | None = None):
    df = load_data()
    if category:
        df = df[df["category"] == category]
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return {"count": int(len(df)), "funds": _clean(df[cols].to_dict("records"))}


@app.get("/api/rank")
def rank(profile: str = Query("balanced"),
         category: str | None = None,
         min_history: str | None = None,
         n: int | None = None):
    if profile not in fs.RISK_PROFILES:
        raise HTTPException(400, f"unknown profile '{profile}'. valid: {fs.ORDER}")
    df = load_data()
    if category:
        df = df[df["category"] == category]
    if min_history:
        ranked = fs.top_n(df, profile=profile, n=(n or len(df)), min_history=min_history)
    else:
        ranked = fs.score_funds(df, profile=profile)
        if n:
            ranked = ranked.groupby("category", group_keys=False).head(n)
    cols = ["score"] + [c for c in DISPLAY_COLS if c in ranked.columns]
    recs = _clean(ranked[cols].to_dict("records"))
    return {"profile": profile, "profile_name": fs.RISK_PROFILES[profile].name,
            "count": len(recs), "funds": recs}


@app.get("/api/rank/dial")
def rank_dial(t: float = Query(0.5, ge=0.0, le=1.0),
              category: str | None = None,
              n: int | None = None):
    prof = fs.blend_profiles("balanced", "aggressive", t)
    df = load_data()
    if category:
        df = df[df["category"] == category]
    ranked = fs.score_funds(df, profile=prof)
    if n:
        ranked = ranked.groupby("category", group_keys=False).head(n)
    cols = ["score"] + [c for c in DISPLAY_COLS if c in ranked.columns]
    recs = _clean(ranked[cols].to_dict("records"))
    return {"t": t, "profile_name": prof.name, "weights": prof.weights,
            "count": len(recs), "funds": recs}


@app.get("/api/fund/{name}")
def fund(name: str):
    df = load_data()
    row = df[df["name"].str.lower() == name.lower()]
    if row.empty:
        row = df[df["name"].str.lower().str.contains(name.lower())]
    if row.empty:
        raise HTTPException(404, f"no fund matching '{name}'")
    rec = _clean(row.to_dict("records"))[0]
    scores = {}
    for key in fs.ORDER:
        s = fs.score_funds(df, profile=key)
        scores[key] = float(s[s["name"] == rec["name"]]["score"].iloc[0])
    return {"fund": rec, "scores_by_profile": scores}


# ----------------------------------------------------------------------
# holdings overlap
# ----------------------------------------------------------------------
def _holdings_payload():
    h = load_holdings()
    funds = load_data()
    return h, funds


@app.get("/api/overlap/status")
def overlap_status():
    st = holdings_status()
    if not st["present"]:
        return {**st, "rows": 0, "funds": 0, "companies": 0}
    h = load_holdings()
    if h.empty:
        return {**st, "rows": 0, "funds": 0, "companies": 0,
                "source": h.attrs.get("source") or st.get("source")}
    prep = prepare_holdings(h)
    return {
        **st,
        "source": h.attrs.get("source") or st.get("source"),
        "rows": int(len(h)),
        "funds": int(h["fund"].nunique()),
        "companies": int(prep["company_key"].nunique()) if len(prep) else 0,
    }


@app.get("/api/overlap/funds")
def overlap_funds():
    h, funds = _holdings_payload()
    if h.empty:
        return {"count": 0, "funds": [], **holdings_status()}
    prep = attach_aum(prepare_holdings(h), funds)
    rows = []
    for fund, grp in prep.groupby("fund"):
        aum = grp["aum"].dropna()
        rows.append({
            "fund": fund,
            "holdings": int(len(grp)),
            "aum": float(aum.iloc[0]) if len(aum) else None,
            "amc": (grp["amc"].dropna().iloc[0]
                    if grp["amc"].notna().any() else None),
            "category": (grp["category"].dropna().iloc[0]
                         if grp["category"].notna().any() else None),
            "aum_matched": bool(len(aum)),
        })
    rows.sort(key=lambda r: r["fund"].lower())
    return {"count": len(rows), "funds": rows, **holdings_status()}


@app.get("/api/overlap")
def overlap(funds: str | None = Query(
                None, description="Comma-separated fund names (empty = all)"),
            min_funds: int = Query(2, ge=1, le=50)):
    h, meta = _holdings_payload()
    st = holdings_status()
    if h.empty:
        return {
            "companies": [], "pairwise": [], "funds_used": [],
            "stats": {"rows": 0, "funds": 0, "companies": 0, "overlapping": 0},
            "min_funds": min_funds,
            **st,
        }
    selected = [x.strip() for x in funds.split(",")] if funds else None
    selected = [x for x in (selected or []) if x] or None
    result = compute_overlap(h, meta, selected_funds=selected, min_funds=min_funds)
    return {**result, "min_funds": min_funds, **st,
            "source": h.attrs.get("source") or st.get("source")}


@app.get("/api/overlap/company/{name:path}")
def overlap_company(name: str):
    name = unquote(name).strip()
    if not name:
        raise HTTPException(400, "company name required")
    h, meta = _holdings_payload()
    if h.empty:
        raise HTTPException(404, "no holdings data — drop holdings.xlsx next to the app")
    detail = company_detail(h, meta, name)
    if detail is None:
        raise HTTPException(404, f"no company matching '{name}'")
    return {"company": detail}


@app.post("/api/data/upload")
async def data_upload(
    file: UploadFile = File(...),
    default_fund: str | None = Form(None),
    mode: str = Form("replace"),  # replace | preview
):
    """
    Upload Excel / CSV / JSON holdings.
    mode=preview → parse only; mode=replace → parse + save as holdings.csv.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "file too large (max 25 MB)")

    parsed = parse_upload(
        filename=file.filename or "upload.csv",
        data=raw,
        default_fund=(default_fund or "").strip() or None,
    )
    payload = {
        "ok": parsed["ok"],
        "errors": parsed.get("errors") or [],
        "warnings": parsed.get("warnings") or [],
        "rows": parsed.get("rows") or 0,
        "funds": parsed.get("funds") or 0,
        "fund_names": parsed.get("fund_names") or [],
        "mapping": parsed.get("mapping") or {},
        "sample": parsed.get("sample") or [],
        "filename": file.filename,
        "saved": False,
    }
    if not parsed["ok"]:
        return JSONResponse(payload, status_code=400)

    if mode == "preview":
        return payload

    try:
        meta = save_holdings_df(parsed["df"], original_filename=file.filename)
    except Exception as e:
        raise HTTPException(500, f"save failed: {e}") from e

    payload["saved"] = True
    payload["meta"] = meta
    payload["status"] = holdings_status()
    return payload


@app.get("/api/data/status")
def data_status_holdings():
    """Holdings store status for the Data tab."""
    import json as _json
    st = holdings_status()
    h = load_holdings()
    meta = None
    if UPLOAD_META.exists():
        try:
            meta = _json.loads(UPLOAD_META.read_text())
        except Exception:
            meta = None
    return {
        **st,
        "rows": int(len(h)) if h is not None else 0,
        "funds": int(h["fund"].nunique()) if h is not None and len(h) else 0,
        "upload_meta": meta,
        "fetch": hfetch.get_progress(),
        "fetch_meta": hfetch.fetch_meta(),
        "enrich": peen.get_progress(),
        "enrich_meta": peen.enrich_meta(),
    }


@app.post("/api/data/fetch-holdings")
def data_fetch_holdings():
    """
    Auto-download portfolio holdings + sectors for every fund in the
    rankings sheet (FinAPI search). Runs in background; poll /api/data/status.
    """
    df = load_data()
    names = df["name"].astype(str).tolist() if "name" in df.columns else []
    if not names:
        raise HTTPException(400, "no funds in rankings data")
    prog = hfetch.get_progress()
    if prog.get("running"):
        return {"started": False, "already_running": True, **prog}
    prog = hfetch.start_fetch_async(names)
    return {"started": True, "already_running": False, **prog}


@app.get("/api/data/fetch-holdings/status")
def data_fetch_holdings_status():
    return {**hfetch.get_progress(), "fetch_meta": hfetch.fetch_meta()}


@app.post("/api/data/enrich-pe")
def data_enrich_pe():
    """
    Fill stock-level P/E and P/B on holdings (Yahoo Finance).
    Needed for green P/E colours / Value Picks — FinAPI holdings lack these.
    """
    prog = peen.get_progress()
    if prog.get("running"):
        return {"started": False, "already_running": True, **prog}
    if hfetch.get_progress().get("running"):
        raise HTTPException(409, "holdings fetch still running — wait, then enrich")
    prog = peen.start_enrich_async()
    return {"started": True, "already_running": False, **prog}


@app.get("/api/data/enrich-pe/status")
def data_enrich_pe_status():
    return {**peen.get_progress(), "enrich_meta": peen.enrich_meta()}


# ----------------------------------------------------------------------
# Live NAV (mfapi.in / AMFI)
# ----------------------------------------------------------------------
@app.get("/api/nav/status")
def nav_status():
    return {**nav.get_progress(), "note": nav.nav_note()}


@app.get("/api/nav/live")
def nav_live(
    refresh: bool = Query(True, description="Kick off a background refresh if idle"),
    force_remap: bool = Query(False, description="Rebuild AMFI scheme matches"),
):
    """
    Returns cached NAVs + progress. On first call (or refresh=true when idle),
    starts a parallel fetch for every fund in the metrics sheet.
    """
    df = load_data()
    names = df["name"].astype(str).tolist() if "name" in df.columns else []
    prog = nav.get_progress()
    # Page load passes refresh=true once → fire parallel fetches (FundLens-style).
    # Polls use refresh=false so we don't restart mid-run.
    if refresh and names and not prog.get("running"):
        prog = nav.start_refresh_async(names, force_remap=force_remap)

    cache = nav.get_nav_cache()
    # slim payload for the table
    by_fund = {}
    for name, row in cache.items():
        by_fund[name] = {
            "nav": row.get("nav"),
            "date": row.get("date"),
            "amfi": row.get("amfi"),
            "matched_scheme": row.get("matched_scheme"),
        }
    return {
        "progress": prog,
        "navs": by_fund,
        "count": len(by_fund),
        "funds_total": len(names),
        "note": nav.nav_note(),
    }


# ----------------------------------------------------------------------
# static frontend
# ----------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
