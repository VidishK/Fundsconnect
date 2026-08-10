"""
pe_enrich.py  --  Fill stock-level P/E and P/B on holdings.csv
================================================================
Holdings fetch (FinAPI) returns name / weight / sector only.
FundLens-style green P/Es need per-stock valuation — we resolve each
company to an NSE/BSE Yahoo ticker and pull trailing P/E + P/B.

Runs in the background; progress via get_progress().
"""
from __future__ import annotations
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE = BASE / ".stock_pe_cache.json"
META = BASE / ".pe_enrich_meta.json"

_SKIP_RE = re.compile(
    r"(triparty|treps|cblo|repo|cash|net current|receivable|payable|"
    r"\bbill\b|\bncd\b|\bbond\b|fixed deposit|\bfd\b|margin|mutual fund|"
    r"etf|index fund|gilt|treasury|commercial paper|certificate of deposit)",
    re.I,
)

_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "done": 0,
    "total": 0,
    "ok": 0,
    "failed": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "message": None,
}


def get_progress() -> dict:
    with _lock:
        return dict(_state)


def enrich_meta() -> dict | None:
    if not META.exists():
        return None
    try:
        return json.loads(META.read_text())
    except Exception:
        return None


def _load_cache() -> dict:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=2))


def _is_skip(name: str) -> bool:
    return not name or bool(_SKIP_RE.search(name))


def _resolve_ticker(company: str) -> str | None:
    import yfinance as yf
    try:
        s = yf.Search(company, max_results=8)
        quotes = s.quotes or []
    except Exception:
        return None
    equity = [q for q in quotes if (q.get("quoteType") or "") == "EQUITY"]
    for suf in (".NS", ".BO"):
        for q in equity:
            sym = q.get("symbol") or ""
            if sym.endswith(suf):
                return sym
    return None


def _fetch_ratios(ticker: str) -> tuple[float | None, float | None]:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None, None
    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    try:
        pe = float(pe) if pe is not None else None
    except (TypeError, ValueError):
        pe = None
    try:
        pb = float(pb) if pb is not None else None
    except (TypeError, ValueError):
        pb = None
    if pe is not None and (pe <= 0 or pe > 500):
        pe = None
    if pb is not None and (pb <= 0 or pb > 100):
        pb = None
    return pe, pb


def _lookup_one(company: str, cache: dict) -> dict:
    if company in cache and cache[company].get("ticker"):
        row = cache[company]
        if row.get("pe") is not None or row.get("pb") is not None:
            return row
    ticker = (cache.get(company) or {}).get("ticker") or _resolve_ticker(company)
    if not ticker:
        return {"ticker": None, "pe": None, "pb": None, "error": "no ticker"}
    pe, pb = _fetch_ratios(ticker)
    return {"ticker": ticker, "pe": pe, "pb": pb, "error": None if (pe or pb) else "no ratios"}


def start_enrich_async(workers: int = 6) -> dict:
    with _lock:
        if _state["running"]:
            return dict(_state)

    from holdings_source import load_holdings, save_holdings_df

    h = load_holdings()
    if h is None or h.empty:
        with _lock:
            _state.update({
                "running": False,
                "error": "no holdings loaded",
                "message": "Load / fetch holdings first",
            })
        return get_progress()

    companies = sorted({
        str(c).strip() for c in h["company"].dropna().unique()
        if not _is_skip(str(c))
    })

    with _lock:
        _state.update({
            "running": True,
            "done": 0,
            "total": len(companies),
            "ok": 0,
            "failed": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "error": None,
            "message": "Resolving stock P/E & P/B…",
        })

    def runner():
        cache = _load_cache()
        results: dict[str, dict] = {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as ex:
                futs = {ex.submit(_lookup_one, c, cache): c for c in companies}
                for fut in as_completed(futs):
                    c = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as e:
                        row = {"ticker": None, "pe": None, "pb": None, "error": str(e)}
                    results[c] = row
                    cache[c] = {**row, "updated_at": datetime.now().isoformat(timespec="seconds")}
                    with _lock:
                        _state["done"] += 1
                        if row.get("pe") is not None or row.get("pb") is not None:
                            _state["ok"] += 1
                        else:
                            _state["failed"] += 1
                        _state["message"] = (
                            f"P/E filled {_state['ok']} · "
                            f"{_state['done']}/{_state['total']} companies"
                        )
                    if _state["done"] % 25 == 0:
                        _save_cache(cache)

            _save_cache(cache)

            df = h.copy()
            if "pe" not in df.columns:
                df["pe"] = None
            if "pb" not in df.columns:
                df["pb"] = None
            pe_map = {c: r.get("pe") for c, r in results.items()}
            pb_map = {c: r.get("pb") for c, r in results.items()}
            df["pe"] = df["company"].map(lambda x: pe_map.get(str(x).strip(), None))
            df["pb"] = df["company"].map(lambda x: pb_map.get(str(x).strip(), None))
            # keep any prior pe if lookup failed
            if "pe" in h.columns:
                old = h["pe"]
                df["pe"] = df["pe"].where(df["pe"].notna(), old)
            if "pb" in h.columns:
                old = h["pb"]
                df["pb"] = df["pb"].where(df["pb"].notna(), old)

            meta = save_holdings_df(df, original_filename="pe-enrich")
            filled = int(df["pe"].notna().sum())
            META.write_text(json.dumps({
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "companies_tried": len(companies),
                "companies_ok": int(_state["ok"]),
                "rows_with_pe": filled,
                "source": "Yahoo Finance (yfinance) trailing P/E + P/B",
                "save_meta": meta,
            }, indent=2))
            with _lock:
                _state["message"] = (
                    f"Done — {_state['ok']} stocks with P/E/P/B · "
                    f"{filled} holdings rows updated"
                )
        except Exception as e:
            with _lock:
                _state["error"] = str(e)
                _state["message"] = f"Enrich failed: {e}"
        finally:
            with _lock:
                _state["running"] = False
                _state["finished_at"] = datetime.now().isoformat(timespec="seconds")

    threading.Thread(target=runner, daemon=True).start()
    return get_progress()
