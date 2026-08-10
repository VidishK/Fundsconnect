"""
holdings_fetch.py  --  Auto-download portfolio holdings for ranked funds
=======================================================================
Uses FinAPI search (free, no key) which returns equity holdings + sectors
for each scheme. Matches Fundsconnect fund names → Growth / Regular plan,
then writes a tidy holdings table (fund, company, weight, sector, …).

Not AMFI official — third-party aggregator. Refresh monthly after AMC disclosures.
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
import ssl
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import pandas as pd

BASE = Path(__file__).resolve().parent
MAP_CACHE = BASE / ".finapi_fund_map.json"
FETCH_META = BASE / ".holdings_fetch_meta.json"

FINAPI = "https://finapi.upvaly.com/api/mf"
UA = "Fundsconnect/1.0 (research; +local)"

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


def _ssl_context():
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _http_get_json(url: str, timeout: float = 25.0) -> Any:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError:
        with urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))


def get_progress() -> dict:
    with _lock:
        return dict(_state)


_EXPAND = [
    (r"\bADITYA BIRLA SL\b", "Aditya Birla Sun Life"),
    (r"\bABSL\b", "Aditya Birla Sun Life"),
    (r"\bICICI PRU\b", "ICICI Prudential"),
    (r"\bNIPPON INDIA GROWTH\b", "Nippon India Growth"),
    (r"UTI-", "UTI "),
    (r"\bSMALLCAP\b", "Small Cap"),
    (r"\bFLEXICAP\b", "Flexi Cap"),
    (r"\bMIDCAP\b", "Mid Cap"),
    (r"\bLARGECAP\b", "Large Cap"),
]


def _expand_display(name: str) -> str:
    s = str(name or "")
    for pat, repl in _EXPAND:
        s = re.sub(pat, repl, s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _norm(s: str) -> str:
    s = _expand_display(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\bFUND\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _query_variants(fund_name: str) -> list[str]:
    """FinAPI often returns 0 hits when the query ends with 'Fund' — try variants."""
    base = _expand_display(fund_name)
    bare = re.sub(r"\s+Fund$", "", base, flags=re.I).strip()
    out = []
    for q in (bare, base, re.sub(r"\s+Fund$", "", fund_name, flags=re.I).strip(), fund_name):
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in out:
            out.append(q)
    return out


def _plan_score(scheme_name: str) -> int:
    n = scheme_name.lower()
    score = 0
    if "growth" in n:
        score += 80
    if "direct" in n:
        score -= 25  # holdings identical; slightly prefer regular naming
    if "idcw" in n or "dividend" in n:
        score -= 60
    if "regular" in n:
        score += 15
    if "institutional" in n:
        score -= 10
    return score


def _name_score(target: str, scheme_name: str) -> int:
    t = _norm(target)
    s = _norm(scheme_name)
    # strip plan suffixes for comparison
    s = re.sub(
        r"\b(GROWTH|DIRECT|REGULAR|IDCW|DIVIDEND|OPTION|PLAN)\b",
        " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()
    if t == s:
        return 100
    if t in s or s in t:
        return 80
    tw, sw = set(t.split()), set(s.split())
    if not tw:
        return 0
    overlap = len(tw & sw) / len(tw)
    return int(overlap * 70)


def _pick_scheme(fund_name: str, results: list[dict]) -> dict | None:
    best, best_sc = None, -10**9
    for row in results or []:
        sn = row.get("schemeName") or ""
        sc = _name_score(fund_name, sn) + _plan_score(sn)
        if sc > best_sc:
            best, best_sc = row, sc
    if best is None or best_sc < 55:
        return None
    return best


def _parse_weight(x) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "NA", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _search_raw(q: str, retries: int = 3) -> list[dict]:
    url = f"{FINAPI}/search?schemeName={quote(q)}"
    last_err = None
    for attempt in range(retries):
        try:
            payload = _http_get_json(url)
            results = payload.get("data") if isinstance(payload, dict) else None
            return results if isinstance(results, list) else []
        except HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))
    if last_err:
        raise last_err
    return []


def search_scheme(fund_name: str) -> dict | None:
    """Search FinAPI and return best matching scheme payload (incl. holdings)."""
    best = None
    best_sc = -10**9
    for q in _query_variants(fund_name):
        results = _search_raw(q)
        if not results:
            continue
        pick = _pick_scheme(fund_name, results)
        if not pick:
            continue
        sc = _name_score(fund_name, pick.get("schemeName") or "") + _plan_score(
            pick.get("schemeName") or ""
        )
        if sc > best_sc:
            best, best_sc = pick, sc
        if best_sc >= 120:
            break
        time.sleep(0.15)
    return best


def holdings_rows_from_scheme(fund_name: str, scheme: dict) -> list[dict]:
    rows = []
    for h in scheme.get("holdings") or []:
        if not isinstance(h, dict):
            continue
        company = (h.get("name") or h.get("stock_name") or "").strip()
        wt = _parse_weight(h.get("weightage") or h.get("weight") or h.get("weight_pct"))
        if not company or wt is None or wt <= 0:
            continue
        sector = (h.get("sector") or "").strip() or None
        rows.append({
            "fund": fund_name,
            "company": company,
            "weight": wt,
            "isin": None,
            "sector": sector,
            "pe": None,
            "pb": None,
            "scheme_code": scheme.get("schemeCode"),
            "matched_scheme": scheme.get("schemeName"),
        })
    return rows


def fetch_one(fund_name: str) -> tuple[str, list[dict], str | None]:
    try:
        scheme = search_scheme(fund_name)
        if not scheme:
            return fund_name, [], "no scheme match"
        rows = holdings_rows_from_scheme(fund_name, scheme)
        if not rows:
            return fund_name, [], "matched but no holdings in response"
        return fund_name, rows, None
    except Exception as e:
        return fund_name, [], str(e)


def start_fetch_async(fund_names: list[str], workers: int = 2) -> dict:
    """Kick off background fetch. Returns progress snapshot."""
    with _lock:
        if _state["running"]:
            return dict(_state)
        _state.update({
            "running": True,
            "done": 0,
            "total": len(fund_names),
            "ok": 0,
            "failed": 0,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "error": None,
            "message": "Fetching holdings…",
        })

    def runner():
        all_rows: list[dict] = []
        fails: list[dict] = []
        try:
            # Keep concurrency low — FinAPI 429s under burst load
            with ThreadPoolExecutor(max_workers=max(1, min(workers, 3))) as ex:
                futs = {ex.submit(fetch_one, n): n for n in fund_names}
                for fut in as_completed(futs):
                    name, rows, err = fut.result()
                    with _lock:
                        _state["done"] += 1
                        if err or not rows:
                            _state["failed"] += 1
                            fails.append({"fund": name, "error": err or "empty"})
                        else:
                            _state["ok"] += 1
                            all_rows.extend(rows)
                        _state["message"] = (
                            f"Fetched {_state['ok']} funds · "
                            f"{_state['done']}/{_state['total']}"
                        )
                    time.sleep(0.12)

            if not all_rows:
                raise RuntimeError("No holdings returned for any fund")

            df = pd.DataFrame(all_rows)
            # keep tidy cols for save
            keep = ["fund", "company", "weight", "isin", "sector", "pe", "pb"]
            for c in keep:
                if c not in df.columns:
                    df[c] = None
            df = df[keep]

            from holdings_source import save_holdings_df
            meta = save_holdings_df(df, original_filename="finapi-auto-fetch")
            FETCH_META.write_text(json.dumps({
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "funds_ok": int(df["fund"].nunique()),
                "rows": int(len(df)),
                "failed": fails[:40],
                "failed_count": len(fails),
                "source": "finapi.upvaly.com search (holdings+sectors)",
                "save_meta": meta,
            }, indent=2))
            with _lock:
                _state["message"] = (
                    f"Saved {int(df['fund'].nunique())} funds · "
                    f"{len(df)} holdings rows"
                    + (f" · {len(fails)} unmatched" if fails else "")
                )
        except Exception as e:
            with _lock:
                _state["error"] = str(e)
                _state["message"] = f"Fetch failed: {e}"
        finally:
            with _lock:
                _state["running"] = False
                _state["finished_at"] = datetime.now().isoformat(timespec="seconds")

    threading.Thread(target=runner, daemon=True).start()
    return get_progress()


def fetch_meta() -> dict | None:
    if not FETCH_META.exists():
        return None
    try:
        return json.loads(FETCH_META.read_text())
    except Exception:
        return None
