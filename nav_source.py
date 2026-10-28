"""
nav_source.py  --  Live NAV from mfapi.in (AMFI)
================================================
On demand, matches every Fundsconnect fund name to an AMFI scheme code,
then fetches latest NAV in parallel. Progress is exposed for the UI
counter ("Live NAV: 18/120").

Notes
-----
* AMFI publishes NAV once per trading day (~9–9:30pm IST). Weekends/holidays
  do not update. Holdings are NOT available from mfapi.in — only NAV.
* Scheme list + AMFI map + NAV results are cached on disk so refresh is fast.
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
import ssl
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE = Path(__file__).resolve().parent
SCHEME_CACHE = BASE / ".mfapi_schemes.json"
AMFI_MAP = BASE / ".amfi_map.json"
NAV_CACHE = BASE / ".nav_cache.json"

MFAPI = "https://api.mfapi.in/mf"
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
}


def _ssl_context():
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        try:
            ctx.check_hostname = True
            return ctx
        except Exception:
            return ssl._create_unverified_context()


def _http_get_json(url: str, timeout: float = 12.0) -> Any:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError:
        # macOS Python installs often lack system CA bundle
        with urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))


# Expand common short forms used in Excel vs AMFI long names
_ALIASES = [
    (r"\bADITYA BIRLA SL\b", "ADITYA BIRLA SUN LIFE"),
    (r"\bABSL\b", "ADITYA BIRLA SUN LIFE"),
    (r"\bICICI PRU\b", "ICICI PRUDENTIAL"),
    (r"\bNIPPON INDIA GROWTH\b", "NIPPON INDIA GROWTH"),
    (r"\bSMALLCAP\b", "SMALL CAP"),
    (r"\bFLEXICAP\b", "FLEXI CAP"),
    (r"\bMIDCAP\b", "MID CAP"),
    (r"\bLARGECAP\b", "LARGE CAP"),
    (r"\bMULTICAP\b", "MULTI CAP"),
]


def _norm(s: str) -> str:
    s = str(s or "").upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    for pat, rep in _ALIASES:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\b(FUND|SCHEME|PLAN|OPTION|THE)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_growth(name: str) -> bool:
    n = name.upper()
    if "IDCW" in n or "DIVIDEND" in n:
        return False
    return "GROWTH" in n or "GROWTH OPTION" in n or n.endswith(" G")


def _is_direct(name: str) -> bool:
    return "DIRECT" in name.upper()


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_scheme_index(force: bool = False) -> list[dict]:
    """Full mfapi scheme list (~37k). Cached 7 days on disk."""
    if not force and SCHEME_CACHE.exists():
        age = time.time() - SCHEME_CACHE.stat().st_mtime
        if age < 7 * 86400:
            data = _load_json(SCHEME_CACHE, None)
            if isinstance(data, list) and data:
                return data
    schemes = _http_get_json(MFAPI, timeout=60)
    if not isinstance(schemes, list):
        raise RuntimeError("mfapi.in scheme list unexpected shape")
    _save_json(SCHEME_CACHE, schemes)
    return schemes


def match_amfi_code(fund_name: str, schemes: list[dict] | None = None) -> dict | None:
    """
    Best-effort map Fundsconnect name → AMFI scheme.
    Prefers Regular Growth (then Direct Growth) equity-style names.
    """
    schemes = schemes if schemes is not None else load_scheme_index()
    target = _norm(fund_name)
    if not target:
        return None

    # Tokens without generic words for scoring
    t_tokens = set(target.split())

    candidates = []
    for s in schemes:
        sn = s.get("schemeName") or ""
        nn = _norm(sn)
        if not nn:
            continue
        # Must contain most of the fund name tokens
        if target in nn or nn.startswith(target) or target[:24] in nn:
            score = 0
            if target == nn:
                score += 100
            elif target in nn:
                score += 60
            # token overlap
            s_tokens = set(nn.split())
            overlap = len(t_tokens & s_tokens) / max(1, len(t_tokens))
            score += int(overlap * 40)
            if _is_growth(sn):
                score += 25
            else:
                score -= 15
            # Prefer Regular retail Growth (FundLens style) over Direct / Institutional
            if _is_direct(sn):
                score -= 5
            else:
                score += 8
            if "INSTITUTIONAL" in sn.upper():
                score -= 20
            if "REGULAR" in sn.upper():
                score += 12
            # Prefer open-ended sounding (no "FOF" penalty light)
            if "FOF" in sn.upper() or "FUND OF FUND" in sn.upper():
                score -= 20
            candidates.append((score, int(s["schemeCode"]), sn))

    if not candidates:
        # looser: require AMC-ish prefix match of first 3 tokens
        parts = target.split()
        if len(parts) >= 2:
            prefix = " ".join(parts[:3])
            for s in schemes:
                sn = s.get("schemeName") or ""
                nn = _norm(sn)
                if prefix in nn and _is_growth(sn):
                    score = 30 + (10 if not _is_direct(sn) else 0)
                    candidates.append((score, int(s["schemeCode"]), sn))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    if best[0] < 35:
        return None
    return {"amfi": best[1], "scheme_name": best[2], "score": best[0]}


def build_amfi_map(fund_names: list[str], force: bool = False) -> dict[str, dict]:
    """Map each fund display name → {amfi, scheme_name}."""
    cached = _load_json(AMFI_MAP, {})
    if not force:
        missing = [n for n in fund_names if n not in cached]
        if not missing:
            return {n: cached[n] for n in fund_names if n in cached}

    schemes = load_scheme_index(force=force)
    out = dict(cached)
    for name in fund_names:
        if not force and name in out and out[name].get("amfi"):
            continue
        m = match_amfi_code(name, schemes)
        if m:
            out[name] = {"amfi": m["amfi"], "scheme_name": m["scheme_name"]}
        else:
            out[name] = {"amfi": None, "scheme_name": None, "error": "no match"}
    _save_json(AMFI_MAP, out)
    return {n: out[n] for n in fund_names if n in out}


def fetch_one_nav(amfi: int) -> dict | None:
    try:
        j = _http_get_json(f"{MFAPI}/{amfi}/latest", timeout=8)
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    if not isinstance(j, dict) or j.get("status") != "SUCCESS":
        return None
    data = j.get("data") or []
    if not data:
        return None
    row = data[0]
    try:
        nav = float(row.get("nav"))
    except (TypeError, ValueError):
        return None
    meta = j.get("meta") or {}
    return {
        "amfi": amfi,
        "nav": nav,
        "date": row.get("date"),
        "scheme_name": meta.get("scheme_name"),
        "fund_house": meta.get("fund_house"),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def get_progress() -> dict:
    with _lock:
        return dict(_state)


def get_nav_cache() -> dict[str, dict]:
    return _load_json(NAV_CACHE, {})


def refresh_navs(fund_names: list[str], force_remap: bool = False) -> dict:
    """
    Resolve AMFI codes and fetch latest NAV for all fund_names in parallel.
    Updates progress state for the UI. Safe to call from a background thread.
    """
    with _lock:
        if _state["running"]:
            return get_progress()
        _state.update({
            "running": True, "done": 0, "total": len(fund_names),
            "ok": 0, "failed": 0, "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None, "error": None,
        })

    try:
        amfi_map = build_amfi_map(fund_names, force=force_remap)
    except Exception as e:
        with _lock:
            _state["running"] = False
            _state["error"] = f"scheme list failed: {e}"
            _state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return get_progress()

    navs = get_nav_cache()
    # Keep previous values until replaced

    def _work(name: str):
        info = amfi_map.get(name) or {}
        amfi = info.get("amfi")
        if not amfi:
            return name, None, "no_amfi"
        nav = fetch_one_nav(int(amfi))
        if not nav:
            return name, None, "fetch_failed"
        nav["fund"] = name
        nav["matched_scheme"] = info.get("scheme_name")
        return name, nav, None

    workers = int(os.environ.get("NAV_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as ex:
        futs = [ex.submit(_work, n) for n in fund_names]
        for fut in as_completed(futs):
            name, nav, err = fut.result()
            with _lock:
                _state["done"] += 1
                if nav:
                    navs[name] = nav
                    _state["ok"] += 1
                else:
                    _state["failed"] += 1
            # Persist periodically
            if _state["done"] % 8 == 0:
                _save_json(NAV_CACHE, navs)

    _save_json(NAV_CACHE, navs)
    with _lock:
        _state["running"] = False
        _state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return get_progress()


def start_refresh_async(fund_names: list[str], force_remap: bool = False) -> dict:
    """Kick off refresh in a daemon thread; return current progress immediately."""
    with _lock:
        if _state["running"]:
            return get_progress()
    t = threading.Thread(
        target=refresh_navs, args=(list(fund_names), force_remap), daemon=True
    )
    t.start()
    # tiny wait so running=True is visible
    time.sleep(0.05)
    return get_progress()


def nav_note() -> str:
    return (
        "NAV from mfapi.in (AMFI). Published once per trading day, usually by "
        "9–9:30pm IST. Morning refresh shows yesterday’s NAV. "
        "Holdings are not in this feed — use the Data tab for portfolios."
    )
