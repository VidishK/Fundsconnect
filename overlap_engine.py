"""
overlap_engine.py  --  Cross-fund holdings overlap + money concentration
================================================================
Input: tidy holdings (fund, company, weight, isin?, sector?) + optional
fund metrics table with AUM for money estimates.

Output:
  * ranked company overlap rows (fund_count, weights, est_money_cr)
  * pairwise fund overlap matrix (shared min-weight %)
"""
from __future__ import annotations
import re
from typing import Iterable
import pandas as pd
import numpy as np

# Corporate suffix noise stripped for fuzzy company matching
_SUFFIX_RE = re.compile(
    r"\b(limited|ltd\.?|private|pvt\.?|inc\.?|incorporated|corp\.?|"
    r"corporation|company|co\.?|plc|llc|llp|the)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^A-Z0-9& ]+")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_company(name: str) -> str:
    """Uppercase, strip Ltd/Limited/etc, collapse whitespace."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().strip()
    s = _SUFFIX_RE.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def normalize_fund(name: str) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().strip()
    s = _MULTI_SPACE.sub(" ", s)
    return s


def company_key(row: pd.Series) -> str:
    """Prefer ISIN as stable key; else normalized company name."""
    isin = row.get("isin")
    if isin is not None and not (isinstance(isin, float) and np.isnan(isin)):
        s = str(isin).strip().upper()
        if s and s not in ("NAN", "NONE", "NA", "N/A"):
            return "ISIN:" + s
    return "NAME:" + normalize_company(row.get("company", ""))


def prepare_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Add normalized keys. Drops rows with empty company key."""
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=[
            "fund", "company", "weight", "isin", "sector",
            "fund_key", "company_key", "company_norm",
        ])
    df = holdings.copy()
    df["fund_key"] = df["fund"].map(normalize_fund)
    df["company_norm"] = df["company"].map(normalize_company)
    df["company_key"] = df.apply(company_key, axis=1)
    df = df[df["company_key"].str.len() > 5]  # "NAME:" + at least 1 char
    df = df[df["fund_key"] != ""]
    return df.reset_index(drop=True)


def attach_aum(holdings: pd.DataFrame, funds: pd.DataFrame | None) -> pd.DataFrame:
    """Join fund AUM / AMC / category onto holdings via normalized fund name."""
    df = holdings.copy()
    df["aum"] = np.nan
    df["amc"] = None
    df["category"] = None
    if funds is None or funds.empty or "name" not in funds.columns:
        return df

    meta = funds.copy()
    meta["fund_key"] = meta["name"].map(normalize_fund)
    cols = ["fund_key"]
    for c in ("aum", "amc", "category", "name"):
        if c in meta.columns:
            cols.append(c)
    meta = meta[cols].drop_duplicates(subset=["fund_key"], keep="first")
    meta = meta.rename(columns={"name": "fund_matched", "aum": "aum_meta",
                                "amc": "amc_meta", "category": "category_meta"})

    # Exact key match first
    merged = df.merge(meta, on="fund_key", how="left")

    # Fuzzy fallback: holdings fund_key contained in metrics name or vice versa
    miss = merged["aum_meta"].isna()
    if miss.any():
        meta_keys = list(zip(meta["fund_key"], meta["aum_meta"],
                             meta.get("amc_meta", pd.Series([None]*len(meta))),
                             meta.get("category_meta", pd.Series([None]*len(meta))),
                             meta.get("fund_matched", pd.Series([None]*len(meta)))))
        for idx in merged.index[miss]:
            hk = merged.at[idx, "fund_key"]
            best = None
            for mk, aum, amc, cat, fname in meta_keys:
                if not mk:
                    continue
                if hk in mk or mk in hk:
                    best = (aum, amc, cat, fname)
                    break
            if best:
                merged.at[idx, "aum_meta"] = best[0]
                merged.at[idx, "amc_meta"] = best[1]
                merged.at[idx, "category_meta"] = best[2]
                merged.at[idx, "fund_matched"] = best[3]

    merged["aum"] = merged["aum_meta"]
    merged["amc"] = merged["amc_meta"]
    merged["category"] = merged["category_meta"]
    drop = [c for c in merged.columns if c.endswith("_meta") or c == "fund_matched"]
    return merged.drop(columns=drop, errors="ignore")


def _filter_funds(df: pd.DataFrame, funds: Iterable[str] | None) -> pd.DataFrame:
    if not funds:
        return df
    wanted = {normalize_fund(f) for f in funds if f}
    if not wanted:
        return df
    # Match exact or substring either way
    mask = df["fund_key"].isin(wanted)
    if not mask.any():
        mask = df["fund_key"].apply(
            lambda k: any(k == w or w in k or k in w for w in wanted)
        )
    # Also allow raw fund name match
    raw = {str(f).strip().lower() for f in funds if f}
    mask = mask | df["fund"].str.strip().str.lower().isin(raw)
    return df[mask].copy()


def compute_overlap(
    holdings: pd.DataFrame,
    funds_meta: pd.DataFrame | None = None,
    selected_funds: list[str] | None = None,
    min_funds: int = 2,
) -> dict:
    """
    Rank companies by how many selected funds hold them, then by estimated money.

    Returns dict with keys: companies, pairwise, funds_used, stats
    """
    prepared = prepare_holdings(holdings)
    prepared = attach_aum(prepared, funds_meta)
    prepared = _filter_funds(prepared, selected_funds)

    if prepared.empty:
        return {
            "companies": [],
            "all_companies": [],
            "pairwise": [],
            "funds_used": [],
            "stats": {"rows": 0, "funds": 0, "companies": 0, "overlapping": 0},
        }

    prepared["est_money_cr"] = prepared["weight"] / 100.0 * prepared["aum"]

    # Display company name: most common original spelling for the key
    name_mode = (
        prepared.groupby("company_key")["company"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("company_display")
    )
    sector_mode = None
    if "sector" in prepared.columns:
        sector_mode = (
            prepared.dropna(subset=["sector"])
            .groupby("company_key")["sector"]
            .agg(lambda s: s.value_counts().index[0] if len(s) else None)
            .rename("sector")
        )

    # One row per (company_key, fund) — max weight if duplicates
    per = (prepared.sort_values("weight", ascending=False)
                   .drop_duplicates(subset=["company_key", "fund"], keep="first"))

    companies = []
    for key, grp in per.groupby("company_key"):
        fund_rows = []
        for _, r in grp.sort_values("weight", ascending=False).iterrows():
            fund_rows.append({
                "fund": r["fund"],
                "weight": float(r["weight"]) if pd.notna(r["weight"]) else None,
                "aum": float(r["aum"]) if pd.notna(r["aum"]) else None,
                "est_money_cr": float(r["est_money_cr"]) if pd.notna(r["est_money_cr"]) else None,
                "amc": r["amc"] if pd.notna(r.get("amc")) else None,
                "category": r["category"] if pd.notna(r.get("category")) else None,
            })
        weights = [f["weight"] for f in fund_rows if f["weight"] is not None]
        moneys = [f["est_money_cr"] for f in fund_rows if f["est_money_cr"] is not None]
        display = name_mode.get(key, key.replace("NAME:", "").replace("ISIN:", ""))
        sector = None
        if sector_mode is not None and key in sector_mode.index:
            sector = sector_mode.get(key)
            if pd.isna(sector):
                sector = None
        pe = pb = None
        if "pe" in grp.columns and grp["pe"].notna().any():
            pe = float(grp["pe"].dropna().iloc[0])
        if "pb" in grp.columns and grp["pb"].notna().any():
            pb = float(grp["pb"].dropna().iloc[0])
        isin = key[5:] if key.startswith("ISIN:") else None
        companies.append({
            "company": display,
            "company_key": key,
            "isin": isin,
            "sector": sector,
            "pe": pe,
            "pb": pb,
            "fund_count": len(fund_rows),
            "sum_weight": float(sum(weights)) if weights else 0.0,
            "avg_weight": float(sum(weights) / len(weights)) if weights else 0.0,
            "est_money_cr": float(sum(moneys)) if moneys else None,
            "funds": fund_rows,
        })

    # Default: most commonly held across selected funds, then weight, then ₹
    companies.sort(key=lambda c: (-c["fund_count"],
                                  -c["avg_weight"],
                                  -(c["est_money_cr"] or 0),
                                  -c["sum_weight"]))
    all_companies = [c for c in companies if c["fund_count"] >= 1]
    overlapping = [c for c in companies if c["fund_count"] >= max(1, int(min_funds))]

    pairwise = _pairwise_overlap(per)
    funds_used = sorted(per["fund"].unique().tolist())

    return {
        "companies": overlapping,
        "all_companies": all_companies,
        "pairwise": pairwise,
        "funds_used": funds_used,
        "stats": {
            "rows": int(len(prepared)),
            "funds": len(funds_used),
            "companies": int(per["company_key"].nunique()),
            "overlapping": len(overlapping),
        },
    }


def _pairwise_overlap(per_fund_company: pd.DataFrame) -> list[dict]:
    """
    For each fund pair, compute:
      shared_count  — companies in both
      overlap_pct   — sum(min(w_a, w_b)) across shared names (portfolio overlap %)
    """
    funds = sorted(per_fund_company["fund"].unique().tolist())
    by_fund: dict[str, dict[str, float]] = {}
    for fund, grp in per_fund_company.groupby("fund"):
        by_fund[fund] = dict(zip(grp["company_key"], grp["weight"].astype(float)))

    out = []
    for i, a in enumerate(funds):
        for b in funds[i + 1:]:
            wa, wb = by_fund[a], by_fund[b]
            shared = set(wa) & set(wb)
            if not shared:
                out.append({
                    "fund_a": a, "fund_b": b,
                    "shared_count": 0, "overlap_pct": 0.0,
                })
                continue
            overlap_pct = sum(min(wa[k], wb[k]) for k in shared)
            out.append({
                "fund_a": a, "fund_b": b,
                "shared_count": len(shared),
                "overlap_pct": round(float(overlap_pct), 2),
            })
    out.sort(key=lambda r: (-r["overlap_pct"], -r["shared_count"]))
    return out


def company_detail(
    holdings: pd.DataFrame,
    funds_meta: pd.DataFrame | None,
    name_or_key: str,
) -> dict | None:
    """Return one company's holding detail across funds."""
    result = compute_overlap(holdings, funds_meta, selected_funds=None, min_funds=1)
    q = name_or_key.strip().lower()
    q_norm = normalize_company(name_or_key).lower()
    for c in result["companies"]:
        if (c["company_key"].lower() == q
                or c["company"].lower() == q
                or normalize_company(c["company"]).lower() == q_norm
                or q in c["company"].lower()
                or (c.get("isin") and c["isin"].lower() == q)):
            return c
    return None
