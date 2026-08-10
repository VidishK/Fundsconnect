"""
fund_scorer.py  --  Quantitative fund-scoring engine (Phase 1)
================================================================
Category-aware, profile-driven scoring for mutual funds of ANY category
(small cap, large cap, flexi cap, ELSS, ...). Funds are always ranked
against peers in their OWN category.

Design notes
------------
* METRIC_DIRECTION is the single source of truth for whether a metric is
  "higher is better" (max) or "lower is better" (min).
* RISK_PROFILES are just weight vectors (+ optional direction overrides).
  The "risk dial" is a blend between two profiles -- see blend_profiles().
* Scoring = percentile-rank each metric WITHIN category -> weighted mean,
  renormalised over the metrics each fund actually has (missing long-period
  data never penalises newer funds).
* EXTENSIBLE: to add rolling returns later, add e.g. "roll3_avg":"max" to
  METRIC_DIRECTION and give it a weight in the profiles. Nothing else changes.
  The qualitative / RAG layer sits OUTSIDE this module by design -- this
  engine only ever touches numbers.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# 1. METRIC LIBRARY  --  canonical optimisation direction per metric
# ----------------------------------------------------------------------
METRIC_DIRECTION: dict[str, str] = {
    # returns (higher is better)
    "cagr": "max", "r1": "max", "r2": "max", "r3": "max",
    "r5": "max", "r7": "max", "r10": "max", "r15": "max",
    # risk-adjusted (higher is better)
    "sharpe": "max", "sortino": "max", "upcap": "max", "infratio": "max",
    # risk / cost / valuation (lower is better)
    "downcap": "min", "sd": "min", "beta": "min", "pe": "min", "ter": "min",
    "var": "min",   # historical Value-at-Risk (worst-case loss), from source data
    # --- future plug-ins (uncomment when data is wired) ---
    # "roll3_avg": "max", "roll5_avg": "max", "roll3_min": "max", "maxdd": "min",
}

# ----------------------------------------------------------------------
# 2. RISK PROFILES  --  weights (0 = metric ignored) + direction overrides
# ----------------------------------------------------------------------
@dataclass
class Profile:
    name: str
    weights: dict
    direction_overrides: dict = field(default_factory=dict)  # e.g. {"beta": "max"}

RISK_PROFILES: dict[str, Profile] = {
    "conservative": Profile(
        "Conservative",
        {"cagr": 6, "r1": 3, "r2": 4, "r3": 8, "r5": 9, "r7": 7, "r10": 5, "r15": 3,
         "sharpe": 14, "sortino": 6, "upcap": 6, "infratio": 8,
         "downcap": 12, "sd": 9, "beta": 7, "pe": 4, "var": 10}),

    "balanced": Profile(
        "Balanced",
        {"cagr": 6, "r1": 3, "r2": 4, "r3": 8, "r5": 9, "r7": 7, "r10": 5, "r15": 3,
         "sharpe": 13, "sortino": 6, "upcap": 10, "infratio": 10,
         "downcap": 8, "sd": 5, "beta": 4, "pe": 3, "var": 6}),

    "growth": Profile(
        "Growth",
        {"cagr": 8, "r1": 5, "r2": 5, "r3": 9, "r5": 8, "r7": 5, "r10": 3, "r15": 2,
         "sharpe": 10, "sortino": 5, "upcap": 13, "infratio": 9,
         "downcap": 6, "sd": 2, "beta": 2, "pe": 2, "var": 4}),

    # "more risk to earn more, faster" -> short-horizon returns weighted heavily,
    # high beta REWARDED (direction flipped), SD/PE penalties dropped.
    "aggressive": Profile(
        "Aggressive (max return, short horizon)",
        {"cagr": 9, "r1": 10, "r2": 8, "r3": 9, "r5": 5, "r7": 3, "r10": 2, "r15": 1,
         "upcap": 15, "beta": 12, "infratio": 8, "sharpe": 6, "sortino": 3,
         "downcap": 5, "var": 3},
        direction_overrides={"beta": "max"}),
}

ORDER = ["conservative", "balanced", "growth", "aggressive"]

# ----------------------------------------------------------------------
# 3. CORE SCORING
# ----------------------------------------------------------------------
def _pct_score(s: pd.Series, direction: str) -> pd.Series:
    """Percentile rank 0-100. NaN stays NaN. 'min' direction is inverted."""
    rank = s.rank(pct=True)
    if direction == "min":
        rank = 1.0 - rank
    return rank * 100.0

def score_funds(df: pd.DataFrame,
                profile: str | Profile = "balanced",
                category_aware: bool = True,
                category_col: str = "category") -> pd.DataFrame:
    """
    Score every fund and return the dataframe with a 'score' column (0-100),
    sorted best-first. When category_aware, percentile ranks are computed
    WITHIN each category so you never compare a large cap to a small cap.
    """
    prof = RISK_PROFILES[profile] if isinstance(profile, str) else profile
    weights = {m: w for m, w in prof.weights.items() if w > 0}

    if category_aware and category_col in df.columns:
        groups = list(df.groupby(category_col))
    else:
        groups = [("ALL", df)]

    scored = []
    for _, g in groups:
        # percentile score per metric within this category
        S = pd.DataFrame(index=g.index)
        for m in weights:
            if m in g.columns:
                direction = prof.direction_overrides.get(m, METRIC_DIRECTION[m])
                S[m] = _pct_score(g[m], direction)
        w = pd.Series({m: weights[m] for m in S.columns}, dtype=float)
        # renormalised weighted mean over available metrics, per fund
        vals = []
        for i in g.index:
            row = S.loc[i]
            av = row.notna()
            wsum = w[av.index][av].sum()
            vals.append((row[av] * w[av.index][av]).sum() / wsum if wsum > 0 else np.nan)
        gg = g.copy()
        gg["score"] = np.round(vals, 1)
        scored.append(gg)

    res = pd.concat(scored)
    sort_cols = ([category_col, "score"] if category_aware and category_col in df.columns
                 else ["score"])
    asc = ([True, False] if len(sort_cols) == 2 else [False])
    return res.sort_values(sort_cols, ascending=asc)

def top_n(df: pd.DataFrame, profile="balanced", n=5,
          category_aware=True, min_history: str | None = None,
          category_col="category") -> pd.DataFrame:
    """Top-n per category for a profile. min_history e.g. 'r5' filters out
    funds without a 5-year record before taking the top n."""
    ranked = score_funds(df, profile, category_aware, category_col)
    if min_history:
        ranked = ranked[ranked[min_history].notna()]
    if category_aware and category_col in df.columns:
        return ranked.groupby(category_col, group_keys=False).head(n)
    return ranked.head(n)

def blend_profiles(p_low: str, p_high: str, t: float) -> Profile:
    """Continuous risk dial: t in [0,1] linearly blends two profiles' weights.
    direction_overrides activate once t >= 0.5 (toward the high-risk profile)."""
    a, b = RISK_PROFILES[p_low], RISK_PROFILES[p_high]
    keys = set(a.weights) | set(b.weights)
    w = {k: round((1 - t) * a.weights.get(k, 0) + t * b.weights.get(k, 0), 2) for k in keys}
    ov = b.direction_overrides if t >= 0.5 else {}
    return Profile(f"Blend({p_low}->{p_high}, t={t:.2f})", w, ov)
