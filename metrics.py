"""
metrics.py  --  Derived-metric engine + formula glossary
================================================================
Two jobs:

1.  enrich_metrics(df)  ->  adds parametric Value-at-Risk (VaR) and
    Conditional VaR / Expected Shortfall columns, computed from each
    fund's annualised standard deviation (SD) and CAGR.

2.  GLOSSARY  ->  a single source of truth describing EVERY metric the
    algorithm uses: what it means, the exact formula, its optimisation
    direction (maximise / minimise) and a healthy reference range.
    The FastAPI layer serves this verbatim at /api/glossary so the
    website can show users precisely how each number was produced.

VaR note
--------
True VaR needs a daily NAV history (added in the rolling-returns phase).
Until then we use the standard *parametric / variance-covariance* method,
which assumes returns are approximately normal:

        VaR_alpha = z_alpha * sigma          (loss, zero-mean convention)
        VaR_alpha(mean-adj) = z_alpha * sigma - mu

where sigma = annualised SD, mu = expected annual return (CAGR proxy),
and z_alpha is the standard-normal quantile (1.645 at 95%, 2.326 at 99%).
Conditional VaR (Expected Shortfall) is the average loss beyond VaR:

        CVaR_alpha = sigma * phi(z_alpha) / (1 - alpha)

phi() is the standard-normal PDF.  Swap these for empirical NAV-based VaR
once the NAV pipeline lands -- the column names stay the same.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# standard-normal quantiles (z) and PDF values at the common confidence levels
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def _phi(z: float) -> float:
    """Standard-normal probability density at z."""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def parametric_var(sd: float, conf: float = 0.95, horizon: str = "annual",
                   mean: float | None = None) -> float:
    """
    Parametric VaR as a POSITIVE percentage loss.

    sd       : annualised standard deviation (in %, e.g. 17.26)
    conf     : confidence level (0.90 / 0.95 / 0.99)
    horizon  : 'annual' or 'monthly' (monthly scales sigma by 1/sqrt(12))
    mean     : if given, mean-adjusts the VaR (z*sigma - mu); else zero-mean
    """
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return float("nan")
    z = _Z[conf]
    sigma = sd / math.sqrt(12) if horizon == "monthly" else sd
    var = z * sigma
    if mean is not None and not (isinstance(mean, float) and math.isnan(mean)):
        mu = mean / 12 if horizon == "monthly" else mean
        var = var - mu
    return round(var, 2)


def conditional_var(sd: float, conf: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall (annual, %), positive loss."""
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return float("nan")
    z = _Z[conf]
    return round(sd * _phi(z) / (1 - conf), 2)


def enrich_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with VaR / CVaR columns added (from SD & CAGR)."""
    out = df.copy()
    sd = out["sd"]
    cagr = out.get("cagr")
    out["var95_annual"] = sd.apply(lambda x: parametric_var(x, 0.95, "annual"))
    out["var99_annual"] = sd.apply(lambda x: parametric_var(x, 0.99, "annual"))
    out["var95_monthly"] = sd.apply(lambda x: parametric_var(x, 0.95, "monthly"))
    out["cvar95_annual"] = sd.apply(lambda x: conditional_var(x, 0.95))
    if cagr is not None:
        out["var95_annual_madj"] = [
            parametric_var(s, 0.95, "annual", m) for s, m in zip(sd, cagr)
        ]
    return out


# ----------------------------------------------------------------------
# GLOSSARY  --  served to the website at /api/glossary
# Each entry: label, group, direction, formula, meaning, healthy range
# ----------------------------------------------------------------------
GLOSSARY: dict[str, dict] = {
    "cagr": {
        "label": "CAGR (Since Inception)", "group": "Return", "direction": "maximise",
        "formula": "CAGR = (Ending NAV / Beginning NAV)^(1/years) - 1",
        "meaning": "Compounded annual growth rate over the fund's life. The smoothed yearly return that turns the starting NAV into the latest one.",
        "range": "Higher is better; for small caps 20%+ over a full cycle is strong."},
    "r1": {"label": "1-Year Return", "group": "Return", "direction": "maximise",
        "formula": "R_1Y = (NAV_today / NAV_1yr_ago) - 1",
        "meaning": "Trailing one-year point-to-point return. Most sensitive to recent momentum.",
        "range": "Context-dependent; compare within category."},
    "r2": {"label": "2-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_2Y = (NAV_today / NAV_2yr_ago)^(1/2) - 1",
        "meaning": "Annualised return over two years.", "range": "Compare within category."},
    "r3": {"label": "3-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_3Y = (NAV_today / NAV_3yr_ago)^(1/3) - 1",
        "meaning": "Annualised three-year return; a common medium-term yardstick.",
        "range": "Compare within category."},
    "r5": {"label": "5-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_5Y = (NAV_today / NAV_5yr_ago)^(1/5) - 1",
        "meaning": "Annualised five-year return; spans most of a market cycle.",
        "range": "Key proven-track-record screen."},
    "r7": {"label": "7-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_7Y = (NAV_today / NAV_7yr_ago)^(1/7) - 1",
        "meaning": "Annualised seven-year return.", "range": "Long-term consistency."},
    "r10": {"label": "10-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_10Y = (NAV_today / NAV_10yr_ago)^(1/10) - 1",
        "meaning": "Annualised ten-year return; full multi-cycle evidence.",
        "range": "Only mature funds have it."},
    "r15": {"label": "15-Year Return (annualised)", "group": "Return", "direction": "maximise",
        "formula": "R_15Y = (NAV_today / NAV_15yr_ago)^(1/15) - 1",
        "meaning": "Annualised fifteen-year return.", "range": "Very few funds qualify."},
    "sharpe": {"label": "Sharpe Ratio", "group": "Risk-adjusted", "direction": "maximise",
        "formula": "Sharpe = (R_p - R_f) / sigma_p",
        "meaning": "Excess return per unit of TOTAL risk. R_p = fund return, R_f = risk-free rate, sigma_p = SD. Higher means you are paid more for the volatility you take.",
        "range": "> 1 good, > 1.5 excellent."},
    "sortino": {"label": "Sortino Ratio", "group": "Risk-adjusted", "direction": "maximise",
        "formula": "Sortino = (R_p - R_f) / sigma_downside",
        "meaning": "Like Sharpe but only penalises DOWNSIDE volatility, ignoring upside swings.",
        "range": "> 2 is strong."},
    "upcap": {"label": "Up-Capture Ratio", "group": "Capture", "direction": "maximise",
        "formula": "UpCapture = (fund return in up-months) / (benchmark return in up-months) * 100",
        "meaning": "How much of the market's GAINS the fund captures when the market rises.",
        "range": "> 100 means it beats the index in up-markets."},
    "downcap": {"label": "Down-Capture Ratio", "group": "Capture", "direction": "minimise",
        "formula": "DownCapture = (fund return in down-months) / (benchmark return in down-months) * 100",
        "meaning": "How much of the market's LOSSES the fund suffers when the market falls.",
        "range": "< 100 means it loses less than the index in down-markets."},
    "infratio": {"label": "Information Ratio", "group": "Risk-adjusted", "direction": "maximise",
        "formula": "IR = (R_p - R_benchmark) / tracking_error",
        "meaning": "Consistency of active outperformance versus the benchmark per unit of active risk.",
        "range": "> 0.5 good, > 1 outstanding."},
    "sd": {"label": "Standard Deviation", "group": "Risk", "direction": "minimise",
        "formula": "sigma = sqrt( sum( (R_i - mean(R))^2 ) / (n - 1) ), annualised",
        "meaning": "Total volatility of returns. Higher SD = wider swings in both directions.",
        "range": "Lower for the same return is better."},
    "beta": {"label": "Beta", "group": "Risk", "direction": "minimise *",
        "formula": "Beta = Cov(R_fund, R_market) / Var(R_market)",
        "meaning": "Sensitivity to market moves. Beta 1.2 = ~20% more volatile than the index. * In the Aggressive profile beta is REWARDED (flipped to maximise) because high beta amplifies upside.",
        "range": "< 1 defensive, > 1 aggressive."},
    "pe": {"label": "Portfolio P/E", "group": "Valuation", "direction": "minimise",
        "formula": "P/E = weighted average of (price / earnings) across holdings",
        "meaning": "How expensively the portfolio's stocks are priced relative to earnings. Lower can mean more value headroom.",
        "range": "Lower is cheaper; judge vs category."},
    "ter": {"label": "Total Expense Ratio", "group": "Cost", "direction": "minimise",
        "formula": "TER = annual fund costs / average AUM * 100",
        "meaning": "Yearly fee drag deducted from returns. Every 1% of TER is 1% off your return.",
        "range": "Lower is better; direct plans are cheaper."},
    "var": {"label": "VaR (historical)", "group": "Risk", "direction": "minimise",
        "formula": "VaR = worst realised loss at the source confidence level (from data)",
        "meaning": "The actual historical Value-at-Risk reported in your dataset (the worst-case loss the fund has shown). Used as the PRIMARY VaR; the parametric VaR below is kept only for comparison. Stored as a positive loss, so lower is safer.",
        "range": "Lower is safer; e.g. 14% is far calmer than 36%."},
    "pb": {"label": "Portfolio P/B", "group": "Valuation", "direction": "minimise",
        "formula": "P/B = weighted average of (price / book value) across holdings",
        "meaning": "How expensively the portfolio trades relative to book value. Lower can signal more value cushion.",
        "range": "Lower is cheaper; judge vs category."},
    "aum": {"label": "AUM (Cr)", "group": "Size", "direction": "context",
        "formula": "AUM = total market value of all assets the fund manages",
        "meaning": "Fund size. Very large AUM can make small-cap strategies harder to deploy nimbly.",
        "range": "Context-dependent."},
    # --- derived in this module ---
    "var95_annual": {"label": "VaR 95% (1yr)", "group": "Risk", "direction": "minimise",
        "formula": "VaR_95 = 1.645 * sigma   (parametric, zero-mean, annual)",
        "meaning": "With 95% confidence, the most you would expect to lose over a year in a normal bad scenario. A VaR of 28 means '~1 year in 20 could be worse than a 28% loss'.",
        "range": "Lower is safer."},
    "var99_annual": {"label": "VaR 99% (1yr)", "group": "Risk", "direction": "minimise",
        "formula": "VaR_99 = 2.326 * sigma   (parametric, zero-mean, annual)",
        "meaning": "The 1-in-100 bad-year loss threshold. More conservative tail estimate than 95%.",
        "range": "Lower is safer."},
    "var95_monthly": {"label": "VaR 95% (1mo)", "group": "Risk", "direction": "minimise",
        "formula": "VaR_95_monthly = 1.645 * (sigma / sqrt(12))",
        "meaning": "The 1-in-20 bad-MONTH loss. Useful for shorter holding horizons.",
        "range": "Lower is safer."},
    "cvar95_annual": {"label": "CVaR 95% (Expected Shortfall)", "group": "Risk", "direction": "minimise",
        "formula": "CVaR_95 = sigma * phi(1.645) / (1 - 0.95)",
        "meaning": "The AVERAGE loss in the worst 5% of years -- i.e. how bad it gets once you are already past the VaR line. Captures tail severity that VaR alone hides.",
        "range": "Lower is safer."},
    "var95_annual_madj": {"label": "VaR 95% (mean-adjusted)", "group": "Risk", "direction": "minimise",
        "formula": "VaR_95_madj = 1.645 * sigma - mu   (mu = CAGR)",
        "meaning": "VaR after crediting the fund's expected return. A negative value means the expected drift more than offsets the 95% downside.",
        "range": "Lower is safer."},
}


if __name__ == "__main__":
    df = pd.read_csv("ranked.csv")
    e = enrich_metrics(df)
    cols = ["name", "sd", "cagr", "var95_annual", "var99_annual",
            "var95_monthly", "cvar95_annual", "var95_annual_madj"]
    print(e[cols].head(8).to_string(index=False))
