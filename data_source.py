"""
data_source.py  --  Single place that loads + cleans the fund data
================================================================
The web app calls load_funds() on EVERY request, so whatever this
function returns is what the website shows. Point it at your Excel file
and edits in Excel appear in the browser on the next refresh.

Priority order (first one found wins):
    1.  funds.xlsx   (your live Excel workbook)   <-- drop your file here
    2.  ranked.csv   (the cleaned fallback)

It auto-detects columns by header keywords, so your Excel headers don't
have to match exactly ("Sharpe Ratio", "sharpe", "SHARPE" all map to
`sharpe`). It also cleans common messiness: strips %, commas, blanks.
If auto-detection ever guesses wrong, hard-wire it in COLUMN_OVERRIDES.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
EXCEL_FILE = BASE / "funds.xlsx"     # <-- save your workbook here
CSV_FILE = BASE / "ranked.csv"       # fallback
EXCEL_SHEET = 0                      # sheet name or index

# canonical_name -> keywords that may appear in your Excel header (lower-cased)
HEADER_KEYWORDS: dict[str, list[str]] = {
    "name":     ["scheme name", "fund name", "scheme", "fund", "name"],
    "amc":      ["amc", "fund house", "house"],
    "aum":      ["aum", "asset", "fund size"],
    "ter":      ["ter", "expense"],
    "cagr":     ["cagr", "since inception", "inception return"],
    "sd":       ["std", "standard dev", "sd", "deviation"],
    "sharpe":   ["sharpe"],
    "sortino":  ["sortino"],
    "beta":     ["beta"],
    "var":      ["var"],
    "pb":       ["p/b", "pb ratio", "price book", "pb"],
    "infratio": ["information", "info ratio", "infratio"],
    "upcap":    ["up capture", "upside capture", "up-capture", "upcap"],
    "downcap":  ["down capture", "downside capture", "down-capture", "downcap"],
    "pe":       ["p/e", "pe ratio", "price earning", "pe"],
    "r1":       ["1 year", "1yr", "1y", "1-year"],
    "r2":       ["2 year", "2yr", "2y", "2-year"],
    "r3":       ["3 year", "3yr", "3y", "3-year"],
    "r5":       ["5 year", "5yr", "5y", "5-year"],
    "r7":       ["7 year", "7yr", "7y", "7-year"],
    "r10":      ["10 year", "10yr", "10y", "10-year"],
    "r15":      ["15 year", "15yr", "15y", "15-year"],
    "category": ["category", "sub category", "fund type", "type"],
}

# force a mapping if auto-detect is wrong:  {"your exact header": "cagr"}
COLUMN_OVERRIDES: dict[str, str] = {}

# columns that are stored as percentages in some sheets and decimals in
# others. We strip "%" everywhere; no scaling is applied (values stay as-is).
NUMERIC_COLS = ["aum", "ter", "cagr", "sd", "sharpe", "sortino", "beta",
                "infratio", "upcap", "downcap", "pe", "pb", "var",
                "r1", "r2", "r3", "r5", "r7", "r10", "r15"]


def _norm(h: str) -> str:
    return re.sub(r"\s+", " ", str(h).strip().lower())


def _auto_map(headers: list[str]) -> dict[str, str]:
    """Return {original_header: canonical_name} best-effort."""
    mapping: dict[str, str] = {}
    used_canon: set[str] = set()
    norm = {h: _norm(h) for h in headers}
    # explicit overrides first
    for h in headers:
        if h in COLUMN_OVERRIDES:
            mapping[h] = COLUMN_OVERRIDES[h]
            used_canon.add(COLUMN_OVERRIDES[h])
    # keyword match (longest keyword wins to avoid 'name' grabbing everything)
    for canon, kws in HEADER_KEYWORDS.items():
        if canon in used_canon:
            continue
        best = None
        for h in headers:
            if h in mapping:
                continue
            for kw in sorted(kws, key=len, reverse=True):
                if kw in norm[h]:
                    best = h
                    break
            if best:
                break
        if best:
            mapping[best] = canon
            used_canon.add(canon)
    return mapping


def _to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--", "NA", "N/A", "na"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_sheet(raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Auto-map + keep canonical columns for one sheet; tag its category."""
    mapping = _auto_map(list(raw.columns))
    df = raw.rename(columns=mapping)
    keep = [c for c in df.columns if c in set(HEADER_KEYWORDS)]
    df = df[keep].copy()
    # a sheet name IS a category (e.g. "Small Cap", "Large Cap", "Flexi Cap").
    if "category" not in df.columns or df["category"].isna().all():
        df["category"] = sheet_name
    return df


def load_funds() -> pd.DataFrame:
    """Load + clean the active data source. Called on every web request.
    Reads EVERY sheet in the workbook and treats each sheet as a category,
    so adding a new category = adding a new sheet (no code change)."""
    if EXCEL_FILE.exists():
        sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)   # dict{name: df}
        parts = [_clean_sheet(raw, name) for name, raw in sheets.items() if len(raw)]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        source = EXCEL_FILE.name
    else:
        df = pd.read_csv(CSV_FILE)
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
        source = CSV_FILE.name

    # clean numerics
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_to_number)

    # beta/sharpe/sortino may arrive as 92 (i.e. 92% = 0.92) or already as 0.92.
    # Only rescale clearly percent-encoded values (thresholds chosen so genuine
    # large ratios are never touched).
    if "beta" in df.columns:
        df["beta"] = df["beta"].apply(lambda v: v / 100 if pd.notna(v) and abs(v) > 3 else v)
    if "sharpe" in df.columns:
        df["sharpe"] = df["sharpe"].apply(lambda v: v / 100 if pd.notna(v) and abs(v) > 15 else v)
    if "sortino" in df.columns:
        df["sortino"] = df["sortino"].apply(lambda v: v / 100 if pd.notna(v) and abs(v) > 15 else v)
    # VaR stored as a positive loss magnitude
    if "var" in df.columns:
        df["var"] = df["var"].abs()

    # tidy text + drop empty rows
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
        df = df[df["name"].notna() & (df["name"] != "") & (df["name"].str.lower() != "nan")]
    if "category" not in df.columns:
        df["category"] = "Small Cap"
    else:
        df["category"] = df["category"].fillna("Small Cap").astype(str).str.strip()

    df = df.reset_index(drop=True)
    df.attrs["source"] = source
    return df


if __name__ == "__main__":
    d = load_funds()
    print("source:", d.attrs.get("source"))
    print("rows:", len(d), "| columns:", list(d.columns))
    print(d.head(3).to_string(index=False))
