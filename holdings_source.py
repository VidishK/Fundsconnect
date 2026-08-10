"""
holdings_source.py  --  Load mutual-fund holdings from drop-in files
================================================================
Priority order (first match wins):
    1. holdings.xlsx
    2. holdings.csv
    3. holdings/ folder  (one CSV/Excel per fund; stem = fund name
       when the file has no fund column)

Returns a tidy long table with columns:
    fund, company, weight, isin?, sector?
Weight is stored as percent (4.2 == 4.2% of AUM).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
EXCEL_FILE = BASE / "holdings.xlsx"
CSV_FILE = BASE / "holdings.csv"
HOLDINGS_DIR = BASE / "holdings"

HEADER_KEYWORDS: dict[str, list[str]] = {
    "fund":    ["scheme name", "fund name", "scheme", "fund", "portfolio"],
    "company": ["company name", "company", "stock name", "stock", "security name",
                "security", "instrument name", "instrument", "scrip", "holding"],
    "weight":  ["% of assets", "% of aum", "portfolio %", "weight %", "weight%",
                "holding %", "% holding", "allocation", "weight", "%"],
    "isin":    ["isin"],
    "sector":  ["sector", "industry", "gics"],
    "pe":      ["p/e", "pe ratio", "price to earning", "price earning"],
    "pb":      ["p/b", "pb ratio", "price to book", "price book"],
}

# force a mapping if auto-detect is wrong: {"Your Exact Header": "company"}
COLUMN_OVERRIDES: dict[str, str] = {}

CANONICAL = list(HEADER_KEYWORDS.keys())


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h).strip().lower())


def _auto_map(headers: list[str]) -> dict[str, str]:
    """Return {original_header: canonical_name} best-effort."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    norms = {h: _norm_header(h) for h in headers}

    for h in headers:
        if h in COLUMN_OVERRIDES:
            mapping[h] = COLUMN_OVERRIDES[h]
            used.add(COLUMN_OVERRIDES[h])

    # Prefer longer keywords so "scheme name" beats "name", "% of assets" beats "%"
    for canon, kws in HEADER_KEYWORDS.items():
        if canon in used:
            continue
        best_h, best_len = None, -1
        for h in headers:
            if h in mapping:
                continue
            n = norms[h]
            for kw in kws:
                if kw in n and len(kw) > best_len:
                    # Avoid mapping a lone "name" column to fund when company is free
                    if canon == "fund" and kw in ("name",) and "fund" not in n and "scheme" not in n:
                        continue
                    if canon == "company" and kw == "holding" and "holdings" in n and "name" not in n:
                        # "No. of Holdings" style columns — skip
                        if any(x in n for x in ("no", "number", "count", "#")):
                            continue
                    best_h, best_len = h, len(kw)
        if best_h is not None:
            mapping[best_h] = canon
            used.add(canon)
    return mapping


def _to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--", "NA", "N/A", "na", "nil"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _clean_frame(raw: pd.DataFrame, default_fund: str | None = None) -> pd.DataFrame:
    """Map + clean one sheet/file into canonical holdings rows."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CANONICAL)

    # Drop fully empty / Unnamed junk columns
    cols = [c for c in raw.columns if not str(c).startswith("Unnamed")]
    raw = raw.loc[:, cols].copy()
    if raw.empty:
        return pd.DataFrame(columns=CANONICAL)

    mapping = _auto_map(list(raw.columns))
    df = raw.rename(columns=mapping)
    keep = [c for c in CANONICAL if c in df.columns]
    df = df[keep].copy()

    if "fund" not in df.columns:
        if default_fund:
            df["fund"] = default_fund
        else:
            return pd.DataFrame(columns=CANONICAL)

    if "company" not in df.columns or "weight" not in df.columns:
        return pd.DataFrame(columns=CANONICAL)

    df["fund"] = df["fund"].astype(str).str.strip()
    df["company"] = df["company"].astype(str).str.strip()
    df["weight"] = df["weight"].apply(_to_number)

    # If weights look like fractions (0.042), scale to percent
    valid = df["weight"].dropna()
    if len(valid) and valid.max() <= 1.5 and valid.min() >= 0:
        df["weight"] = df["weight"] * 100.0

    if "isin" in df.columns:
        df["isin"] = df["isin"].astype(str).str.strip().str.upper()
        df.loc[df["isin"].isin(("NAN", "NONE", "", "NA", "N/A")), "isin"] = np.nan
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()
        df.loc[df["sector"].str.lower().isin(("nan", "none", "")), "sector"] = np.nan
    for col in ("pe", "pb"):
        if col in df.columns:
            df[col] = df[col].apply(_to_number)

    df = df[df["fund"].notna() & (df["fund"] != "") & (df["fund"].str.lower() != "nan")]
    df = df[df["company"].notna() & (df["company"] != "") & (df["company"].str.lower() != "nan")]
    df = df[df["weight"].notna() & (df["weight"] > 0)]

    # Deduplicate identical fund+company (+isin) keeping max weight
    keys = ["fund", "company"]
    if "isin" in df.columns:
        keys.append("isin")
    df = (df.sort_values("weight", ascending=False)
            .drop_duplicates(subset=keys, keep="first")
            .reset_index(drop=True))
    return df


def _read_tabular(path: Path) -> list[tuple[str | None, pd.DataFrame]]:
    """Return list of (sheet_or_stem_hint, dataframe)."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        return [(name, raw) for name, raw in sheets.items() if raw is not None and len(raw)]
    if suffix == ".csv":
        return [(None, pd.read_csv(path))]
    return []


def holdings_status() -> dict:
    """Lightweight status for /api/overlap/status without full load."""
    import datetime as _dt
    active = None
    kind = None
    if EXCEL_FILE.exists():
        active, kind = EXCEL_FILE, "excel"
    elif CSV_FILE.exists():
        active, kind = CSV_FILE, "csv"
    elif HOLDINGS_DIR.is_dir():
        files = sorted(
            p for p in HOLDINGS_DIR.iterdir()
            if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith(".")
        )
        if files:
            active, kind = HOLDINGS_DIR, "folder"
            mtime = max(p.stat().st_mtime for p in files)
            return {
                "present": True,
                "source": f"holdings/ ({len(files)} files)",
                "kind": kind,
                "updated_at": _dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "updated_ts": mtime,
            }
    if active is None:
        return {"present": False, "source": None, "kind": None,
                "updated_at": None, "updated_ts": 0}
    mtime = active.stat().st_mtime
    return {
        "present": True,
        "source": active.name if kind != "folder" else "holdings/",
        "kind": kind,
        "updated_at": _dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
        "updated_ts": mtime,
    }


def load_holdings() -> pd.DataFrame:
    """Load + clean holdings. Empty DataFrame if nothing dropped in."""
    parts: list[pd.DataFrame] = []
    source = None

    if EXCEL_FILE.exists():
        source = EXCEL_FILE.name
        for sheet_name, raw in _read_tabular(EXCEL_FILE):
            # Multi-sheet workbook: if no fund column, treat sheet name as fund
            parts.append(_clean_frame(raw, default_fund=sheet_name))
    elif CSV_FILE.exists():
        source = CSV_FILE.name
        for _, raw in _read_tabular(CSV_FILE):
            parts.append(_clean_frame(raw))
    elif HOLDINGS_DIR.is_dir():
        files = sorted(
            p for p in HOLDINGS_DIR.iterdir()
            if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith(".")
        )
        if files:
            source = f"holdings/ ({len(files)} files)"
            for path in files:
                for sheet_name, raw in _read_tabular(path):
                    default = sheet_name if sheet_name else path.stem.replace("_", " ").replace("-", " ")
                    parts.append(_clean_frame(raw, default_fund=default))

    if not parts:
        df = pd.DataFrame(columns=CANONICAL)
        df.attrs["source"] = None
        return df

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if df.empty:
        df = pd.DataFrame(columns=CANONICAL)
    df.attrs["source"] = source
    return df


# ----------------------------------------------------------------------
# Upload parse + save (Data tab)
# ----------------------------------------------------------------------
UPLOAD_META = BASE / ".holdings_upload_meta.json"
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}


def _frames_from_json(obj, default_fund: str | None) -> list[tuple[str | None, pd.DataFrame]]:
    """Turn common JSON shapes into (default_fund_hint, DataFrame) parts."""
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(x, dict) for x in obj):
            return [(default_fund, pd.DataFrame(obj))]
        return []

    if not isinstance(obj, dict):
        return []

    # { "holdings" | "data" | "rows": [ {...}, ... ] }
    for key in ("holdings", "data", "rows", "records"):
        if key in obj and isinstance(obj[key], list):
            return _frames_from_json(obj[key], default_fund)

    # { "fund": "...", "holdings": [...] }
    if "fund" in obj and isinstance(obj.get("holdings"), list):
        return _frames_from_json(obj["holdings"], str(obj["fund"]))

    # { "Bandhan Small Cap Fund": [ {...} ], ... }
    parts = []
    for k, v in obj.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            parts.append((str(k), pd.DataFrame(v)))
        elif isinstance(v, dict) and any(
            isinstance(v.get(x), list) for x in ("holdings", "data", "rows")
        ):
            for key in ("holdings", "data", "rows"):
                if isinstance(v.get(key), list):
                    parts.append((str(k), pd.DataFrame(v[key])))
                    break
    return parts


def _read_bytes(filename: str, data: bytes) -> list[tuple[str | None, pd.DataFrame]]:
    """Parse upload bytes into raw frames."""
    import io
    import json as _json

    name = (filename or "upload.csv").strip()
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported file type '{suffix or '(none)'}'. "
            f"Use: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".json":
        text = data.decode("utf-8-sig")
        obj = _json.loads(text)
        return _frames_from_json(obj, None)

    bio = io.BytesIO(data)
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(bio, sheet_name=None)
        return [(n, raw) for n, raw in sheets.items() if raw is not None and len(raw)]
    # csv
    return [(None, pd.read_csv(bio))]


def _mapping_report(raw: pd.DataFrame) -> dict:
    headers = [str(c) for c in raw.columns if not str(c).startswith("Unnamed")]
    mapping = _auto_map(headers)
    return {
        "headers": headers,
        "mapping": mapping,  # original -> canonical
        "mapped": {v: k for k, v in mapping.items()},  # canonical -> original
        "has_fund": "fund" in mapping.values(),
        "has_company": "company" in mapping.values(),
        "has_weight": "weight" in mapping.values(),
    }


def parse_upload(
    filename: str,
    data: bytes,
    default_fund: str | None = None,
) -> dict:
    """
    Parse an uploaded file into tidy holdings + diagnostics.
    Does not write to disk.
    """
    warnings: list[str] = []
    errors: list[str] = []
    mapping_reports: list[dict] = []
    parts: list[pd.DataFrame] = []

    try:
        frames = _read_bytes(filename, data)
    except Exception as e:
        return {
            "ok": False,
            "errors": [str(e)],
            "warnings": [],
            "rows": 0,
            "funds": 0,
            "mapping": {},
            "sample": [],
            "df": pd.DataFrame(columns=CANONICAL),
        }

    if not frames:
        return {
            "ok": False,
            "errors": ["file is empty or could not be read"],
            "warnings": [],
            "rows": 0,
            "funds": 0,
            "mapping": {},
            "sample": [],
            "df": pd.DataFrame(columns=CANONICAL),
        }

    stem_default = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    for sheet_hint, raw in frames:
        report = _mapping_report(raw)
        report["sheet"] = sheet_hint
        mapping_reports.append(report)

        # Prefer: fund column > sheet name > form default_fund > filename stem
        fund_hint = None
        if report["has_fund"]:
            fund_hint = None  # column present
        elif sheet_hint and str(sheet_hint).lower() not in ("sheet1", "sheet 1", "data"):
            fund_hint = str(sheet_hint)
        elif default_fund:
            fund_hint = default_fund
        elif len(frames) == 1:
            fund_hint = stem_default or default_fund

        cleaned = _clean_frame(raw, default_fund=fund_hint)
        if cleaned.empty:
            missing = []
            if not report["has_company"]:
                missing.append("company/stock")
            if not report["has_weight"]:
                missing.append("weight/%")
            if not report["has_fund"] and not fund_hint:
                missing.append("fund (or set a default fund name)")
            label = sheet_hint or "file"
            errors.append(
                f"'{label}': could not map required columns"
                + (f" — missing {', '.join(missing)}" if missing else "")
                + f". Headers seen: {', '.join(report['headers'][:12]) or '(none)'}"
            )
        else:
            if not report["has_fund"] and fund_hint:
                warnings.append(f"No fund column in '{sheet_hint or filename}' — used '{fund_hint}'.")
            parts.append(cleaned)

    if not parts:
        return {
            "ok": False,
            "errors": errors or ["no usable holdings rows found"],
            "warnings": warnings,
            "rows": 0,
            "funds": 0,
            "mapping": mapping_reports[0]["mapping"] if mapping_reports else {},
            "mappings": mapping_reports,
            "sample": [],
            "df": pd.DataFrame(columns=CANONICAL),
        }

    df = pd.concat(parts, ignore_index=True)
    # Dedupe across sheets
    keys = ["fund", "company"] + (["isin"] if "isin" in df.columns else [])
    df = (df.sort_values("weight", ascending=False)
            .drop_duplicates(subset=keys, keep="first")
            .reset_index(drop=True))

    sample_cols = [c for c in CANONICAL if c in df.columns]
    sample = df[sample_cols].head(8).replace({np.nan: None}).to_dict("records")
    # JSON-safe floats
    for row in sample:
        for k, v in list(row.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None

    merged_map = {}
    for rep in mapping_reports:
        merged_map.update(rep.get("mapping") or {})

    return {
        "ok": True,
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "funds": int(df["fund"].nunique()),
        "fund_names": sorted(df["fund"].unique().tolist()),
        "mapping": merged_map,
        "mappings": mapping_reports,
        "sample": sample,
        "df": df,
    }


def save_holdings_df(df: pd.DataFrame, original_filename: str | None = None) -> dict:
    """
    Persist tidy holdings as holdings.csv (canonical).
    Removes holdings.xlsx so CSV becomes the active source.
    """
    import datetime as _dt
    import json as _json

    if df is None or df.empty:
        raise ValueError("nothing to save")

    out_cols = [c for c in CANONICAL if c in df.columns]
    tidy = df[out_cols].copy()
    tidy.to_csv(CSV_FILE, index=False)

    # Excel takes priority in load_holdings — retire it so upload wins
    if EXCEL_FILE.exists():
        bak = BASE / "holdings.xlsx.bak"
        try:
            if bak.exists():
                bak.unlink()
            EXCEL_FILE.rename(bak)
        except OSError:
            EXCEL_FILE.unlink(missing_ok=True)

    meta = {
        "original_filename": original_filename,
        "saved_as": CSV_FILE.name,
        "rows": int(len(tidy)),
        "funds": int(tidy["fund"].nunique()) if "fund" in tidy.columns else 0,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    UPLOAD_META.write_text(_json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    d = load_holdings()
    print("source:", d.attrs.get("source"))
    print("rows:", len(d), "| funds:", d["fund"].nunique() if len(d) else 0)
    print("columns:", list(d.columns))
    if len(d):
        print(d.head(5).to_string(index=False))
