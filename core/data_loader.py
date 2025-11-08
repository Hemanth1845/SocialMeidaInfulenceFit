# core/data_loader.py
import os
import re
from functools import lru_cache
from typing import List, Set

import pandas as pd

_CORE_DIR = os.path.dirname(os.path.abspath(__file__))

_BACKEND_DIR = os.path.dirname(_CORE_DIR)


def _env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def _to_snake(s: str) -> str:
    s = s.strip().replace("-", " ").replace("/", " ")
    s = re.sub(r"[^A-Za-z0-9_ ]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s.lower()


PLATFORM_HINTS = {
    "instagram": ["instagram", "ig", "insta"],
    "tiktok": ["tiktok", "tt"],
    "youtube": ["youtube", "yt"],
    "threads": ["threads"],
}

COLUMN_ALIASES = {
    "name": ["name", "full_name", "creator", "channel_name"],
    "handle": ["handle", "username", "user", "account", "profile", "channel_handle", "@"],
    "followers": ["followers", "follower", "subs", "subscribers", "following", "fans", "audience", "audiencia"],
    "er": ["er", "engagement_rate", "engagement", "avg_engagement_rate", "eng_rate", "engagement %", "engagement%"],
    "country": ["country", "location", "region", "pais"],
    "topic_of_influence": ["topic_of_influence", "topic", "topics", "category", "niche", "tema", "tp_tema_de_influencia"],
    "potential_reach": ["potential_reach", "reach", "est_reach", "estimated_reach"],
    "bio": ["bio", "about", "description", "desc"],
    "platform": ["platform", "source"],
}


def _guess_platform_from_filename(path: str) -> str | None:
    fname = os.path.basename(path).lower()
    for plat, hints in PLATFORM_HINTS.items():
        if any(h in fname for h in hints):
            return plat
    return None


def _parse_followers(val) -> float:
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in {"", "-", "na", "n/a", "none"}:
        return float("nan")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmb])?$", s)
    if m:
        num = float(m.group(1))
        suf = (m.group(2) or "").lower()
        if suf == "k":
            return num * 1_000
        if suf == "m":
            return num * 1_000_000
        if suf == "b":
            return num * 1_000_000_000
        return num
    s2 = re.sub(r"[,_ ]", "", s)
    if s2.isdigit():
        return float(int(s2))
    return float("nan")


def _parse_er(val) -> float:
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (int, float)):
        v = float(val)
        return v / 100.0 if v > 1.5 else v
    s = str(val).strip().lower()
    if s in {"", "-", "na", "n/a", "none"}:
        return float("nan")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return float("nan")
    try:
        v = float(s)
        return v / 100.0 if v > 1.5 else v
    except Exception:
        return float("nan")


def _unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    lower_map = {_to_snake(c): c for c in df.columns}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            a_snake = _to_snake(a)
            if a_snake in lower_map:
                mapping[canon] = lower_map[a_snake]
                break
    out = pd.DataFrame()
    for canon in COLUMN_ALIASES.keys():
        out[canon] = df[mapping[canon]] if canon in mapping else ""

    out["followers"] = out["followers"].apply(_parse_followers)
    out["er"] = out["er"].apply(_parse_er)
    for col in ["name", "handle", "platform", "country", "topic_of_influence", "bio"]:
        out[col] = out[col].astype(str).fillna("").map(lambda x: x.strip())
    return out


def _country_from_path(path: str) -> str | None:
    parts = os.path.normpath(path).split(os.sep)
    # .../data/<country>/file.csv
    if "data" in [p.lower() for p in parts]:
        try:
            i = [p.lower() for p in parts].index("data")
            return parts[i + 1].lower()
        except Exception:
            pass
    if len(parts) >= 2:
        return parts[-2].lower()
    return None


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding_errors="ignore", on_bad_lines="skip")
    except Exception:
        try:
            return pd.read_csv(path, dtype=str, encoding="latin1", encoding_errors="ignore", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def _normalize_row(df: pd.DataFrame, file_platform: str | None, file_country: str | None) -> pd.DataFrame:
    if file_platform:
        mask = (df["platform"].astype(str).str.strip() == "")
        df.loc[mask, "platform"] = file_platform
    if file_country:
        mask = (df["country"].astype(str).str.strip() == "")
        df.loc[mask, "country"] = file_country
    df["country"] = df["country"].astype(str).str.lower()

    def norm_handle(h: str) -> str:
        h = (h or "").strip()
        if not h:
            return h
        if h.startswith("@") or " " in h:
            return h
        return f"@{h}"

    df["handle"] = df["handle"].astype(str).map(norm_handle)
    return df


def _scan_csv_files(data_dir: str) -> List[str]:
    csvs: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs


@lru_cache(maxsize=1)
def load_all() -> pd.DataFrame:
   
    data_dir = _env("DATA_DIR", os.path.join(_BACKEND_DIR, "data"))
    
    if not os.path.isdir(data_dir):
        print(f"Warning: Data directory not found at {data_dir}")
        return pd.DataFrame(columns=list(COLUMN_ALIASES.keys()))
        
    files = _scan_csv_files(data_dir)
    frames = []
    for path in files:
        raw = _safe_read_csv(path)
        if raw.empty:
            continue
        uni = _unify_columns(raw)
        fplat = _guess_platform_from_filename(path)
        if not fplat:
            if not uni["platform"].isna().all():
                fplat = (uni["platform"].astype(str).str.lower().mode().iloc[0] or "")
        fcountry = _country_from_path(path)
        uni = _normalize_row(uni, fplat, fcountry)
        frames.append(uni)
    if not frames:
        return pd.DataFrame(columns=list(COLUMN_ALIASES.keys()))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates()
    return df



@lru_cache(maxsize=1)
def available_locations() -> List[str]:

    data_dir = _env("DATA_DIR", os.path.join(_BACKEND_DIR, "data"))

    if not os.path.isdir(data_dir):
        return []
    countries: Set[str] = set()
    for root, dirs, _files in os.walk(data_dir):
        if os.path.abspath(root) == os.path.abspath(data_dir):
            for d in dirs:
                countries.add(d.lower())
            break
    return sorted(countries)


@lru_cache(maxsize=1)
def platforms_present() -> Set[str]:
    df = load_all()
    if df.empty:
        return set()
    vals = set(df["platform"].astype(str).str.lower().unique().tolist())
    vals.discard("")
    return vals