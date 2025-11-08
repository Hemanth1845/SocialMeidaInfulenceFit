# core/nlp.py
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
from difflib import get_close_matches

from .data_loader import available_locations

PLAT_SYNONYMS = {
    "instagram": {"instagram", "ig", "insta"},
    "tiktok": {"tiktok", "tt"},
    "youtube": {"youtube", "yt"},
    "threads": {"threads"},
}

SENIORITY_WORDS = {"senior", "experienced", "pro", "veteran", "expert"}
TOPIC_SEPARATORS = re.compile(r"[,/]|\band\b|\bor\b", flags=re.I)


@dataclass
class ParsedQuery:
    brief: str
    top_k: int = 5
    platforms: Set[str] = field(default_factory=set)
    country: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    follower_min: Optional[int] = None
    follower_max: Optional[int] = None
    sort_by_followers: bool = False
    strict_quality: bool = False
    experience_years: int = 0
    seniority_hint: bool = False


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _extract_top_k(text: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d{1,2})\b", text, flags=re.I)
    if m:
        return _clamp(int(m.group(1)), 1, 20)
    m = re.search(r"\b(\d{1,2})\s*(results|influencers|creators|channels)\b", text, flags=re.I)
    if m:
        return _clamp(int(m.group(1)), 1, 20)
    return _clamp(default, 1, 20)


def _extract_platforms(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    plats: Set[str] = set()
    for plat, syns in PLAT_SYNONYMS.items():
        if tokens & syns:
            plats.add(plat)
    return plats


def _parse_compact_number(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(float(v))
    s = str(v).strip().lower()
    if s in {"", "na", "n/a", "-", "none"}:
        return None
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmb])?$", s)
    if not m:
        try:
            return int(float(s))
        except Exception:
            return None
    num = float(m.group(1))
    suf = (m.group(2) or "").lower()
    mul = 1
    if suf == "k":
        mul = 1_000
    elif suf == "m":
        mul = 1_000_000
    elif suf == "b":
        mul = 1_000_000_000
    return int(num * mul)


def _extract_followers(text: str) -> tuple[Optional[int], Optional[int], bool]:
    t = text.lower()
    sort_by_followers = bool(re.search(r"most\s+following|highest\s+followers|most\s+followers", t))

    fmin = None
    fmax = None

    def parse_num(s: str) -> Optional[int]:
        return _parse_compact_number(s)

    for op, pat in [
        (">", r">\s*([0-9.]+\s*[kmb]?)"),
        ("<", r"<\s*([0-9.]+\s*[kmb]?)"),
        ("under", r"under\s*([0-9.]+\s*[kmb]?)"),
        ("over", r"over\s*([0-9.]+\s*[kmb]?)"),
        ("between", r"between\s*([0-9.]+\s*[kmb]?)\s*(and|-)\s*([0-9.]+\s*[kmb]?)"),
    ]:
        for m in re.finditer(pat, t):
            try:
                if op == "between":
                    lo = parse_num(m.group(1))
                    hi = parse_num(m.group(3))
                    if lo is not None:
                        fmin = lo if fmin is None else max(fmin, lo)
                    if hi is not None:
                        fmax = hi if fmax is None else min(fmax, hi)
                else:
                    val = parse_num(m.group(1))
                    if val is None:
                        continue
                    if op in (">", "over"):
                        fmin = val if fmin is None else max(fmin, val)
                    else:
                        fmax = val if fmax is None else min(fmax, val)
            except Exception:
                pass
    return fmin, fmax, sort_by_followers


def _extract_topics(text: str) -> List[str]:
    parts = [p.strip() for p in TOPIC_SEPARATORS.split(text) if p.strip()]
    topics: List[str] = []
    for p in parts:
        p2 = re.sub(r"[^a-zA-Z0-9 ]+", "", p).strip()
        if not p2:
            continue
        words = p2.lower().split()
        stop_words = {
            "find", "influencers", "influencer", "creators", "creator", "years", "experience",
            "remove", "fake", "followers", "follower", "most", "top", "with", "in", "for", "on",
            "at", "by", "from", "to", "and", "or", "the", "a", "an", "who", "have", "more",
            "than", "under", "over", "between"
        }
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        if filtered:
            topics.append(" ".join(filtered[:4]))
    seen = set()
    out = []
    for t in topics:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _extract_country(text: str) -> Optional[str]:
    locs = available_locations()
    if not locs:
        return None
    words = set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()))
    for w in words:
        if w in locs:
            return w
    cand = get_close_matches(" ".join(words), locs, n=1, cutoff=0.6)
    if cand:
        return cand[0]
    for w in words:
        m = get_close_matches(w, locs, n=1, cutoff=0.8)
        if m:
            return m[0]
    return None


def parse_brief(text: str, ui_overrides: Dict | None = None) -> ParsedQuery:
    ui_overrides = ui_overrides or {}
    pq = ParsedQuery(brief=text or "")
    pq.top_k = _extract_top_k(text, default=ui_overrides.get("top_k", 5))

    plats = _extract_platforms(text)
    if ui_overrides.get("platforms"):
        plats |= {p.lower() for p in ui_overrides.get("platforms")}
    pq.platforms = plats

    pq.country = ui_overrides.get("location") or _extract_country(text)
    pq.topics = ui_overrides.get("topics") or _extract_topics(text)

    fmin_txt = ui_overrides.get("follower_min")
    fmax_txt = ui_overrides.get("follower_max")
    fmin_b, fmax_b, sortf_b = _extract_followers(text)
    fmin_ui = _parse_compact_number(fmin_txt)
    fmax_ui = _parse_compact_number(fmax_txt)
    pq.follower_min = fmin_ui if fmin_ui is not None else fmin_b
    pq.follower_max = fmax_ui if fmax_ui is not None else fmax_b
    pq.sort_by_followers = sortf_b

    pq.strict_quality = bool(ui_overrides.get("strict_quality")) or bool(
        re.search(r"remove\s+fake|ignore\s+fake|inactive", text, flags=re.I)
    )
    pq.experience_years = int(ui_overrides.get("experience", 0))
    pq.seniority_hint = any(w in text.lower() for w in SENIORITY_WORDS) or (pq.experience_years >= 4)
    return pq
