# core/scoring.py
import re
from typing import List, Dict, Any, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOW_QUALITY_KEYWORDS = ["fanpage", "unofficial", "giveaway", "free followers", "shop", "spam"]


def _quality_penalty(row: pd.Series, strict: bool) -> float:
    penalty = 0.0
    followers = row.get("followers") or 0
    er = row.get("er")
    name = str(row.get("name") or "")
    bio = str(row.get("bio") or "")

    if not followers or followers <= 0:
        return 100.0 if strict else 50.0

    if (pd.isna(er) or er == 0) and followers > 100_000:
        return 60.0 if strict else 25.0

    if followers >= 5_000_000 and (pd.isna(er) or er < 0.0002):
        return 100.0 if strict else 40.0
    if 100_000 <= followers < 1_000_000 and (pd.isna(er) or er < 0.001):
        return 100.0 if strict else 35.0

    text = f"{name} {bio}".lower()
    if strict:
        for kw in LOW_QUALITY_KEYWORDS:
            if kw in text:
                return 100.0

    return penalty


def _drop_or_penalize(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        pen = _quality_penalty(r, strict)
        if strict and pen >= 100.0:
            continue
        r = r.copy()
        r["quality_penalty"] = pen
        rows.append(r)
    return pd.DataFrame(rows)


def _normalize_followers(df: pd.DataFrame) -> pd.Series:
    vals = df["followers"].fillna(0).astype(float).clip(lower=0) + 1
    logv = np.log10(vals)
    if logv.max() == logv.min():
        return pd.Series([50.0] * len(df), index=df.index)
    return 100 * (logv - logv.min()) / (logv.max() - logv.min())


def _percentile_er(df: pd.DataFrame) -> pd.Series:
    ers = df["er"].fillna(0).astype(float).clip(lower=0)
    ranks = ers.rank(method="average", pct=True)
    return (ranks * 100.0)


def _topic_match_scores(df: pd.DataFrame, topics: List[str]) -> pd.Series:
    if not topics:
        return pd.Series([50.0] * len(df), index=df.index)
    docs = (df["topic_of_influence"].fillna("") + "\n" + df["bio"].fillna(""))
    corpus = [" ".join(topics)] + docs.tolist()
    try:
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        X = tfidf.fit_transform(corpus)
        q = X[0]
        D = X[1:]
        sims = cosine_similarity(q, D).flatten()
        return pd.Series([s * 100.0 for s in sims], index=df.index)
    except Exception:
        topic_set = set([t.lower() for t in topics])
        scores = []
        for _, r in df.iterrows():
            text = (str(r.get("topic_of_influence") or "") + " " + str(r.get("bio") or "")).lower()
            hits = sum(1 for t in topic_set if t in text)
            scores.append(100.0 * hits / max(1, len(topic_set)))
        return pd.Series(scores, index=df.index)


def compute_scores(df: pd.DataFrame, topics: List[str], experience_years: int, seniority_hint: bool) -> pd.DataFrame:
    if df.empty:
        return df.assign(final_score=[])
    followers_score = _normalize_followers(df)
    engagement_score = _percentile_er(df)
    topic_match = _topic_match_scores(df, topics)
    exp_bonus = min(10.0, max(0, experience_years) * 0.8) if seniority_hint and experience_years > 0 else 0.0

    scores = 0.45 * followers_score + 0.35 * engagement_score + 0.20 * topic_match + exp_bonus
    out = df.copy()
    out["followers_score"] = followers_score
    out["engagement_score"] = engagement_score
    out["topic_match"] = topic_match
    out["final_score"] = scores - out.get("quality_penalty", 0)
    return out


def apply_filters(
    df: pd.DataFrame,
    country: str | None,
    platforms: Set[str] | None,
    topics: List[str] | None,
    follower_min: int | None,
    follower_max: int | None,
) -> pd.DataFrame:
    sel = df.copy()
    if country:
        sel = sel[sel["country"].astype(str).str.lower() == country.lower()]

    if platforms:
        platforms_norm = {p.lower() for p in platforms}
        sel = sel[sel["platform"].astype(str).str.lower().isin(platforms_norm)]

    if topics:
        mask = pd.Series(False, index=sel.index)
        for t in topics:
            t = t.lower()
            mask = mask | sel["topic_of_influence"].astype(str).str.lower().str.contains(re.escape(t), na=False)
            mask = mask | sel["bio"].astype(str).str.lower().str.contains(re.escape(t), na=False)
        sel = sel[mask] if mask.any() else sel

    if follower_min is not None:
        sel = sel[(sel["followers"].fillna(0).astype(float) >= float(follower_min))]
    if follower_max is not None:
        sel = sel[(sel["followers"].fillna(0).astype(float) <= float(follower_max))]

    if not sel.empty:
        sel = sel.sort_values(["platform", "handle", "followers"], ascending=[True, True, False])
        sel = sel.drop_duplicates(subset=["platform", "handle"], keep="first")
    return sel


def quality_filter(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    return _drop_or_penalize(df, strict)






