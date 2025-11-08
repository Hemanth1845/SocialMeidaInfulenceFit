# core/outreach.py
# Kept for future AI outreach; not used by current manual messaging flow.
import os
import textwrap
from typing import Dict, Optional
import requests

def _env(key: str, default: str | None = None) -> Optional[str]:
    return os.getenv(key, default)

def _templated(profile: Dict, brief: str) -> str:
    name = profile.get("name") or profile.get("handle") or "there"
    handle = profile.get("handle") or ""
    platform = profile.get("platform") or ""
    topic = profile.get("topic_of_influence") or "your niche"
    msg = (
        f"Hi {name} ({handle})! We love your {platform} content around {topic}. "
        f"We're planning a campaign: {brief[:180]}... Would you be open to a quick collab chat? "
        f"We think your voice fits our audience. If interested, reply here and we’ll share details. Thanks!"
    )
    return msg[:450]

def build_outreach(profile: Dict, brief: str) -> str:
    # Current UI uses manual per-influencer messaging; return empty by default.
    # You can switch to Gemini by setting GEMINI_API_KEY and calling this where needed.
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        return ""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        prompt = textwrap.dedent(f"""
        Write a concise (<= 450 chars) influencer outreach.
        Creator:
        - Name: {profile.get('name','')}
        - Handle: {profile.get('handle','')}
        - Platform: {profile.get('platform','')}
        - Country: {profile.get('country','')}
        - Topic: {profile.get('topic_of_influence','')}
        - Bio: {profile.get('bio','')[:200]}
        Brief:
        {brief}
        """)
        payload = {"contents":[{"role":"user","parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.8,"maxOutputTokens":200}}
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        data = r.json()
        text = data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text")
        return (text or "").strip()[:450]
    except Exception:
        return _templated(profile, brief)
