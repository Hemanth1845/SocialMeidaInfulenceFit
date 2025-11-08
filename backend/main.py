# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Set, Dict, Any
from dotenv import load_dotenv

from core.data_loader import load_all, available_locations, platforms_present
from core.scoring import apply_filters, quality_filter, compute_scores
from core.nlp import parse_brief
from core.outreach import build_outreach

import pandas as pd
import numpy as np

load_dotenv()

app = FastAPI(
    title="Influencer Fit Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    brief: str
    country: Optional[str] = None
    top_k: int = 5
    experience: int = 0
    strict_quality: bool = False
    platforms: Optional[List[str]] = None
    follower_min: Optional[str | float | int] = None
    follower_max: Optional[str | float | int] = None


class OutreachRequest(BaseModel):
    brief: str
    profile: Dict[str, Any]


@app.get("/")
def root():
    return {"status": "ok", "message": "Influencer Fit Agent Backend Running"}


@app.get("/locations")
def get_locations():
    try:
        return {"locations": available_locations()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/platforms")
def get_platforms():
    try:
        return {"platforms": sorted(list(platforms_present() or set()))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        brief = (req.brief or "").strip()
 
        if not brief:
            raise HTTPException(status_code=400, detail="Brief is required")

        parsed = parse_brief(
            brief,
            ui_overrides={
                "location": req.country,
                "top_k": req.top_k,
                "experience": req.experience,
                "strict_quality": req.strict_quality,
                "platforms": req.platforms or [],
                "follower_min": req.follower_min,
                "follower_max": req.follower_max,
            },
        )

        df = load_all()
        if df.empty:
            return {"recommendations": []}

       
        sel = apply_filters(
            df,
            parsed.country,
            parsed.platforms,
            parsed.topics,
            parsed.follower_min,
            parsed.follower_max,
        )
        sel = quality_filter(sel, parsed.strict_quality)
        if sel.empty:
            return {"recommendations": []}

        scored = compute_scores(
            sel,
            parsed.topics,
            parsed.experience_years,
            parsed.seniority_hint,
        )

        if scored.empty:
            return {"recommendations": []}

        top = scored.sort_values("final_score", ascending=False).head(parsed.top_k)

       
        top = top.replace([np.inf, -np.inf], np.nan)
        top = top.where(pd.notnull(top), None)

        return {"recommendations": top.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/generate_outreach")
def generate_outreach(req: OutreachRequest):
    try:
        if not req.brief:
            raise HTTPException(status_code=400, detail="Brief is required")
        if not req.profile:
            raise HTTPException(status_code=400, detail="Profile is required")
            
      
        message = build_outreach(req.profile, req.brief)
        
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

