# Influencer Fit Agent

Production-ready full-stack app to find and rank best-fit influencers and generate outreach.

## Quickstart

1) Install dependencies

```
pip install -r requirements.txt
```

2) Set environment (create `.env` from `.env.example`)

- Set `GEMINI_API_KEY` if you want AI-written outreach (otherwise fallback template is used)
- Optionally set `BACKEND_BASE` for the frontend to reach the API (e.g., `http://127.0.0.1:8000`)
- Optionally set `DATA_DIR` (defaults to `./data`)

3) Run backend (FastAPI)

```
uvicorn backend.main:app --reload
```

4) Run frontend (Streamlit)

```
streamlit run frontend/app.py
```

## Data Layout

Place CSVs under `./data/<country>/`. Any `*.csv` is discovered. Typical files may include
`instagram_data_<country>.csv`, `tiktok_data_<country>.csv`, `youtube_data_<country>.csv`, `threads_data_<country>.csv`.
Columns are unified automatically.

## Notes

- No hardcoded backend URLs; the frontend uses an adapter that reads `BACKEND_BASE` or defaults to same-origin `/api/*`.
- No fixed CSV names; loader discovers recursively.
- Secrets are never logged; the Gemini key is read from env only.
