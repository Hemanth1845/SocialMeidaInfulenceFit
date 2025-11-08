# frontend/app.py
import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Influencer Fit Agent", layout="wide")

st.markdown("""
<style>
body { background-color: #0E1117; color: #EDEDED; }
.influencer-card {
  background-color: #1C1F26; border-radius: 16px; padding: 20px; margin: 18px 0;
  border: 1px solid #2E3440; box-shadow: 0 4px 12px rgba(0,0,0,0.45);
}
.fit-score { float: right; color: #8A6DF1; font-size: 26px; font-weight: 700; }
.outreach-box { background-color: #2E2E3E; border-radius: 10px; padding: 12px 14px; margin-top: 12px;
  color:#E0E0E0; border-left: 4px solid #9B6BFF; }
.small-muted { color:#A7A7A7; font-size:13px; }
/* Style for the new 'Not Interested' button */
div[data-testid*="stButton"] > button[kind="secondary"] {
    color: #FFB0B0;
    border-color: #FFB0B0;
}
div[data-testid*="stButton"] > button[kind="secondary"]:hover {
    color: #FFFFFF;
    border-color: #FF4B4B;
    background-color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

st.title("✨ Influencer Fit Agent")
st.markdown("Find and contact the best-fit creators for your campaign using AI.")

# --- Utility Functions ---
@st.cache_data(ttl=600)
def call_backend(path: str, method="GET", payload=None):
    try:
        url = f"{BACKEND_URL}/{path}"
        if method == "GET":
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=60)

        if r.status_code == 200:
            return r.json()

        try:
            detail = r.json().get("detail", r.text)
            st.error(f"Error {r.status_code}: {detail}")
        except:
            st.error(f"Error {r.status_code}: {r.text}")

    except requests.exceptions.ConnectionError:
        st.error(f"Backend unavailable. Is the server running at {BACKEND_URL}?")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None


def parse_compact_number(s: str | None):
    if not s:
        return None
    s = str(s).strip().lower()
    m = pd.Series([s]).str.extract(r"^([0-9]*\.?[0-9]+)\s*([kmb])?$", expand=True)
    if pd.isna(m.iloc[0, 0]):
        try:
            return int(float(s))
        except Exception:
            return None
    num = float(m.iloc[0, 0])
    suf = (m.iloc[0, 1] or "").lower()
    mul = 1
    if suf == "k":
        mul = 1_000
    elif suf == "m":
        mul = 1_000_000
    elif suf == "b":
        mul = 1_000_000_000
    return int(num * mul)


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = {}
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "ignore_list" not in st.session_state:
    st.session_state.ignore_list = set()

# --- UI Controls ---
colL, colR = st.columns([3, 2])
with colL:
    brief = st.text_area(
        "📋 Your campaign brief:",
        placeholder="e.g., Find 5 influencers for a fashion brand in India with > 100k followers",
    )

with colR:
    years = st.slider("Minimum Experience (years)", 0, 20, 0,
                      help="Boosts score for experienced creators")

    loc_resp = call_backend("locations")
    locations = (loc_resp or {}).get("locations", [])
    if not locations and loc_resp is not None:
        st.warning("No locations found. Check 'backend/data' directory.")

    location = st.selectbox("🌍 Location", options=["(auto)"] + locations, index=0)
    topk = st.number_input("Top K influencers", min_value=1, max_value=50, value=10, step=1)
    strict_quality = st.toggle("🧹 Remove fake/inactive (Strict)", value=False,
                               help="Aggressively filters out low-quality or suspicious profiles.")

st.markdown("---")
st.subheader("📈 Followers Range")
c1, c2 = st.columns(2)
with c1:
    fmin_text = st.text_input("Min Followers (e.g., 10k, 1m)", value="")
with c2:
    fmax_text = st.text_input("Max Followers (e.g., 500k, 5m)", value="")

st.subheader("🎯 Choose Social Media Platforms")
plat_resp = call_backend("platforms")
plat_opts = (plat_resp or {}).get("platforms", ["instagram", "tiktok", "youtube", "threads"])

platforms = st.multiselect(
    "Select one or more platforms:",
    options=plat_opts, default=[]
)

st.markdown("---")

if st.button("📢 Find Influencers", type="primary", use_container_width=True):
    if not (brief or "").strip():
        st.warning("Please enter a campaign brief.")
    else:
        payload = {
            "brief": brief.strip(),
            "country": None if location == "(auto)" else location,
            "top_k": int(topk),
            "experience": int(years),
            "strict_quality": bool(strict_quality),
            "platforms": platforms,
            "follower_min": fmin_text.strip() or None,
            "follower_max": fmax_text.strip() or None,
        }
        with st.spinner("Finding best-fit influencers..."):
            result = call_backend("recommend", method="POST", payload=payload)

        if result and "recommendations" in result:
            recs = result["recommendations"]
            st.session_state.recommendations = recs
            st.success(f"✅ Found {len(recs)} matching influencers!")
            if recs:
                st.balloons()
        else:
            st.session_state.recommendations = []
            st.error("Failed to retrieve recommendations.")

# --- Display Results ---
recs = st.session_state.get("recommendations", [])

filtered_recs = [
    r for r in recs
    if r.get("handle") not in st.session_state.ignore_list
]

if filtered_recs:
    st.markdown("---")
    st.subheader("📊 Top Matches")

    df = pd.DataFrame(filtered_recs)
    show_cols = ["name", "handle", "platform", "followers", "er", "country", "topic_of_influence", "final_score"]
    show_cols = [c for c in show_cols if c in df.columns]

    df_display = df[show_cols].copy()
    df_display["followers"] = df_display["followers"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    df_display["er"] = df_display["er"].apply(lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "N/A")
    df_display["final_score"] = df_display["final_score"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "N/A")

    st.dataframe(df_display, use_container_width=True)

    @st.cache_data
    def convert_df_to_csv(df_in):
        return df_in.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df[show_cols])
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="influencer_recommendations.csv",
        mime="text/csv",
    )

    st.markdown("### 🤖 AI-Powered Influencer Cards")
    st.markdown("---")

    for i, r in enumerate(filtered_recs):
        handle_key = (r.get("handle") or f"h_{i}")
        msg_key = f"msg_{handle_key}"
        open_key = f"open_{msg_key}"

        if open_key not in st.session_state:
            st.session_state[open_key] = False

        followers = int(float(r.get("followers") or 0))
        er_pct = round((float(r.get("er") or 0) * 100.0), 2)

        # === Updated Card with White Username ===
        st.markdown(f"""
        <div class="influencer-card" style="color:white;">
            <div style="display:flex;align-items:center;gap:16px;">
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="84">
                <div>
                    <h4 style="margin:0; color:white;">
                        {r.get('name','Unknown')} 
                        <span style="color:white;">{r.get('handle','')}</span>
                    </h4>
                    <p class="small-muted" style="margin:0;">{str(r.get('platform','')).capitalize()} • {str(r.get('country','')).capitalize()}</p>
                    <p style="margin:0;">👥 <b>{followers:,}</b> followers &nbsp; 💬 <b>{er_pct}%</b> engagement</p>
                    <p class="small-muted" style="margin-top:4px;">🎯 Topic: {r.get('topic_of_influence','N/A')}</p>
                </div>
                <div style="margin-left:auto;text-align:right;">
                    <div class="small-muted">Fit Score</div>
                    <div class="fit-score">{round(float(r.get('final_score') or 0), 2)}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Card Controls ---
        col_m, col_b, col_i = st.columns([2, 2, 8])

        with col_m:
            if st.button("💬 Message", key=f"btn_{msg_key}"):
                st.session_state[open_key] = not st.session_state[open_key]

        with col_b:
            with st.expander("View Bio"):
                bio_text = r.get('bio', 'N/A')
                st.markdown(f"_{bio_text if bio_text.strip() else 'No bio available.'}_")

        with col_i:
            if st.button("Not Interested", key=f"ignore_{msg_key}", type="secondary"):
                st.session_state.ignore_list.add(r.get('handle'))
                st.rerun()

        if st.session_state.get(open_key):
            msg_val = st.session_state.messages.get(msg_key, "")

            new_txt = st.text_area("Type a message or generate with AI:",
                                   value=msg_val, key=f"ta_{msg_key}", height=100)

            if new_txt != msg_val:
                st.session_state.messages[msg_key] = new_txt

            col_save, col_gen = st.columns(2)
            with col_save:
                if st.button("Save Message", key=f"save_{msg_key}"):
                    st.session_state.messages[msg_key] = new_txt
                    st.success("Message saved!")

            with col_gen:
                if st.button("🤖 Generate with AI", key=f"gen_{msg_key}"):
                    if not brief.strip():
                        st.warning("Please enter a campaign brief above.")
                    else:
                        with st.spinner("Generating AI message..."):
                            ai_payload = {"brief": brief, "profile": r}
                            resp = call_backend("generate_outreach", method="POST", payload=ai_payload)

                            if resp and resp.get("message"):
                                st.session_state.messages[msg_key] = resp["message"]
                                st.rerun()
                            else:
                                st.error("Failed to generate AI message.")

        current = st.session_state.messages.get(msg_key, "").strip()
        if current:
            st.markdown(
                f"""<div class="outreach-box">📝 <b>Outreach Message:</b><br>{current.replace("\\n", "<br>")}</div>""",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
