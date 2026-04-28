from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

# ================== 1. Config & Snowflake Connection ==================
APP_DIR = Path(".")
DATA_DIR = APP_DIR / "data"
IMAGE_INDEX_CSV = DATA_DIR / "image_index.csv"
TOP_K = 10

# Database Tables
TABLE_LIKELIST = "data_lab_test.recommender.recommendation_assessment_platform_likelist"
TABLE_COMMENTS = "data_lab_test.recommender.recommendation_assessment_platform_comments"

# Logic Configuration
SECTIONS: List[Tuple[str, str]] = [
    ("1", "same"),
    ("1", "different"),
    ("2", "same"),
    ("2", "different"),
]
NAME_TEMPLATE = "recs_{logic}_{pline}.csv"

st.set_page_config(page_title="SEP Recommendation Platform", page_icon="🧩", layout="wide")

# Establish Snowflake Connection
conn = st.connection("snowflake")

# ================== 2. Styling (CSS) ==================
st.markdown("""
<style>
div[data-testid="column"] > div:has(img) {
  padding: 10px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,.05);
  background-color: white; text-align: center; margin-bottom: 10px;
}
hr.section-sep { border: none; border-top: 1px solid rgba(0,0,0,.1); margin: 2rem 0; }
.center-title { text-align: center; margin-bottom: 1rem; }
.stButton button { width: 100%; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ================== 3. Data Actions & Loaders ==================

def save_feedback(queried_sku: str, rec_sku: str, is_like: int):
    """Saves Like (1) or Dislike (0) to Snowflake using '?' placeholder."""
    query = f"""
    INSERT INTO {TABLE_LIKELIST} (queried_sku, recommended_sku, "LIKE")
    VALUES (?, ?, ?)
    """
    try:
        conn.cursor().execute(query, (queried_sku, rec_sku, is_like))
        st.toast(f"Feedback recorded for {rec_sku}!", icon="✅")
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

def save_comment(queried_sku: str, logic_index: str, comment_text: str):
    """Saves Logic specific comments to Snowflake."""
    if not comment_text.strip():
        st.warning("Comment cannot be empty.")
        return
    
    query = f"""
    INSERT INTO {TABLE_COMMENTS} (queried_sku, logic_index, comments)
    VALUES (?, ?, ?)
    """
    try:
        idx = int(logic_index)
        conn.cursor().execute(query, (queried_sku, idx, comment_text))
        st.success(f"Comments for Logic {logic_index} submitted!")
    except Exception as e:
        st.error(f"Failed to save comment: {e}")

@st.cache_data(show_spinner=False)
def load_recs_df(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_image_index(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    idx = pd.read_csv(path)
    idx.columns = [c.strip().lower() for c in idx.columns]
    if "sku" not in idx.columns: return None
    idx["sku"] = idx["sku"].astype(str).str.strip().str.upper()
    return idx

image_index = load_image_index(IMAGE_INDEX_CSV)

def get_imgix_url(sku: str) -> str:
    """Generates Imgix URL at 3/4 size (450px)."""
    return f"https://snap-on-products-hr.imgix.net/{sku.strip()}.jpg?w=450&dpr=2&auto=format&fit=max&q=25"

def render_grid(rows: pd.DataFrame, query_sku: str, logic: str, pline: str):
    """Renders grid with 3/4 sized images and assessment buttons."""
    cols_per_row = 5
    items = rows.to_dict("records")
    for i in range(0, len(items), cols_per_row):
        chunk = items[i:i+cols_per_row]
        cols = st.columns(len(chunk))
        for c, r in zip(cols, chunk):
            rec_sku = r["rec_sku"]
            with c:
                # Center and scale image to 3/4 of the column width
                _, img_col, _ = st.columns([0.125, 0.75, 0.125])
                with img_col:
                    st.image(get_imgix_url(rec_sku), use_container_width=True)
                
                url = f"https://sep.snapon.com/product/{rec_sku}"
                st.markdown(f"**[{rec_sku}]({url})**")
                
                b1, b2 = st.columns(2)
                if b1.button("👍", key=f"like_{logic}_{pline}_{rec_sku}"):
                    save_feedback(query_sku, rec_sku, 1)
                if b2.button("👎", key=f"dis_{logic}_{pline}_{rec_sku}"):
                    save_feedback(query_sku, rec_sku, 0)

# ================== 4. Main Interface ==================

st.title("SEP Recommendation Assessment Platform")
st.caption("Perform evaluation on Logic 1 and Logic 2. All feedback is recorded for model refinement.")

with st.form("search_form"):
    sku_input = st.text_input("Enter Query SKU", placeholder="e.g., A123456").strip().upper()
    submitted = st.form_submit_button("Start Assessment", use_container_width=True)

if submitted:
    st.session_state["current_sku"] = sku_input

query_sku = st.session_state.get("current_sku")

if query_sku:
    st.markdown(f"<h2 class='center-title'>Assessing SKU: {query_sku}</h2>", unsafe_allow_html=True)
    
    # Scale main image similarly
    _, mid, _ = st.columns([1.2, 1, 1.2]) 
    with mid:
        st.image(get_imgix_url(query_sku), caption="Queried Item Image", use_container_width=True)
    
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    for logic_idx in ["1", "2"]:
        st.header(f"Logic {logic_idx} Results")
        
        for pline in ["same", "different"]:
            csv_path = DATA_DIR / NAME_TEMPLATE.format(logic=logic_idx, pline=pline)
            st.subheader(f"{pline.capitalize()} Product Line")
            
            df = load_recs_df(csv_path)
            if df.empty:
                st.info("No data source found for this category.")
                continue

            recs = df[df["item_sku"].str.upper() == query_sku.upper()].head(TOP_K)
            
            if recs.empty:
                st.warning(f"No recommendations found for {query_sku}.")
            else:
                render_grid(recs, query_sku, logic_idx, pline)

        st.markdown(f"#### 💬 Overall Comments for Logic {logic_idx}")
        comment_val = st.text_area(f"How accurate is Logic {logic_idx}? Any feedback?", key=f"text_{logic_idx}", height=100)
        if st.button(f"Submit Logic {logic_idx} Comment", key=f"btn_{logic_idx}"):
            save_comment(query_sku, logic_idx, comment_val)
        
        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
else:
    st.info("Please enter a SKU and click Search to begin.")