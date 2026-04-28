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

# Database Table Names
TABLE_LIKELIST = "data_lab_test.recommender.RECOMMENDATION_ASSESSMENT_PLATFORM_LIKELIST"
TABLE_COMMENTS = "data_lab_test.recommender.RECOMMENDATION_ASSESSMENT_PLATFORM_COMMENTS"

# Logic Section Configuration
# Index 1: Logic 1 Same, Index 2: Logic 1 Different, Index 3: Logic 2 Same, Index 4: Logic 2 Different
SECTIONS: List[Tuple[str, str]] = [
    ("1", "same"),
    ("1", "different"),
    ("2", "same"),
    ("2", "different"),
]
NAME_TEMPLATE = "recs_{logic}_{pline}.csv"

st.set_page_config(page_title="SEP Recommendation Assessment", page_icon="🧩", layout="wide")

# Initialize Snowflake connection
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
/* Container for section-level controls */
.section-control-box {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #007bff;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================== 3. Database Actions & Loaders ==================

def save_feedback(queried_sku: str, logic_index: int, if_like: int):
    """Saves section-level like/dislike (1-4 index)."""
    query = f"""
    INSERT INTO {TABLE_LIKELIST} (ITEM_SKU, LOGIC_INDEX, IF_LIKE)
    VALUES (?, ?, ?)
    """
    try:
        conn.cursor().execute(query, (queried_sku, logic_index, if_like))
        st.toast(f"Section {logic_index} feedback recorded!", icon="✅")
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

def save_comment(queried_sku: str, logic_index: int, comment_text: str):
    """Saves section-level comment (1-4 index)."""
    if not comment_text.strip():
        st.warning("Please enter a comment before submitting.")
        return
    
    query = f"""
    INSERT INTO {TABLE_COMMENTS} (QUERIED_SKU, LOGIC_INDEX, COMMENTS)
    VALUES (?, ?, ?)
    """
    try:
        conn.cursor().execute(query, (queried_sku, logic_index, comment_text))
        st.success(f"Comment for Section {logic_index} submitted!")
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

def get_imgix_url(sku: str) -> str:
    """Generates 3/4 sized image URL."""
    return f"https://snap-on-products-hr.imgix.net/{sku.strip()}.jpg?w=450&dpr=2&auto=format&fit=max&q=25"

def render_grid(rows: pd.DataFrame):
    """Renders images in a grid."""
    cols_per_row = 5
    items = rows.to_dict("records")
    for i in range(0, len(items), cols_per_row):
        chunk = items[i:i+cols_per_row]
        cols = st.columns(len(chunk))
        for c, r in zip(cols, chunk):
            rec_sku = r["rec_sku"]
            with c:
                _, img_col, _ = st.columns([0.125, 0.75, 0.125])
                with img_col:
                    st.image(get_imgix_url(rec_sku), use_container_width=True)
                url = f"https://sep.snapon.com/product/{rec_sku}"
                st.markdown(f"**[{rec_sku}]({url})**")

# ================== 4. Main UI Logic ==================

st.title("SEP Recommendation Assessment Platform")
st.caption("Provide feedback for each logic section. Images are scaled to 3/4 size.")

with st.form("search_form"):
    sku_input = st.text_input("Enter Item SKU", placeholder="e.g., A123456").strip().upper()
    submitted = st.form_submit_button("Search & Assess", use_container_width=True)

if submitted:
    st.session_state["current_sku"] = sku_input

query_sku = st.session_state.get("current_sku")

if query_sku:
    st.markdown(f"<h2 class='center-title'>Assessing SKU: {query_sku}</h2>", unsafe_allow_html=True)
    
    # Main item image
    _, mid, _ = st.columns([1.2, 1, 1.2]) 
    with mid:
        st.image(get_imgix_url(query_sku), caption="Queried Item", use_container_width=True)
    
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    # Counter for sections 1 to 4
    section_idx = 1

    for logic_val in ["1", "2"]:
        st.header(f"Logic {logic_val}")
        
        for pline in ["same", "different"]:
            csv_path = DATA_DIR / NAME_TEMPLATE.format(logic=logic_val, pline=pline)
            st.subheader(f"{pline.capitalize()} Product Line")
            
            df = load_recs_df(csv_path)
            if df.empty:
                st.info(f"No data for Logic {logic_val} - {pline}.")
                section_idx += 1
                continue

            recs = df[df["item_sku"].str.upper() == query_sku.upper()].head(TOP_K)
            
            if recs.empty:
                st.warning(f"No results found in this section.")
            else:
                # 1. Show the recommended items
                render_grid(recs)
                
                # 2. Section Feedback Area (Likes & Individual Comment)
                with st.container():
                    st.markdown(f"**Assessment for Logic {logic_val} - {pline.capitalize()} (Index {section_idx})**")
                    
                    # Thumbs Buttons
                    c1, c2, _ = st.columns([1, 1, 8])
                    if c1.button("👍", key=f"like_{section_idx}"):
                        save_feedback(query_sku, section_idx, 1)
                    if c2.button("👎", key=f"dis_{section_idx}"):
                        save_feedback(query_sku, section_idx, 0)
                    
                    # Individual Comment Box for THIS section
                    comment_text = st.text_area(
                        f"Comments for this specific section:", 
                        key=f"comm_{section_idx}", 
                        height=80,
                        placeholder="Why is this section good or bad?"
                    )
                    if st.button("Submit Section Comment", key=f"btn_{section_idx}"):
                        save_comment(query_sku, section_idx, comment_text)
            
            st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)
            section_idx += 1
            
        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

else:
    st.info("Please enter a SKU to begin evaluation.")