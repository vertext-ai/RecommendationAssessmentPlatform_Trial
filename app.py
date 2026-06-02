from __future__ import annotations

import pandas as pd
import streamlit as st

# ================== 1. Config ==================
TOP_K = 10

TABLE_LIKELIST = "data_lab_test.recommender.RECOMMENDATION_ASSESSMENT_PLATFORM_LIKELIST"
TABLE_COMMENTS = "data_lab_test.recommender.RECOMMENDATION_ASSESSMENT_PLATFORM_COMMENTS"

# Logic 2 variant options — order must match the radio widget index
LOGIC2_VARIANTS = [
    {
        "label":       "Split Hammers, Punches and Chisels",
        "description": "Hammers vs Punches & Chisels sub-categories",
        "same_csv":    "recs_2_set1_same.csv",
        "diff_csv":    "recs_2_set1_different.csv",
        "same_idx":    3,   # DB logic_index for this section's same file
        "diff_idx":    4,
    },
    {
        "label":       "Split Blue-Point Tools",
        "description": "Blue-Point vs non-Blue-Point within each category",
        "same_csv":    "recs_2_set2_same.csv",
        "diff_csv":    "recs_2_set2_different.csv",
        "same_idx":    5,
        "diff_idx":    6,
    },
    {
        "label":       "Split Drive Tools",
        "description": "Socket & Drive Tools sub-categories by Entity Name (6)",
        "same_csv":    "recs_2_set3_same.csv",
        "diff_csv":    "recs_2_set3_different.csv",
        "same_idx":    7,
        "diff_idx":    8,
    },
]

st.set_page_config(page_title="SEP Recommendation Assessment", page_icon="🧩", layout="wide")

conn = st.connection("snowflake")

# ================== 2. Styling ==================
st.markdown("""
<style>
hr.section-sep { border: none; border-top: 1px solid rgba(0,0,0,.1); margin: 2rem 0; }
.center-title { text-align: center; margin-bottom: 1rem; }
.stButton button { width: 100%; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ================== 3. DB helpers ==================

def save_feedback(queried_sku: str, logic_index: int, if_like: int):
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

DATA_DIR = "data"

@st.cache_data(show_spinner=False)
def load_recs_df(csv_name: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{csv_name}"
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def get_imgix_url(sku: str) -> str:
    return f"https://snap-on-products-hr.imgix.net/{sku.strip()}.jpg?w=450&dpr=2&auto=format&fit=max&q=25"

def render_grid(rows: pd.DataFrame):
    items = rows.to_dict("records")
    for i in range(0, len(items), 5):
        chunk = items[i:i+5]
        cols = st.columns(5)
        for j, r in enumerate(chunk):
            rec_sku = r["rec_sku"]
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="width:100%;aspect-ratio:4/3;overflow:hidden;
                                display:flex;align-items:center;justify-content:center;
                                background:#fff;border-radius:8px;
                                box-shadow:0 2px 8px rgba(0,0,0,.07)">
                      <img src="{get_imgix_url(rec_sku)}"
                           style="width:100%;height:100%;object-fit:contain" />
                    </div>
                    <div style="text-align:center;margin-top:5px;font-size:0.85rem">
                      <a href="https://sep.snapon.com/product/{rec_sku}" target="_blank">
                        <b>{rec_sku}</b>
                      </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def render_section(csv_name: str, logic_index: int, section_label: str,
                   description: str, query_sku: str, display_k: int):
    """Render one same/different section with its grid and feedback controls."""
    st.subheader(section_label)
    st.caption(description)

    df = load_recs_df(csv_name)
    if df.empty:
        st.info(f"No data found ({csv_name}).")
        return

    recs = df[df["item_sku"].str.upper() == query_sku.upper()].head(display_k)
    if recs.empty:
        st.warning("No results found for this SKU in this section.")
        return

    render_grid(recs)

    with st.container():
        st.markdown(f"**Assessment — {section_label} (Index {logic_index})**")
        c1, c2, _ = st.columns([1, 1, 8])
        if c1.button("👍", key=f"like_{logic_index}"):
            save_feedback(query_sku, logic_index, 1)
        if c2.button("👎", key=f"dis_{logic_index}"):
            save_feedback(query_sku, logic_index, 0)
        comment_text = st.text_area(
            "Comments for this section:",
            key=f"comm_{logic_index}",
            height=80,
            placeholder="Why is this section good or bad?"
        )
        if st.button("Submit Section Comment", key=f"btn_{logic_index}"):
            save_comment(query_sku, logic_index, comment_text)

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)


# ================== 4. Main UI ==================

st.title("SEP Recommendation Assessment Platform")
st.caption("Provide feedback for each logic section. Images are scaled to 3/4 size.")

with st.form("search_form"):
    sku_input = st.text_input("Enter Item SKU", placeholder="e.g., A123456").strip().upper()
    submitted = st.form_submit_button("Search & Assess", use_container_width=True)

if submitted:
    st.session_state["current_sku"] = sku_input

if "show_top5" not in st.session_state:
    st.session_state["show_top5"] = False

btn_label = "✅ Showing Top 5  |  Click to Show All 10" if st.session_state["show_top5"] \
            else "Show Top 5 / 10 Recommendations"
if st.button(btn_label, use_container_width=True):
    st.session_state["show_top5"] = not st.session_state["show_top5"]
    st.rerun()

display_k = 5 if st.session_state["show_top5"] else TOP_K
query_sku = st.session_state.get("current_sku")

if query_sku:
    st.markdown(f"<h2 class='center-title'>Assessing SKU: {query_sku}</h2>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1.2, 1, 1.2])
    with mid:
        st.image(get_imgix_url(query_sku), caption="Queried Item", use_container_width=True)

    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    # ── Logic 1 ──────────────────────────────────────────────────────────────
    st.header("Logic 1")
    render_section("recs_1_same.csv",      1, "Same Category",
                   "Apriori + ALS hybrid · same Entity Name (4) category",
                   query_sku, display_k)
    render_section("recs_1_different.csv", 2, "Different Category",
                   "Apriori + ALS hybrid · different Entity Name (4) category",
                   query_sku, display_k)

    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    # ── Logic 2 ──────────────────────────────────────────────────────────────
    st.header("Logic 2")

    selected_idx = st.radio(
        "Category split method:",
        options=list(range(len(LOGIC2_VARIANTS))),
        format_func=lambda i: LOGIC2_VARIANTS[i]["label"],
        index=0,       # default: Split Hammers, Punches and Chisels
        horizontal=True,
    )

    variant = LOGIC2_VARIANTS[selected_idx]

    render_section(variant["same_csv"], variant["same_idx"], "Same Category",
                   variant["description"] + " · same sub-category",
                   query_sku, display_k)
    render_section(variant["diff_csv"], variant["diff_idx"], "Different Category",
                   variant["description"] + " · different sub-category",
                   query_sku, display_k)

else:
    st.info("Please enter a SKU to begin evaluation.")
