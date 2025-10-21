# import os
# import glob
# from pathlib import Path
# from typing import List, Optional

# import pandas as pd
# from PIL import Image
# import streamlit as st

# # ================== Config ==================
# DATA_DIR = Path("data")
# IMAGES_DIR = Path("images")
# IMAGE_INDEX_CSV = DATA_DIR / "image_index.csv"   # optional: sku,image_path
# TOP_K = 10

# # File naming template for 3 (A/B/C) × 2 (same/different) = 6 CSVs
# NAME_TEMPLATE = "recs_{logic}_{pline}.csv"  # logic in a/b/c, pline in same/different

# st.set_page_config(page_title="SKU Recommender (Multi-Logic)", page_icon="🧩", layout="wide")

# # Simple styling for image cards
# st.markdown("""
# <style>
# div[data-testid="column"] > div:has(img) {
#   padding: 8px; border-radius: 16px; box-shadow: 0 2px 10px rgba(0,0,0,.06);
# }
# </style>
# """, unsafe_allow_html=True)

# # ================== Helpers (cached) ==================
# @st.cache_data(show_spinner=False)
# def load_recs_df(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df.columns = [c.strip().lower() for c in df.columns]
#     for c in df.columns:
#         df[c] = df[c].astype(str).str.strip()
#     return df

# @st.cache_data(show_spinner=False)
# def load_image_index(path: Path) -> Optional[pd.DataFrame]:
#     if not path.exists():
#         return None
#     idx = pd.read_csv(path)
#     idx.columns = [c.strip().lower() for c in idx.columns]
#     assert "sku" in idx.columns and "image_path" in idx.columns, \
#         "image_index.csv must contain columns: sku,image_path"
#     idx["sku"] = idx["sku"].astype(str).str.strip()
#     idx["image_path"] = idx["image_path"].astype(str).str.strip()
#     return idx

# image_index = load_image_index(IMAGE_INDEX_CSV)

# def parse_rec_list_from_row(s: str) -> List[str]:
#     if pd.isna(s) or not s:
#         return []
#     parts = [p.strip() for p in str(s).replace("，", ",").replace("|", ",").split(",")]
#     return [p for p in parts if p]

# def get_recommendations(recs: pd.DataFrame, sku: str, top_k: int = TOP_K) -> List[str]:
#     cols = set(recs.columns)

#     # Wide table: item_sku + rec_skus
#     if "rec_skus" in cols:
#         row = recs.loc[recs["item_sku"] == sku]
#         if row.empty:
#             return []
#         rec_list = parse_rec_list_from_row(row.iloc[0]["rec_skus"])
#         return list(dict.fromkeys(rec_list))[:top_k]

#     # Long table: item_sku, rec_sku[, score]
#     if "rec_sku" in cols:
#         sub = recs.loc[recs["item_sku"] == sku].copy()
#         if sub.empty:
#             return []
#         if "score" in sub.columns:
#             sub["score_num"] = pd.to_numeric(sub["score"], errors="coerce")
#             sub = sub.sort_values("score_num", ascending=False)
#         rec_list = sub["rec_sku"].astype(str).str.strip().tolist()
#         return list(dict.fromkeys(rec_list))[:top_k]

#     # Fallback: infer columns
#     possible_item = [c for c in recs.columns if "item" in c and "sku" in c]
#     possible_rec = [c for c in recs.columns if "rec" in c and "sku" in c]
#     if possible_item and possible_rec:
#         sub = recs.loc[recs[possible_item[0]] == sku, possible_rec[0]]
#         rec_list = sub.astype(str).str.strip().tolist()
#         return list(dict.fromkeys(rec_list))[:top_k]

#     st.warning("Unrecognized schema. Expected (item_sku, rec_sku[, score]) or (item_sku, rec_skus).")
#     return []

# def find_image_by_sku(sku: str) -> Optional[Path]:
#     # Prefer index CSV
#     if image_index is not None:
#         hit = image_index.loc[image_index["sku"] == sku]
#         if not hit.empty:
#             p = Path(hit.iloc[0]["image_path"])
#             if p.exists():
#                 return p
#             if not p.is_absolute():
#                 p1 = Path(p)
#                 p2 = IMAGES_DIR / p
#                 if p1.exists(): return p1
#                 if p2.exists(): return p2

#     # Conventional directory lookup
#     patterns = [f"{sku}.jpg", f"{sku}.jpeg", f"{sku}.png", f"{sku}.webp"]
#     for pat in patterns:
#         cand = IMAGES_DIR / pat
#         if cand.exists():
#             return cand

#     # Loose match by filename contains sku
#     globs = []
#     for ext in ("jpg", "jpeg", "png", "webp"):
#         globs.extend(glob.glob(str(IMAGES_DIR / f"*{sku}*.{ext}")))
#     if globs:
#         return Path(globs[0])

#     return None

# def load_image_safe(path: Path, max_size=(700, 700)) -> Optional[Image.Image]:
#     try:
#         img = Image.open(path)
#         img.thumbnail(max_size)
#         return img
#     except Exception as e:
#         st.debug(f"Failed to open image {path}: {e}")
#         return None

# def norm_sku(s: str) -> str:
#     return str(s).strip().upper()

# def csv_path_for(logic_letter: str, pline_key: str) -> Path:
#     fname = NAME_TEMPLATE.format(logic=logic_letter.lower(), pline=pline_key)
#     return DATA_DIR / fname

# def show_centered_image(img: Image.Image, caption: str):
#     """Render an image centered on the page."""
#     left, mid, right = st.columns([1, 2, 1])
#     with mid:
#         st.image(img, caption=caption, use_container_width=False)

# # ================== UI: Tabs + Product Line (no sidebar) ==================
# st.title("SEP SKU Recommender")
# st.caption("Choose Logic (A/B/C) and Product Line (Same/Different), enter an item_sku, and get up to 10 recommended SKUs with their images.")

# logic_tabs = st.tabs(["Logic A", "Logic B", "Logic C"])
# logic_letters = ["A", "B", "C"]

# for tab, logic in zip(logic_tabs, logic_letters):
#     with tab:
#         st.subheader(f"Logic {logic}")
#         pline_label = st.selectbox(
#             "Product Line",
#             options=["Same", "Different"],
#             index=0,
#             key=f"pline_{logic}"
#         )
#         pline_key = "same" if pline_label.lower() == "same" else "different"
#         current_csv = csv_path_for(logic, pline_key)

#         st.caption(f"Current data file: `{current_csv}`")
#         with st.form(f"form_{logic}_{pline_key}", clear_on_submit=False):
#             sku_input = st.text_input("Enter item_sku", value="", placeholder="e.g., A123456", key=f"sku_{logic}")
#             do_search = st.form_submit_button("Search", use_container_width=True)

#         if do_search:
#             query_sku = norm_sku(sku_input)
#             if not query_sku:
#                 st.warning("Please enter a valid item_sku.")
#                 st.stop()

#             if not current_csv.exists():
#                 st.error(f"Data file not found: {current_csv}. Please check the file name and path.")
#                 st.stop()

#             with st.spinner("Fetching recommendations..."):
#                 recs_df = load_recs_df(current_csv)

#                 # Normalize SKU columns before lookup
#                 _recs = recs_df.copy()
#                 if "item_sku" in _recs.columns:
#                     _recs["item_sku"] = _recs["item_sku"].astype(str).str.strip().str.upper()
#                 if "rec_sku" in _recs.columns:
#                     _recs["rec_sku"] = _recs["rec_sku"].astype(str).str.strip().str.upper()
#                 if "rec_skus" in _recs.columns:
#                     _recs["rec_skus"] = _recs["rec_skus"].astype(str)

#                 recs = get_recommendations(_recs, query_sku, TOP_K)

#             st.write(f"Query: **{query_sku}** (source: `{current_csv.name}`)")

#             # ===== NEW: show the queried item's image centered at the very top =====
#             item_img_path = find_image_by_sku(query_sku)
#             if item_img_path and item_img_path.exists():
#                 item_img = load_image_safe(item_img_path)
#                 if item_img is not None:
#                     st.markdown("#### Queried item")
#                     show_centered_image(item_img, caption=query_sku)
#                 else:
#                     st.warning("Queried item image appears to be corrupted.")
#             else:
#                 st.info("No image found for the queried item.")

#             # ======================================================================

#             if not recs:
#                 st.info("No recommendations found. Please verify the SKU exists or check the CSV schema.")
#                 st.stop()

#             st.write(f"Returned {len(recs)} recommendations (showing up to {TOP_K}).")

#             cols_per_row = 5
#             for i in range(0, len(recs), cols_per_row):
#                 row = recs[i:i+cols_per_row]
#                 cols = st.columns(len(row))
#                 for c, rsku in zip(cols, row):
#                     with c:
#                         p = find_image_by_sku(rsku)
#                         if p is not None and p.exists():
#                             img = load_image_safe(p)
#                             if img is not None:
#                                 st.image(img, caption=rsku, use_container_width=True)
#                             else:
#                                 st.error(f"{rsku}\n(Image corrupted)")
#                         else:
#                             st.warning(f"{rsku}\n(Image not found)")

#             out_df = pd.DataFrame({
#                 "logic": [logic]*len(recs),
#                 "product_line": [pline_key]*len(recs),
#                 "item_sku": [query_sku]*len(recs),
#                 "rec_sku": recs
#             })
#             st.download_button(
#                 "Download results as CSV",
#                 data=out_df.to_csv(index=False).encode("utf-8"),
#                 file_name=f"{logic}_{pline_key}_{query_sku}_recs.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )











from __future__ import annotations
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# ================== Config ==================
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
IMAGES_DIR = APP_DIR / "images"
IMAGE_INDEX_CSV = DATA_DIR / "image_index.csv"   # optional: sku,[name],image_path
# SIMILAR_CSV = DATA_DIR / "similar_products.csv"
TOP_K = 10

# We'll render 4 sections in this fixed order:
SECTIONS: List[Tuple[str, str]] = [
    ("1", "same"),
    ("1", "different"),
    ("2", "same"),
    ("2", "different"),
]
NAME_TEMPLATE = "recs_{logic}_{pline}.csv"  # logic in 1/2, pline in same/different

st.set_page_config(page_title="SKU Recommender (Single Page, Price Filters)", page_icon="🧩", layout="wide")

# ================== Styling ==================
st.markdown("""
<style>
/* card-like image containers */
div[data-testid="column"] > div:has(img) {
  padding: 10px; border-radius: 16px; box-shadow: 0 2px 10px rgba(0,0,0,.06);
}
hr.section-sep { border: none; border-top: 1px solid rgba(0,0,0,.08); margin: 1.25rem 0; }
.small-dim { opacity: .7; font-size: 0.9rem; }
.center-title { text-align: center; margin: .25rem 0 .5rem; }
</style>
""", unsafe_allow_html=True)

# ================== Data loaders (cached) ==================
@st.cache_data(show_spinner=False)
def load_recs_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize key columns as string
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # price numeric
    if "price" in df.columns:
        df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price_num"] = pd.NA
    # score numeric (optional)
    if "score" in df.columns:
        df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    else:
        df["score_num"] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def load_image_index(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    idx = pd.read_csv(path)
    idx.columns = [c.strip().lower() for c in idx.columns]
    # only require 'sku'; other columns optional
    if "sku" not in idx.columns:
        return None
    for c in idx.columns:
        idx[c] = idx[c].astype(str).str.strip()
    return idx

image_index = load_image_index(IMAGE_INDEX_CSV)

# ================== Helpers ==================
# --- unified image size for all SKUs ---
FIXED_IMG_SIZE = (600, 600)  # 可改为 (400,400) 等

def load_image_fixed(path: Path, box=FIXED_IMG_SIZE, bg=(255, 255, 255)):
    """
    Read image -> scale to fit within `box` (keep aspect) -> paste centered
    on a fixed-size canvas (letterbox, no crop). Returns a PIL.Image or None.
    """
    try:
        img = Image.open(path).convert("RGB")
        fitted = ImageOps.contain(img, box)  # keep aspect ratio, <= box
        canvas = Image.new("RGB", box, bg)
        offset = ((box[0] - fitted.width) // 2, (box[1] - fitted.height) // 2)
        canvas.paste(fitted, offset)
        return canvas
    except Exception as e:
        st.debug(f"Failed to open/standardize image {path}: {e}")
        return None


# @st.cache_data(show_spinner=False)
# def load_similar_df(path: Path) -> pd.DataFrame:
#     """Load similar_products.csv: item_sku, rec_sku, similarity_score"""
#     df = pd.read_csv(path)
#     df.columns = [c.strip().lower() for c in df.columns]
#     for c in df.columns:
#         df[c] = df[c].astype(str).str.strip()
#     if "similarity_score" in df.columns:
#         df["similarity_score_num"] = pd.to_numeric(df["similarity_score"], errors="coerce")
#     else:
#         df["similarity_score_num"] = pd.NA
#     return df

# def get_similar_for_item(df: pd.DataFrame, item_sku: str) -> pd.DataFrame:
#     """
#     Filter similar items for one item_sku.
#     Expect columns: item_sku, rec_sku, similarity_score_num
#     """
#     if not {"item_sku", "rec_sku"}.issubset(set(df.columns)):
#         st.warning("similar_products.csv must contain columns: item_sku, rec_sku, similarity_score.")
#         return pd.DataFrame(columns=["item_sku", "rec_sku", "similarity_score_num"])

#     _df = df.copy()
#     _df["item_sku"] = _df["item_sku"].str.upper()
#     _df["rec_sku"] = _df["rec_sku"].str.upper()

#     sub = _df.loc[_df["item_sku"] == item_sku].copy()
#     if sub.empty:
#         return sub

#     if "similarity_score_num" in sub.columns:
#         sub = sub.sort_values("similarity_score_num", ascending=False, na_position="last")

#     # 去重：同一 rec_sku 仅保留最高分
#     sub = sub.drop_duplicates(subset=["rec_sku"], keep="first")

#     return sub

# def render_grid_with_score(rows: pd.DataFrame):
#     cols_per_row = 5
#     items = rows.to_dict("records")
#     for i in range(0, len(items), cols_per_row):
#         chunk = items[i:i+cols_per_row]
#         cols = st.columns(len(chunk))
#         for c, r in zip(cols, chunk):
#             with c:
#                 sku = r["rec_sku"]
#                 score = r.get("similarity_score_num", None)
#                 url = f"https://sep.snapon.com/product/{sku}"

#                 # 统一生成“链接 + 分数”的文本（1 位小数）
#                 score_txt = f" — score {float(score):.1f}" if pd.notna(score) else ""
#                 link_line = f"[**{sku}**]({url}){score_txt}"

#                 p = find_image_by_sku(sku)
#                 if p and p.exists():
#                     img = load_image_fixed(p)
#                     if img is not None:
#                         st.image(img, use_container_width=True)
#                         st.markdown(link_line)
#                     else:
#                         st.error(f"{sku}\n(Image corrupted)")
#                         st.markdown(link_line)
#                 else:
#                     st.warning(f"{sku}\n(Image not found)")
#                     st.markdown(link_line)


def csv_path_for(logic_letter: str, pline_key: str) -> Path:
    return DATA_DIR / NAME_TEMPLATE.format(logic=logic_letter, pline=pline_key)

def norm_sku(s: str) -> str:
    return str(s).strip().upper()

def get_item_name(sku: str) -> Optional[str]:
    if image_index is not None and "name" in image_index.columns:
        hit = image_index.loc[image_index["sku"].str.upper() == sku]
        if not hit.empty:
            return hit.iloc[0]["name"]
    return None

def find_image_by_sku(sku: str) -> Optional[Path]:
    # Prefer index CSV if it has image_path
    if image_index is not None and "image_path" in image_index.columns:
        hit = image_index.loc[image_index["sku"].str.upper() == sku]
        if not hit.empty:
            p = Path(hit.iloc[0]["image_path"])
            # allow relative paths
            if not p.is_absolute():
                p1 = (APP_DIR / p)
                p2 = (IMAGES_DIR / p)
                if p1.exists(): return p1
                if p2.exists(): return p2
            if p.exists():
                return p

    # Fallback: by convention in images/
    for ext in ("jpg", "jpeg", "png", "webp"):
        cand = IMAGES_DIR / f"{sku}.{ext}"
        if cand.exists():
            return cand

    # Loose match
    for ext in ("jpg", "jpeg", "png", "webp"):
        globs = glob.glob(str(IMAGES_DIR / f"*{sku}*.{ext}"))
        if globs:
            return Path(globs[0])
    return None

def load_image_safe(path: Path, max_size=(720, 720)) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.thumbnail(max_size)
        return img
    except Exception:
        return None

def get_recs_for_item(df: pd.DataFrame, item_sku: str) -> pd.DataFrame:
    """
    Expect long-form: item_sku, rec_sku, price[, score]
    """
    if not {"item_sku", "rec_sku"}.issubset(set(df.columns)):
        st.warning("CSV must contain columns: item_sku, rec_sku, price[, score].")
        return pd.DataFrame(columns=["item_sku", "rec_sku", "price_num", "score_num"])

    # normalize to uppercase for join
    _df = df.copy()
    _df["item_sku"] = _df["item_sku"].str.upper()
    _df["rec_sku"] = _df["rec_sku"].str.upper()

    sub = _df.loc[_df["item_sku"] == item_sku].copy()
    if sub.empty:
        return sub

    # prefer higher score if provided; then de-duplicate by rec_sku
    if "score_num" in sub.columns:
        sub = sub.sort_values("score_num", ascending=False, na_position="last")
    # keep first occurrence per rec_sku
    sub = sub.drop_duplicates(subset=["rec_sku"], keep="first")

    return sub

def section_title(logic: str, pline: str) -> str:
    return f"Logic {logic} — {'Same' if pline=='same' else 'Different'} product line"

def price_box_key(logic: str, pline: str, end: str) -> str:
    return f"price_{logic}_{pline}_{end}"

def render_grid(rows: pd.DataFrame):
    cols_per_row = 5
    items = rows.to_dict("records")
    for i in range(0, len(items), cols_per_row):
        chunk = items[i:i+cols_per_row]
        cols = st.columns(len(chunk))
        for c, r in zip(cols, chunk):
            with c:
                sku = r["rec_sku"]
                price = r.get("price_num", None)
                url = f"https://sep.snapon.com/product/{sku}"

                p = find_image_by_sku(sku)
                if p and p.exists():
                    img = load_image_safe(p)
                    if img is not None:
                        # 显示图片（不再用 caption）
                        st.image(img, use_container_width=True)
                        # 在图片下方显示超链接 + 可选价格
                        if pd.notna(price):
                            st.markdown(f"[**{sku}**]({url}) — ${price:,.2f}")
                        else:
                            st.markdown(f"[**{sku}**]({url})")
                    else:
                        st.error(f"{sku}\n(Image corrupted)")
                        st.markdown(f"[**{sku}**]({url})")
                else:
                    st.warning(f"{sku}\n(Image not found)")
                    st.markdown(f"[**{sku}**]({url})")

# ================== UI ==================
st.title("SEP Recommender")
st.caption("Enter an item_sku once; below you'll see results for Logic 1/2 × Same/Different product lines. Use the price range in each section to filter results.")

# --- Search form (top) ---
with st.form("search", clear_on_submit=False):
    sku_input = st.text_input("Enter item_sku", value="", placeholder="e.g., A123456")
    submitted = st.form_submit_button("Search", use_container_width=True)

# Remember last successful query so price filters re-render without re-submission
if submitted:
    st.session_state["current_sku"] = norm_sku(sku_input)

query_sku = st.session_state.get("current_sku", None)

# --- Queried item header (name + centered image) ---
if query_sku:
    # name (optional)
    item_name = get_item_name(query_sku)
    title = item_name if item_name else query_sku
    st.markdown(f"<h3 class='center-title'>Selected item: {title}{'' if item_name is None else f'  ·  {query_sku}'}</h3>", unsafe_allow_html=True)

    img_path = find_image_by_sku(query_sku)
    if img_path and img_path.exists():
        img = load_image_safe(img_path)
        if img is not None:
            left, mid, right = st.columns([1, 2, 1])
            with mid:
                st.image(img, caption=query_sku if item_name else None, use_container_width=False)
    else:
        st.info("No image found for the selected item.")

    # small hint
    st.markdown("<div class='small-dim center-title'>Results limited to top 10 after filtering.</div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

    # # ===== NEW: Similar items generated by GNN (above Logic 1 — Same) =====
    # st.subheader("Similar Items Generated by Graph Neural Network")
    # st.caption(f"Source file: `{SIMILAR_CSV.name}`")

    # if not SIMILAR_CSV.exists():
    #     st.warning("Data file not found. Please ensure 'data/similar_products.csv' exists.")
    # else:
    #     with st.spinner("Loading data..."):
    #         sdf = load_similar_df(SIMILAR_CSV)
    #         srows = get_similar_for_item(sdf, query_sku)

    #     if srows.empty:
    #         st.info("No similar items found for this SKU.")
    #     else:
    #         # 仅展示前 TOP_K 个（与你其他区块一致）
    #         sshow = srows.head(TOP_K)
    #         st.write(f"Showing {len(sshow)} item(s).")
    #         render_grid_with_score(sshow[["rec_sku", "similarity_score_num"]])

    #         # 可选：下载按钮
    #         out = sshow[["item_sku", "rec_sku", "similarity_score_num"]].rename(
    #             columns={"similarity_score_num": "similarity_score"}
    #         )
    #         st.download_button(
    #             "Download section CSV",
    #             data=out.to_csv(index=False).encode("utf-8"),
    #             file_name=f"gnn_similar_{query_sku}.csv",
    #             mime="text/csv",
    #             use_container_width=False
    #         )

    # st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
    # # ===== END NEW =====





    # --- Render each section in fixed order ---
    for logic, pline in SECTIONS:
        csv_path = csv_path_for(logic, pline)
        st.subheader(section_title(logic, pline))
        st.caption(f"Source file: `{csv_path.name}`")

        if not csv_path.exists():
            st.warning("Data file not found. Please ensure the CSV exists.")
            st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
            continue

        with st.spinner("Loading data..."):
            df = load_recs_df(csv_path)
            recs = get_recs_for_item(df, query_sku)

        if recs.empty:
            st.info("No recommendations for this item in this logic/line.")
            st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)
            continue

        # price range defaults from available rows
        has_price = pd.notna(recs["price_num"]).any()
        if has_price:
            min_price = float(recs["price_num"].min(skipna=True))
            max_price = float(recs["price_num"].max(skipna=True))
        else:
            min_price = 0.0
            max_price = 0.0

        c1, c2, c3 = st.columns([1, 1, 6])
        with c1:
            pmin = st.number_input(
                "Min price",
                min_value=0.0,
                value=min_price if has_price else 0.0,
                key=price_box_key(logic, pline, "min"),
                help="Lower bound (inclusive)."
            )
        with c2:
            pmax = st.number_input(
                "Max price",
                min_value=0.0,
                value=max_price if has_price else 0.0,
                key=price_box_key(logic, pline, "max"),
                help="Upper bound (inclusive)."
            )

        # apply price filter (only if price is available)
        recs_f = recs.copy()
        if has_price:
            # ensure pmin <= pmax (swap if needed)
            if pmin > pmax:
                pmin, pmax = pmax, pmin
            recs_f = recs_f.loc[
                (recs_f["price_num"] >= float(pmin)) & (recs_f["price_num"] <= float(pmax))
            ]

        # limit to TOP_K after filtering, preserving score order
        if "score_num" in recs_f.columns:
            recs_f = recs_f.sort_values("score_num", ascending=False, na_position="last")
        recs_f = recs_f.head(TOP_K)

        st.write(f"Showing {len(recs_f)} item(s){' with price filter applied' if has_price else ''}.")

        if recs_f.empty:
            st.info("No results within the selected price range.")
        else:
            render_grid(recs_f[["rec_sku", "price_num", "score_num"]])

        # Optional: download this section's results
        out = recs_f[["item_sku","rec_sku","price_num","score_num"]].rename(
            columns={"price_num":"price","score_num":"score"}
        )
        st.download_button(
            "Download section CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"{logic}_{pline}_{query_sku}_filtered_recs.csv",
            mime="text/csv",
            use_container_width=False
        )

        st.markdown("<hr class='section-sep'/>", unsafe_allow_html=True)

else:
    st.info("Enter an item_sku above and click Search.")
