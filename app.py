"""Digit Pixelation Visualizer - Streamlit UI only."""
from __future__ import annotations

import io

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from canvas_utils import canvas_to_grayscale, has_ink
from pixelation import GRID_SIZE, process_digit, render_pixel_grid_with_values

CANVAS_SIZE = 280

st.set_page_config(page_title="Digit Pixelation Visualizer", page_icon="✏️", layout="wide")

# ---------------------------------------------------------------- session state
if "reset_all" not in st.session_state:
    st.session_state.reset_all = 0  # bump to clear every canvas
if "reset_digit" not in st.session_state:
    st.session_state.reset_digit = {d: 0 for d in range(10)}  # bump to clear one


def clear_all() -> None:
    st.session_state.reset_all += 1


def clear_digit(d: int) -> None:
    st.session_state.reset_digit[d] += 1


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    stroke_width = st.slider("Brush size", 8, 40, 20, help="Stroke width in pixels")
    render_size = st.select_slider("Pixel-grid size", [256, 320, 384, 448], value=320)
    show_values = st.toggle("Show 0–255 values", value=True)
    st.button("🧹 Clear All", on_click=clear_all, use_container_width=True)
    st.markdown("---")
    st.caption(
        "Pipeline: grayscale → bounding box → center & pad (20%) → "
        "resize to 32×32 (area averaging)."
    )

# Ten canvases do not fit in ten equal columns; let the page scroll sideways.
st.markdown(
    f"""
    <style>
    .block-container {{ min-width: {10 * (render_size + 40)}px; }}
    .centered {{ text-align: center; }}
    .hero {{
        padding: 1.2rem 1.6rem; border-radius: 16px; color: white;
        background: linear-gradient(120deg, #6a11cb 0%, #2575fc 100%);
        margin-bottom: 1rem;
    }}
    .hero h1 {{ margin: 0; font-size: 2.2rem; }}
    .hero p {{ margin: .3rem 0 0 0; opacity: .9; }}
    .badge {{ text-align:center; font-size:.8rem; padding:2px 8px; border-radius:10px; }}
    .on {{ background:#1f9d55; color:white; }}
    .off {{ background:#444; color:#bbb; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------ header
st.markdown(
    "<div class='hero'><h1>Hello there, Mayank this side 👋</h1>"
    "<p>Draw a digit in any box below and watch it get centered and pixelated "
    "to 32×32 live.</p></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------- the columns
cols = st.columns(10)
drawn = 0
for digit, col in enumerate(cols):
    with col:
        st.markdown(f"<h3 class='centered'>{digit}</h3>", unsafe_allow_html=True)

        # 1. Drawing canvas (mouse or pen), white stroke on black.
        result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            update_streamlit=True,  # live update on every stroke
            key=f"canvas_{digit}_{st.session_state.reset_all}_{st.session_state.reset_digit[digit]}",
        )

        # 2. Arrow + caption.
        st.markdown(
            "<div class='centered' style='font-size:2rem;line-height:1'>↓</div>"
            "<div class='centered'><small>raw → pixelated</small></div>",
            unsafe_allow_html=True,
        )

        # 3 + 4. Pixelated 32x32 result with intensity values overlaid.
        gray = canvas_to_grayscale(result.image_data)
        inked = has_ink(gray)
        if inked:
            pixels = process_digit(gray)
            drawn += 1
        else:  # empty canvas -> placeholder grid of zeros
            pixels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        img = render_pixel_grid_with_values(pixels, render_size, show_values)
        st.image(img, width=render_size)

        # Interactive extras: status badge, live stats, per-digit actions.
        if inked:
            st.markdown("<div class='badge on'>drawn ✓</div>", unsafe_allow_html=True)
            lit = int((pixels > 0).sum())
            st.caption(f"lit cells: {lit}/{GRID_SIZE**2} · max {pixels.max()} · mean {pixels.mean():.1f}")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("⬇ PNG", buf.getvalue(), f"digit_{digit}.png", "image/png", key=f"dl_{digit}")
        else:
            st.markdown("<div class='badge off'>empty</div>", unsafe_allow_html=True)
        st.button("Clear", key=f"clr_{digit}", on_click=clear_digit, args=(digit,))

# Progress summary (rendered after the loop so it reflects the current run).
st.sidebar.progress(drawn / 10, text=f"{drawn}/10 digits drawn")
