"""Digit Pixelation Visualizer - Streamlit UI only."""
from __future__ import annotations

import io

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from canvas_utils import canvas_to_grayscale, has_ink
from pixelation import COLORMAPS, GRID_SIZE, process_digit, render_pixel_grid_with_values

CANVAS_SIZE = 280

st.set_page_config(page_title="Digit Pixelation Visualizer", page_icon="✏️", layout="wide")

# ---------------------------------------------------------------- session state
st.session_state.setdefault("reset_all", 0)  # bump to clear every canvas
st.session_state.setdefault("reset_digit", {d: 0 for d in range(10)})  # clear one
st.session_state.setdefault("celebrated", False)


def clear_all() -> None:
    st.session_state.reset_all += 1
    st.session_state.celebrated = False


def clear_digit(d: int) -> None:
    st.session_state.reset_digit[d] += 1
    st.session_state.celebrated = False


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control panel")
    stroke_width = st.slider("🖌️ Brush size", 8, 40, 20, help="Stroke width in pixels")
    render_size = st.select_slider("🔍 Pixel-grid size", [256, 320, 384, 448], value=320)
    colormap = st.selectbox("🎨 Color theme", list(COLORMAPS), index=1)
    show_values = st.toggle("🔢 Show 0–255 values", value=True)
    st.button("🧹 Clear all canvases", on_click=clear_all, width="stretch")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n1. grayscale\n2. bounding box of ink\n3. center + 20% padding\n"
        "4. area-resize to 32×32"
    )
    sidebar_slot = st.container()  # filled after the loop (progress + export)

# ------------------------------------------------------------------------- style
st.markdown(
    f"""
    <style>
    .block-container {{ min-width: {10 * (render_size + 40)}px; padding-top: 2rem; }}
    .centered {{ text-align: center; }}
    .hero {{
        padding: 1.6rem 2rem; border-radius: 20px; color: white; margin-bottom: 1.2rem;
        background: linear-gradient(120deg,#6a11cb,#2575fc,#00c6ff,#6a11cb);
        background-size: 300% 300%; animation: flow 10s ease infinite;
        box-shadow: 0 10px 30px rgba(37,117,252,.35);
    }}
    .hero h1 {{ margin: 0; font-size: 2.6rem; }}
    .hero p {{ margin: .4rem 0 0 0; font-size: 1.1rem; opacity: .92; }}
    .wave {{ display:inline-block; animation: wave 2s infinite; transform-origin: 70% 70%; }}
    @keyframes flow {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
    @keyframes wave {{ 0%,60%,100%{{transform:rotate(0)}} 10%,30%{{transform:rotate(16deg)}} 20%{{transform:rotate(-8deg)}} 40%{{transform:rotate(-4deg)}} 50%{{transform:rotate(10deg)}} }}
    div[data-testid="stColumn"] > div {{
        background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px; padding: .8rem .6rem; transition: all .25s ease;
    }}
    div[data-testid="stColumn"] > div:hover {{
        transform: translateY(-4px); border-color: #2575fc;
        box-shadow: 0 8px 24px rgba(37,117,252,.28);
    }}
    .digit {{ text-align:center; font-size:2.4rem; font-weight:800; line-height:1;
        background: linear-gradient(120deg,#6a11cb,#2575fc); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; margin-bottom:.4rem; }}
    .badge {{ text-align:center; font-size:.8rem; padding:3px 10px; border-radius:12px; margin:.3rem 0; }}
    .on {{ background:#1f9d55; color:white; }}
    .off {{ background:rgba(128,128,128,.3); }}
    canvas {{ border-radius: 12px; cursor: crosshair; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------ header
st.markdown(
    "<div class='hero'><h1>Hello there, Mayank this side <span class='wave'>👋</span></h1>"
    "<p>Draw a digit in any box and watch it get centered, cropped and pixelated "
    "to 32×32 — live, on every stroke.</p></div>",
    unsafe_allow_html=True,
)
metrics_slot = st.container()  # filled after the loop

# ------------------------------------------------------------------- the columns
cols = st.columns(10)
drawn = 0
total_lit = 0
export_rows: list[np.ndarray] = []
for digit, col in enumerate(cols):
    with col:
        st.markdown(f"<div class='digit'>{digit}</div>", unsafe_allow_html=True)

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
            return_image_data=True,  # needed by newer canvas versions
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
        total_lit += int((pixels > 0).sum())
        export_rows.append(pixels.flatten())
        img = render_pixel_grid_with_values(pixels, render_size, show_values, colormap)
        st.image(img, width=render_size)

        # Interactive extras: status badge, live stats, per-digit actions.
        if inked:
            st.markdown("<div class='badge on'>drawn ✓</div>", unsafe_allow_html=True)
            st.caption(
                f"lit {int((pixels > 0).sum())}/{GRID_SIZE**2} · max {pixels.max()} · mean {pixels.mean():.1f}"
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("⬇ PNG", buf.getvalue(), f"digit_{digit}.png", "image/png", key=f"dl_{digit}")
        else:
            st.markdown("<div class='badge off'>empty</div>", unsafe_allow_html=True)
        st.button("🗑 Clear", key=f"clr_{digit}", on_click=clear_digit, args=(digit,))

# ----------------------------------------------------- summary (after the loop)
with metrics_slot:
    m1, m2, m3 = st.columns(3)
    m1.metric("Digits drawn", f"{drawn} / 10")
    m2.metric("Total lit cells", f"{total_lit:,}")
    m3.metric("Grid resolution", f"{GRID_SIZE}×{GRID_SIZE}")

with sidebar_slot:
    st.progress(drawn / 10, text=f"{drawn}/10 digits drawn")
    csv = "\n".join(",".join(map(str, r)) for r in export_rows)
    st.download_button("📄 Export all as CSV", csv, "digits_32x32.csv", "text/csv", width="stretch")

# Little celebration the first time all ten digits are drawn.
if drawn == 10 and not st.session_state.celebrated:
    st.session_state.celebrated = True
    st.balloons()
