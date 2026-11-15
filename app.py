"""Digit Pixelation Visualizer - Streamlit UI only."""
from __future__ import annotations

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from canvas_utils import canvas_to_grayscale, has_ink
from pixelation import GRID_SIZE, process_digit, render_pixel_grid_with_values

CANVAS_SIZE = 280
STROKE_WIDTH = 20
RENDER_SIZE = 320

st.set_page_config(page_title="Digit Pixelation Visualizer", layout="wide")

# Ten 280px canvases will not fit in ten equal columns on a normal screen,
# so give the page a minimum width and let it scroll horizontally.
st.markdown(
    f"""
    <style>
    .block-container {{ min-width: {10 * (RENDER_SIZE + 40)}px; }}
    .centered {{ text-align: center; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Digit Pixelation Visualizer")
st.caption("Draw each digit; it is cropped, centered and downsampled to 32×32 live.")

# Changing the widget keys is how we reset every canvas at once.
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0


def clear_all() -> None:
    st.session_state.reset_counter += 1


st.button("Clear All", on_click=clear_all)

cols = st.columns(10)
for digit, col in enumerate(cols):
    with col:
        st.markdown(f"<h3 class='centered'>{digit}</h3>", unsafe_allow_html=True)

        # 1. Drawing canvas (mouse or pen), white stroke on black.
        result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=STROKE_WIDTH,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            update_streamlit=True,  # live update on every stroke
            key=f"canvas_{digit}_{st.session_state.reset_counter}",
        )

        # 2. Arrow + caption.
        st.markdown(
            "<div class='centered' style='font-size:2rem;line-height:1'>↓</div>"
            "<div class='centered'><small>raw → pixelated</small></div>",
            unsafe_allow_html=True,
        )

        # 3 + 4. Pixelated 32x32 result with intensity values overlaid.
        gray = canvas_to_grayscale(result.image_data)
        if has_ink(gray):
            pixels = process_digit(gray)
        else:  # empty canvas -> placeholder grid of zeros
            pixels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        st.image(render_pixel_grid_with_values(pixels, RENDER_SIZE), width=RENDER_SIZE)
