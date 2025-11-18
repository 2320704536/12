# ============================================================
# Emotional Crystal — Full Professional Version (Performance Optimized)
# app.py — Main Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import random
from datetime import datetime

# Utils modules
from utils.sentiment import fetch_news_data, analyze_sentiment_dataframe
from utils.palette import (
    load_default_palette,
    palette_ui_section,
    get_active_palette
)
from utils.crystal_engine import (
    render_crystalmix
)
from utils.cinematic import (
    apply_cinematic_pipeline
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Emotional Crystal — Final Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# INITIAL SESSION STATE
# ============================================================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "palette_custom" not in st.session_state:
    st.session_state.palette_custom = {}

if "use_csv_only" not in st.session_state:
    st.session_state.use_csv_only = False

if "random_mode" not in st.session_state:
    st.session_state.random_mode = False

# ============================================================
# TITLE
# ============================================================
st.title("❄ Emotional Crystal — Final Pro (Performance Optimized Version)")
st.caption("Generate cinematic emotional crystal art from text emotions, news articles, or random mode.")


# ============================================================
# SIDEBAR — SECTION 1: DATA SOURCE
# ============================================================
st.sidebar.header("📡 Data Source")

source_mode = st.sidebar.radio(
    "Choose Mode:",
    ["NewsAPI Text Mode", "Random Crystal Mode"],
)

# ============================================================
# NEWS MODE
# ============================================================
if source_mode == "NewsAPI Text Mode":
    st.session_state.random_mode = False

    keyword = st.sidebar.text_input("Keyword for NewsAPI (English only)", "")
    fetch_btn = st.sidebar.button("🔍 Fetch News")

    if fetch_btn and keyword.strip():
        with st.spinner("Fetching news from NewsAPI..."):
            df = fetch_news_data(keyword)
            if not df.empty:
                df = analyze_sentiment_dataframe(df)
                st.session_state.df = df

# ============================================================
# RANDOM MODE
# ============================================================
else:
    st.session_state.random_mode = True
    if st.sidebar.button("✨ Random Generate (Crystal Mode)"):
        # Randomly generate a DataFrame with random emotions only
        df = pd.DataFrame({
            "emotion": np.random.choice(
                list(load_default_palette().keys()),
                size=100,
                replace=True
            )
        })
        df["compound"] = 0.0
        st.session_state.df = df


# ============================================================
# GET CURRENT DATAFRAME
# ============================================================
df = st.session_state.df
# ============================================================
# SIDEBAR — SECTION 2: EMOTION MAPPING & FILTERING
# ============================================================

st.sidebar.header("🎭 Emotion Mapping")

# Compound score filter
compound_min, compound_max = st.sidebar.slider(
    "Filter by VADER Compound Score:",
    -1.0, 1.0, (-1.0, 1.0), 0.01
)

# If in random mode, no sentiment filtering
if not st.session_state.random_mode and not df.empty:
    df = df[(df["compound"] >= compound_min) & (df["compound"] <= compound_max)]

# Auto Top-3 Emotion Selection
auto_top3 = st.sidebar.checkbox("Auto select Top-3 emotions", False)

if not df.empty:
    unique_emotions = df["emotion"].unique().tolist()
else:
    unique_emotions = []

# Build emotion labels with color name
emotion_display_labels = []
for emo in unique_emotions:
    emotion_display_labels.append(emo)

# Emotion multiselect
selected_emotions = st.sidebar.multiselect(
    "Select Emotions to Include:",
    options=unique_emotions,
    default=unique_emotions[:3] if auto_top3 and len(unique_emotions) >= 3 else unique_emotions,
)

# Filter by selected emotions
if not df.empty and len(selected_emotions) > 0:
    df = df[df["emotion"].isin(selected_emotions)]

st.session_state.df = df  # sync back


# ============================================================
# SIDEBAR — SECTION 3: CRYSTAL ENGINE CONTROLS
# ============================================================

st.sidebar.header("❄ Crystal Engine Settings")

layers = st.sidebar.slider("Total Crystal Layers", 1, 30, 10)
shapes_per_emotion = st.sidebar.slider("Crystals per Emotion", 1, 40, 20)
min_size = st.sidebar.slider("Min Crystal Size", 5, 100, 20)
max_size = st.sidebar.slider("Max Crystal Size", 20, 200, 120)
wobble = st.sidebar.slider("Crystal Wobble (Randomness)", 0.0, 1.0, 0.35)
fill_alpha = st.sidebar.slider("Crystal Alpha", 0.05, 1.0, 0.5)
blur_px = st.sidebar.slider("Crystal Softness (Blur px)", 1, 25, 6)
seed = st.sidebar.slider("Random Seed", 0, 9999, 1234)


# ============================================================
# SIDEBAR — SECTION 4: BACKGROUND COLOR
# ============================================================

st.sidebar.header("🌈 Background Color")
bg_hex = st.sidebar.color_picker("Pick Background Color", "#0A0A14")
bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (1, 3, 5))


# ============================================================
# SIDEBAR — SECTION 5: CINEMATIC COLOR GRADING
# (Performance-optimized version)
# ============================================================

st.sidebar.header("🎬 Cinematic Color Controls")

exposure = st.sidebar.slider("Exposure (stops)", -2.0, 2.0, 0.0)
contrast = st.sidebar.slider("Contrast", -0.5, 1.0, 0.2)
saturation = st.sidebar.slider("Saturation", -0.5, 1.5, 0.1)
gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0)

# White Balance
st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Yellow)", -1.0, 1.0, 0.0)
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, 0.0)

# Highlight control
highlight_rolloff = st.sidebar.slider(
    "Highlight Roll-off Strength", 0.0, 1.0, 0.3
)


# ============================================================
# SIDEBAR — SECTION 6: SPLIT TONING
# ============================================================

st.sidebar.header("🎨 Split Toning")

sh_r = st.sidebar.slider("Shadows R", 0, 255, 180)
sh_g = st.sidebar.slider("Shadows G", 0, 255, 200)
sh_b = st.sidebar.slider("Shadows B", 0, 255, 255)

hi_r = st.sidebar.slider("Highlights R", 0, 255, 255)
hi_g = st.sidebar.slider("Highlights G", 0, 255, 220)
hi_b = st.sidebar.slider("Highlights B", 0, 255, 180)

tone_balance = st.sidebar.slider(
    "Tone Balance (Shadow ↔ Highlight)",
    -1.0, 1.0, 0.0
)


# ============================================================
# SIDEBAR — SECTION 7: BLOOM & VIGNETTE
# ============================================================

st.sidebar.header("✨ Bloom & Vignette")

bloom_radius = st.sidebar.slider("Bloom Radius", 0, 50, 18)
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.5, 0.4)
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 1.0, 0.35)


# ============================================================
# SIDEBAR — SECTION 8: AUTO BRIGHTNESS
# ============================================================

st.sidebar.header("⚡ Auto Brightness Compensation")

enable_auto_brightness = st.sidebar.checkbox("Enable Auto Brightness", False)

abc_target_mean = st.sidebar.slider("Target Mean Brightness", 0.1, 0.9, 0.5)
abc_remap_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, 0.5)
abc_black = st.sidebar.slider("Black Point %", 0.0, 5.0, 0.5)
abc_white = st.sidebar.slider("White Point %", 95.0, 100.0, 99.5)
abc_max_gain = st.sidebar.slider("Max Gain Factor", 1.0, 5.0, 2.0)
# ============================================================
# SIDEBAR — SECTION 9: CUSTOM PALETTE SYSTEM (CSV + Manual)
# ============================================================

st.sidebar.header("🎨 Custom Palette (CSV + RGB)")

active_palette = palette_ui_section(
    st.sidebar,
    default_palette=load_default_palette(),
    custom_palette_state=st.session_state.palette_custom,
)


# ============================================================
# MAIN PANEL — LAYOUT
# ============================================================

col_left, col_right = st.columns([1.8, 1])

# ============================================================
# IMAGE GENERATION
# ============================================================

with col_left:

    st.subheader("❄ Crystal Mix Visualization")

    if df.empty:
        st.info("No data available. Try fetching news or using Random Mode.")
        final_img = None
    else:
        with st.spinner("Rendering emotional crystal..."):
            # ---- Crystal Rendering ----
            img = render_crystalmix(
                df=df,
                palette=active_palette,
                width=900,
                height=900,
                seed=seed,
                layers=layers,
                shapes_per_emotion=shapes_per_emotion,
                min_size=min_size,
                max_size=max_size,
                fill_alpha=fill_alpha,
                blur_px=blur_px,
                wobble=wobble,
                bg_color=bg_rgb,
            )

            # ---- Cinematic Processing ----
            final_img = apply_cinematic_pipeline(
                img,
                exposure=exposure,
                contrast=contrast,
                saturation=saturation,
                gamma=gamma,
                wb_temp=temp,
                wb_tint=tint,
                highlight_rolloff=highlight_rolloff,
                split_shadow_rgb=(sh_r, sh_g, sh_b),
                split_highlight_rgb=(hi_r, hi_g, hi_b),
                tone_balance=tone_balance,
                bloom_radius=bloom_radius,
                bloom_intensity=bloom_intensity,
                vignette_strength=vignette_strength,
                enable_auto_brightness=enable_auto_brightness,
                abc_target_mean=abc_target_mean,
                abc_remap_strength=abc_remap_strength,
                abc_black=abc_black,
                abc_white=abc_white,
                abc_max_gain=abc_max_gain,
            )

        # ---- Display Final Image ----
        st.image(final_img, use_column_width=True)

        # ---- Download Button ----
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        st.download_button(
            "⬇ Download Crystal PNG",
            data=buf.getvalue(),
            file_name="emotional_crystal.png",
            mime="image/png",
        )


# ============================================================
# RIGHT PANEL — DATA TABLE
# ============================================================

with col_right:

    st.subheader("📊 Data & Emotion Mapping")

    if df.empty:
        st.info("No data to display.")
    else:
        # Display color names in a readable table
        df_show = df.copy()
        df_show["emotion_color"] = df_show["emotion"].apply(
            lambda e: str(active_palette.get(e, (200, 200, 200)))
        )

        st.dataframe(df_show, height=600)


# ============================================================
# RESET BUTTON
# ============================================================

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()


# ============================================================
# END OF FILE
# ============================================================
