# ============================================================
# palette.py — Emotional Crystal Pro
# Palette management: default colors, custom colors, CSV IO, UI
# ============================================================

import streamlit as st
import pandas as pd
import io


# ============================================================
# DEFAULT 20+ EMOTION COLORS
# ============================================================

def load_default_palette():
    """
    Returns the predefined 20+ emotion → RGB palette.
    """
    return {
        "joy": (255, 200, 60),
        "love": (255, 95, 150),
        "pride": (255, 160, 70),
        "hope": (255, 220, 120),

        "calm": (120, 200, 255),
        "curiosity": (200, 220, 255),
        "surprise": (150, 230, 255),
        "trust": (100, 180, 255),
        "awe": (180, 150, 255),
        "nostalgia": (255, 180, 200),

        "anger": (245, 60, 60),
        "fear": (120, 60, 200),
        "sadness": (70, 120, 255),
        "anxiety": (130, 150, 200),
        "disgust": (150, 200, 90),

        "boredom": (180, 180, 180),
        "neutral": (200, 200, 200),
        "mixed": (210, 160, 160),
    }


# ============================================================
# GET ACTIVE PALETTE (DEFAULT + CUSTOM OR CSV ONLY)
# ============================================================

def get_active_palette(default_palette, custom_palette, use_csv_only: bool):
    """
    Combines default palette + custom palette.
    If use_csv_only is True → return only custom_palette.
    """
    if use_csv_only:
        return custom_palette.copy()

    merged = default_palette.copy()
    merged.update(custom_palette)
    return merged


# ============================================================
# CSV IMPORT LOGIC
# ============================================================

def import_palette_csv(uploaded_file, custom_palette_state):
    """
    Import CSV containing emotion, r, g, b columns.
    Updates custom_palette_state dict.
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        required = {"emotion", "r", "g", "b"}
        if not required.issubset(set(df.columns)):
            st.error("CSV must contain columns: emotion, r, g, b")
            return

        count = 0
        for _, row in df.iterrows():
            emo = str(row["emotion"]).strip()
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            custom_palette_state[emo] = (r, g, b)
            count += 1

        st.success(f"Imported {count} colors from CSV!")

    except Exception as e:
        st.error(f"Failed to import CSV: {e}")


# ============================================================
# CSV EXPORT LOGIC
# ============================================================

def export_palette_csv(palette):
    """
    Convert palette dict → downloadable CSV file.
    Returns bytes.
    """
    rows = []
    for emo, rgb in palette.items():
        r, g, b = rgb
        rows.append({"emotion": emo, "r": r, "g": g, "b": b})

    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ============================================================
# FULL PALETTE UI SECTION (SIDEBAR)
# ============================================================

def palette_ui_section(panel, default_palette, custom_palette_state):
    """
    Builds the full palette section in sidebar:
    - custom color entries
    - CSV upload/import
    - CSV download
    - shows full palette table
    """

    # --------------------------------------------------------
    # USE CSV ONLY
    # --------------------------------------------------------

    use_csv_only = panel.checkbox(
        "Use CSV palette only (ignore default)",
        st.session_state.use_csv_only
    )
    st.session_state.use_csv_only = use_csv_only

    # --------------------------------------------------------
    # ADD CUSTOM EMOTION COLOR
    # --------------------------------------------------------

    with panel.expander("➕ Add Custom Emotion Color"):
        emo_name = panel.text_input("Emotion Name")
        c1, c2, c3 = panel.columns(3)
        r = c1.number_input("R", 0, 255, 150)
        g = c2.number_input("G", 0, 255, 150)
        b = c3.number_input("B", 0, 255, 150)

        if panel.button("Add Color"):
            if emo_name.strip():
                custom_palette_state[emo_name.strip()] = (int(r), int(g), int(b))
                st.success(f"Added emotion color: {emo_name}")
            else:
                st.warning("Emotion name cannot be empty.")


    # --------------------------------------------------------
    # IMPORT CSV
    # --------------------------------------------------------

    with panel.expander("📥 Import Palette CSV"):
        uploaded_csv = panel.file_uploader(
            "Upload CSV (emotion, r, g, b)",
            type=["csv"]
        )
        if uploaded_csv is not None:
            import_palette_csv(uploaded_csv, custom_palette_state)


    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    with panel.expander("📤 Export Current Palette CSV"):
        active = get_active_palette(default_palette, custom_palette_state, use_csv_only)

        csv_bytes = export_palette_csv(active)

        panel.download_button(
            "Download Palette CSV",
            data=csv_bytes,
            file_name="palette_export.csv",
            mime="text/csv"
        )


    # --------------------------------------------------------
    # PREVIEW TABLE
    # --------------------------------------------------------

    active_palette = get_active_palette(default_palette, custom_palette_state, use_csv_only)
    panel.write("### Current Active Palette")
    panel.dataframe(
        pd.DataFrame(
            [{"emotion": e, "rgb": active_palette[e]} for e in active_palette]
        )
    )

    return active_palette
