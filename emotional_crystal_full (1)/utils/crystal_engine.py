# ============================================================
# crystal_engine.py — Emotional Crystal Pro
# High-performance Crystal Mix rendering engine
# ============================================================

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random, math


# ============================================================
# RANDOM CRYSTAL SHAPE GENERATION
# ============================================================

def crystal_shape(center_x, center_y, radius, wobble):
    """
    Generate an irregular polygon shaped crystal.
    - center_x, center_y: crystal center position
    - radius: base radius
    - wobble: [0~1] randomness for radius variation
    """
    points = []
    # Random number of vertices (crystal-like)
    n = random.randint(5, 12)

    for i in range(n):
        # angle per vertex, but randomness added
        angle = 2 * math.pi * (i / n) + random.uniform(-0.25, 0.25)

        # radius varies randomly within wobble range
        r = radius * (1 + wobble * random.uniform(-1, 1))

        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        points.append((x, y))

    # Close polygon
    points.append(points[0])
    return points



# ============================================================
# COLOR JITTER (slight random variation)
# ============================================================

def jitter_color(rgb, amount=18):
    """Add slight random variation to an RGB color."""
    r, g, b = rgb
    return (
        max(0, min(255, r + random.randint(-amount, amount))),
        max(0, min(255, g + random.randint(-amount, amount))),
        max(0, min(255, b + random.randint(-amount, amount))),
    )



# ============================================================
# DRAW SOFT POLYGON (Gaussian Blur)
# ============================================================

def draw_polygon_soft(base_img, points, fill_rgb, alpha, blur_px):
    """
    Draw crystal polygon onto base_img using:
    - RGBA overlay layer
    - Gaussian blur for soft edges
    - alpha controls transparency
    """
    w, h = base_img.size

    # Create transparent temp layer
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    drawer = ImageDraw.Draw(layer)

    # Apply alpha to fill color
    fill = (*fill_rgb, int(alpha * 255))

    # Draw polygon
    drawer.polygon(points, fill=fill)

    # Apply blur for soft crystal glow
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur_px))

    # Composite layer onto base
    base_img.alpha_composite(layer)



# ============================================================
# CRYSTAL MIX RENDERER (MAIN ENGINE)
# ============================================================

def render_crystalmix(
    df,
    palette,
    width=900,
    height=900,
    seed=0,
    layers=8,
    shapes_per_emotion=20,
    min_size=20,
    max_size=120,
    fill_alpha=0.5,
    blur_px=6,
    wobble=0.35,
    bg_color=(10, 10, 20),
):
    """
    Core crystal rendering engine.
    HIGH PERFORMANCE VERSION.
    """

    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Base image (RGBA)
    base = Image.new("RGBA", (width, height), (*bg_color, 255))

    # Transparent working canvas
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Determine which emotions appear (sorted by frequency)
    if df.empty:
        emotions = ["joy", "love", "calm"]
    else:
        emo_counts = df["emotion"].value_counts()
        emotions = emo_counts.index.tolist()

    # Main multi-layer rendering loop
    for layer_i in range(layers):
        for emotion in emotions:

            if emotion not in palette:
                color = (200, 200, 200)
            else:
                color = palette[emotion]

            # Vibrancy boost (slightly brighter)
            color = tuple(min(255, int(c * 1.05)) for c in color)

            for _ in range(shapes_per_emotion):

                # Random placement with margins
                cx = random.randint(50, width - 50)
                cy = random.randint(50, height - 50)

                # Random size
                radius = random.randint(min_size, max_size)

                # Create irregular polygon
                points = crystal_shape(cx, cy, radius, wobble)

                # Jitter the color
                c2 = jitter_color(color, amount=18)

                # Randomize alpha slightly
                alpha_var = max(0.05, min(1.0, fill_alpha + random.uniform(-0.08, 0.08)))

                # Randomize blur
                blur_var = max(1, blur_px + random.randint(-2, 2))

                # Draw crystal fragment
                draw_polygon_soft(
                    canvas,
                    points,
                    fill_rgb=c2,
                    alpha=alpha_var,
                    blur_px=blur_var,
                )

    # Composite final onto base
    base.alpha_composite(canvas)

    # Return as RGB for cinematic processing
    return base.convert("RGB")
