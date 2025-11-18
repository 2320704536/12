# ============================================================
# cinematic.py — Emotional Crystal Pro
# Full cinematic color pipeline (performance optimized version)
# ============================================================

import numpy as np
from PIL import Image, ImageFilter


# ============================================================
# sRGB ↔ Linear Conversion
# ============================================================

def srgb_to_linear(arr):
    """Convert sRGB image → linear light."""
    arr = arr / 255.0
    thresh = 0.04045
    low = arr <= thresh
    high = ~low
    out = np.zeros_like(arr)
    out[low] = arr[low] / 12.92
    out[high] = ((arr[high] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(arr):
    """Convert linear light → sRGB."""
    thresh = 0.0031308
    low = arr <= thresh
    high = ~low
    out = np.zeros_like(arr)
    out[low] = arr[low] * 12.92
    out[high] = 1.055 * (arr[high] ** (1/2.4)) - 0.055
    out = np.clip(out, 0, 1)
    return out * 255.0



# ============================================================
# Exposure (in linear space)
# ============================================================

def apply_exposure(lin, exposure):
    """
    Exposure in stops: multiply linear by 2**exposure.
    """
    factor = 2 ** exposure
    return lin * factor



# ============================================================
# White Balance (Temperature + Tint)
# ============================================================

def apply_white_balance(lin, temp, tint):
    """
    temp: [-1,1]  blue ↔ yellow shift
    tint: [-1,1]  green ↔ magenta shift
    """
    r = lin[..., 0]
    g = lin[..., 1]
    b = lin[..., 2]

    # Temperature adjustment (blue ↔ yellow)
    temp_scale = 1 + temp * 0.20
    r *= (1 + temp * 0.10)
    b *= (1 - temp * 0.10)
    g *= 1.0

    # Tint adjustment (green ↔ magenta)
    g *= (1 + tint * 0.20)
    r *= (1 - tint * 0.05)
    b *= (1 - tint * 0.05)

    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, None)



# ============================================================
# Highlight Roll-off
# ============================================================

def apply_highlight_rolloff(lin, strength=0.3):
    """
    Soften near-white highlights to avoid clipping.
    """
    lum = np.mean(lin, axis=2, keepdims=True)
    roll = 1.0 - strength * np.clip((lum - 0.8) / 0.2, 0, 1)
    return lin * roll



# ============================================================
# Contrast
# ============================================================

def apply_contrast(srgb_arr, contrast):
    """
    Simple contrast around mid gray 0.5
    """
    arr = srgb_arr / 255.0
    arr = (arr - 0.5) * (1 + contrast) + 0.5
    arr = np.clip(arr, 0, 1)
    return arr * 255.0



# ============================================================
# Saturation
# ============================================================

def apply_saturation(srgb_arr, saturation):
    """
    Adjust saturation in sRGB space.
    """
    arr = srgb_arr / 255.0
    lum = np.dot(arr, [0.299, 0.587, 0.114])
    lum = lum[..., None]
    arr = lum + (arr - lum) * (1 + saturation)
    arr = np.clip(arr, 0, 1)
    return arr * 255.0



# ============================================================
# Gamma Correction
# ============================================================

def apply_gamma(srgb_arr, gamma):
    arr = srgb_arr / 255.0
    arr = np.power(arr, 1.0 / gamma)
    arr = np.clip(arr, 0, 1)
    return arr * 255.0



# ============================================================
# Split Toning
# ============================================================

def apply_split_toning(arr, sh_rgb, hi_rgb, balance):
    """
    sh_rgb: (R,G,B) for shadows
    hi_rgb: (R,G,B) for highlights
    balance: -1 → shadows, +1 → highlights
    """
    arr_f = arr / 255.0
    lum = np.mean(arr_f, axis=2, keepdims=True)

    # shadow / highlight weights
    shadow_w = np.clip(1 - ((lum - 0.5) * 2 * (1 + balance)), 0, 1)
    highlight_w = 1 - shadow_w

    sh = np.array(sh_rgb) / 255.0
    hi = np.array(hi_rgb) / 255.0

    tinted = (arr_f * (1 - shadow_w - highlight_w)
              + sh * shadow_w
              + hi * highlight_w)

    tinted = np.clip(tinted, 0, 1)
    return tinted * 255.0



# ============================================================
# Bloom (High Performance)
# ============================================================

def apply_bloom(img, radius, intensity):
    """
    Simple high-performance bloom: blur + add.
    """
    if radius <= 0 or intensity <= 0:
        return img

    blur = img.filter(ImageFilter.GaussianBlur(radius=radius))
    arr = np.asarray(img).astype(np.float32)
    blur_arr = np.asarray(blur).astype(np.float32)

    out = arr + blur_arr * intensity
    out = np.clip(out, 0, 255)
    return Image.fromarray(out.astype(np.uint8))



# ============================================================
# Vignette
# ============================================================

def apply_vignette(img, strength):
    if strength <= 0:
        return img

    w, h = img.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    mask = 1 - np.clip(dist * strength, 0, 1)

    arr = np.asarray(img).astype(np.float32)
    mask = mask[..., None]
    out = arr * mask
    out = np.clip(out, 0, 255)
    return Image.fromarray(out.astype(np.uint8))



# ============================================================
# Auto Brightness Compensation
# ============================================================

def apply_auto_brightness(srgb_arr, target_mean, remap_strength,
                          black_point, white_point, max_gain):
    """
    Histogram-based brightness remap.
    """
    arr = srgb_arr.astype(np.float32)
    lum = np.mean(arr, axis=2)

    # Percentile black/white
    lo = np.percentile(lum, black_point)
    hi = np.percentile(lum, white_point)

    if hi - lo < 1e-5:
        return srgb_arr  # avoid divide-by-zero

    # Normalize
    lum_norm = (lum - lo) / (hi - lo)
    lum_norm = np.clip(lum_norm, 0, 1)

    # Remap blend
    lum_remap = lum * (1 - remap_strength) + lum_norm * 255 * remap_strength

    # Global gain
    current_mean = lum_remap.mean()
    gain = target_mean * 255 / max(current_mean, 1)
    gain = min(gain, max_gain)

    arr = arr * gain
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)



# ============================================================
# FULL CINEMATIC PIPELINE
# ============================================================

def apply_cinematic_pipeline(
    img,
    exposure,
    contrast,
    saturation,
    gamma,
    wb_temp,
    wb_tint,
    highlight_rolloff,
    split_shadow_rgb,
    split_highlight_rgb,
    tone_balance,
    bloom_radius,
    bloom_intensity,
    vignette_strength,
    enable_auto_brightness,
    abc_target_mean,
    abc_remap_strength,
    abc_black,
    abc_white,
    abc_max_gain,
):
    """
    Full cinematic pipeline (performance optimized).
    """

    # Convert to NumPy
    arr = np.asarray(img).astype(np.float32)

    # ----------------------------------------
    # 1. Convert to linear
    lin = srgb_to_linear(arr)

    # ----------------------------------------
    # 2. Exposure
    lin = apply_exposure(lin, exposure)

    # ----------------------------------------
    # 3. White Balance
    lin = apply_white_balance(lin, wb_temp, wb_tint)

    # ----------------------------------------
    # 4. Highlight Roll-off
    lin = apply_highlight_rolloff(lin, strength=highlight_rolloff)

    # ----------------------------------------
    # 5. Back to sRGB
    arr = linear_to_srgb(lin)

    # ----------------------------------------
    # 6. Contrast
    arr = apply_contrast(arr, contrast)

    # ----------------------------------------
    # 7. Saturation
    arr = apply_saturation(arr, saturation)

    # ----------------------------------------
    # 8. Gamma
    arr = apply_gamma(arr, gamma)

    # ----------------------------------------
    # 9. Split Toning
    arr = apply_split_toning(arr, split_shadow_rgb, split_highlight_rgb, tone_balance)

    # Convert to PIL before bloom/vignette
    img2 = Image.fromarray(arr.astype(np.uint8))

    # ----------------------------------------
    # 10. Bloom
    img2 = apply_bloom(img2, radius=bloom_radius, intensity=bloom_intensity)

    # ----------------------------------------
    # 11. Vignette
    img2 = apply_vignette(img2, strength=vignette_strength)

    # ----------------------------------------
    # 12. Auto Brightness Compensation
    if enable_auto_brightness:
        arr2 = np.asarray(img2).astype(np.float32)
        arr2 = apply_auto_brightness(
            arr2,
            target_mean=abc_target_mean,
            remap_strength=abc_remap_strength,
            black_point=abc_black,
            white_point=abc_white,
            max_gain=abc_max_gain,
        )
        img2 = Image.fromarray(arr2)

    return img2
