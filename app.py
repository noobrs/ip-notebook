# app.py
# Streamlit frontend for a robust, blind DWT watermarking demo (color host, mono watermark)
# Constraints respected:
# 1) No resizing of host/watermark (we embed watermark bits via spread-spectrum; size-agnostic)
# 2) Host image must be color; watermark can be mono
# 3) Extraction needs only the watermarked image (blind, key-based)

import io
import os
import base64
import numpy as np
import streamlit as st
from PIL import Image
import pywt
import cv2
from skimage.metrics import structural_similarity as ssim
from math import log10

# =============== Utility helpers ===============

def pil_to_np_uint8(img: Image.Image):
    return np.array(img)

def ensure_color(img_np):
    # Expect HxWx3
    if img_np.ndim == 2:
        return np.stack([img_np]*3, axis=-1)
    if img_np.shape[2] == 4:  # RGBA -> RGB
        return cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return img_np

def ensure_gray(img_np):
    if img_np.ndim == 3:
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_np

def to_binary(img_gray_np, method="otsu", thresh=127):
    if method == "otsu":
        _, bw = cv2.threshold(img_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(img_gray_np, thresh, 255, cv2.THRESH_BINARY)
    bw = (bw > 0).astype(np.uint8)
    return bw

def psnr(img1, img2):
    # 8-bit images
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10((255 ** 2) / mse)

def rgb_to_ycbcr(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)  # (note: OpenCV uses YCrCb ordering)

def ycbcr_to_rgb(ycbcr):
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

def bytes_from_image(img_np, fmt="PNG"):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()

def seed_rng(key: str):
    # Deterministic RNG from key
    return np.random.default_rng(abs(hash(key)) % (2**32))

# =============== Spread-Spectrum DWT (Blind) ===============
# We embed a (binary) watermark's flattened bits via PN sequences into a chosen DWT subband of the Y channel.
# Extraction uses only the watermarked image + key, correlating against the PN sequences.

def dwt2_y_channel(rgb_img, wavelet="haar", level=1):
    ycc = rgb_to_ycbcr(rgb_img)
    Y = ycc[:, :, 0].astype(np.float32)
    coeffs = pywt.wavedec2(Y, wavelet=wavelet, level=level)
    return coeffs, ycc

def idwt2_to_rgb(coeffs, ycc, wavelet="haar"):
    Y_rec = pywt.waverec2(coeffs, wavelet=wavelet)
    Y_rec = np.clip(Y_rec, 0, 255).astype(np.uint8)
    out = ycc.copy()
    out[:, :, 0] = Y_rec
    return ycbcr_to_rgb(out)

def pick_subband(coeffs, band="HL", level=1):
    # coeffs = [LL, (LH1, HL1, HH1), (LH2, HL2, HH2), ...]
    # level=1 uses coeffs[1]; level=k uses coeffs[level]
    assert level >= 1 and level < len(coeffs), "Invalid DWT level"
    LH, HL, HH = coeffs[level]
    if band == "LH":
        return LH
    if band == "HL":
        return HL
    if band == "HH":
        return HH
    raise ValueError("band must be one of 'LH','HL','HH'")

def set_subband(coeffs, new_band, band="HL", level=1):
    LH, HL, HH = coeffs[level]
    if band == "LH":
        coeffs[level] = (new_band, HL, HH)
    elif band == "HL":
        coeffs[level] = (LH, new_band, HH)
    elif band == "HH":
        coeffs[level] = (LH, HL, new_band)
    else:
        raise ValueError("band must be one of 'LH','HL','HH'")

def embed_spread_spectrum(rgb_img, wm_bits, key, alpha=5.0, wavelet="haar", level=1, band="HL"):
    coeffs, ycc = dwt2_y_channel(rgb_img, wavelet, level)
    B = pick_subband(coeffs, band, level)

    rng = seed_rng(key)
    H, W = B.shape
    capacity = H * W  # one PN per bit, spanning whole subband
    nbits = wm_bits.size

    # Generate one PN field per bit by chunking the subband; if bits exceed tiles, wrap
    # For simplicity and to avoid resizing, map each bit to a disjoint random mask of equal size
    # Create masks by permuting all indices then splitting
    idx = rng.permutation(H * W)
    tiles = np.array_split(idx, nbits)
    B_mod = B.copy().astype(np.float32)

    # For each bit, generate a PN (+1/-1), add or subtract scaled PN on the allocated indices
    for i, bit in enumerate(wm_bits):
        pn = rng.integers(0, 2, size=tiles[i].size, dtype=np.uint8)
        pn = (pn * 2 - 1).astype(np.float32)  # {0,1} -> {-1,+1}
        sign = 1.0 if bit == 1 else -1.0
        flat = B_mod.ravel()
        flat[tiles[i]] = flat[tiles[i]] + sign * alpha * pn

    B_mod = B_mod.reshape(H, W)
    set_subband(coeffs, B_mod, band, level)
    wm_rgb = idwt2_to_rgb(coeffs, ycc, wavelet)
    return wm_rgb

def extract_spread_spectrum(rgb_img_wm, nbits, key, wavelet="haar", level=1, band="HL"):
    coeffs, _ = dwt2_y_channel(rgb_img_wm, wavelet, level)
    B = pick_subband(coeffs, band, level).astype(np.float32)
    H, W = B.shape

    rng = seed_rng(key)
    idx = rng.permutation(H * W)
    tiles = np.array_split(idx, nbits)

    bits = []
    # Correlate sign of sum with PN to decide bit
    for i in range(nbits):
        pn = rng.integers(0, 2, size=tiles[i].size, dtype=np.uint8)
        pn = (pn * 2 - 1).astype(np.float32)  # {-1,+1}
        flat = B.ravel()
        v = flat[tiles[i]]
        score = np.sum(v * pn)
        bits.append(1 if score >= 0 else 0)

    return np.array(bits, dtype=np.uint8)

def bits_from_binary_image(bw_np):
    return bw_np.flatten().astype(np.uint8)

def binary_image_from_bits(bits, shape):
    return (bits.reshape(shape) * 255).astype(np.uint8)

# =============== Simple Attacks ===============

def jpeg_compress(np_img_rgb, quality=50):
    pil = Image.fromarray(np_img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))

def small_rotation(np_img_rgb, deg=2.0):
    h, w, _ = np_img_rgb.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    rotated = cv2.warpAffine(np_img_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def small_crop(np_img_rgb, crop_ratio=0.95):
    h, w, _ = np_img_rgb.shape
    nh, nw = int(h * crop_ratio), int(w * crop_ratio)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    crop = np_img_rgb[y0:y0+nh, x0:x0+nw, :]
    # Place back into original canvas without resizing, pad with reflect to preserve original size
    canvas = np_img_rgb.copy()
    canvas[:] = cv2.copyMakeBorder(crop, y0, h - y0 - nh, x0, w - x0 - nw, cv2.BORDER_REFLECT)
    return canvas

# =============== Streamlit UI ===============

st.set_page_config(page_title="DWT Watermarking Demo", page_icon="💧", layout="wide")

st.markdown("""
<style>
/* Simple polish */
.block-container {padding-top: 1.5rem;}
.sidebar .sidebar-content {background: #0f172a12;}
.big-title {font-size: 2rem; font-weight: 800; margin-bottom: .25rem;}
.subtle {opacity: .8; margin-bottom: 1rem;}
.thumb {border-radius: 10px; border: 1px solid #eee;}
.param-card {padding: 1rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">💧 DWT Watermarking (Blind) — Streamlit Frontend</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Color host • Mono watermark • No resizing • Blind extraction with secret key • Robustness demo</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Parameters")
    mode = st.radio("Mode", ["Embed watermark", "Extract watermark only"])
    wavelet = st.selectbox("Wavelet", ["haar", "db2", "db4", "sym4"], index=0)
    level = st.slider("DWT level", 1, 3, 1)
    band = st.selectbox("Subband", ["HL", "LH", "HH"], index=0)
    alpha = st.slider("Embedding strength (α)", 1.0, 20.0, 6.0, 0.5)
    secret_key = st.text_input("Secret key (for RNG)", value="my-secret-key-123")
    st.caption("Keep the same key for embed & extract. Changing it will break extraction.")

    st.divider()
    st.subheader("🧪 Robustness Attacks (Optional)")
    do_jpeg = st.checkbox("JPEG compression", value=True)
    quality = st.slider("JPEG quality", 10, 95, 60)
    do_rot = st.checkbox("Small rotation", value=True)
    deg = st.slider("Rotation degrees", -5, 5, 2)
    do_crop = st.checkbox("Small crop", value=True)
    crop_ratio = st.slider("Crop keep ratio", 0.85, 0.99, 0.95)

colL, colR = st.columns([1, 1])

if mode == "Embed watermark":
    with colL:
        st.subheader("1) Upload Images")
        host_file = st.file_uploader("Host image (color)", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=False)
        wm_file = st.file_uploader("Watermark image (mono)", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=False)

        bin_method = st.selectbox("Binarize watermark", ["otsu (auto)", "fixed threshold 127"], index=0)
        st.caption("We embed watermark bits via spread-spectrum; host/watermark are never resized.")

    proceed = False
    if host_file and wm_file:
        host_np = ensure_color(pil_to_np_uint8(Image.open(host_file).convert("RGB")))
        wm_gray = ensure_gray(pil_to_np_uint8(Image.open(wm_file).convert("L")))
        wm_bin = to_binary(wm_gray, method="otsu" if bin_method.startswith("otsu") else "fixed")
        wm_bits = bits_from_binary_image(wm_bin)

        with colL:
            st.image(host_np, caption=f"Host ({host_np.shape[1]}×{host_np.shape[0]})", use_column_width=True)
            st.image((wm_bin * 255).astype(np.uint8), clamp=True, caption=f"Watermark binary ({wm_bin.shape[1]}×{wm_bin.shape[0]})", use_column_width=True)
        proceed = True

    if proceed:
        with colR:
            st.subheader("2) Embed")
            if st.button("🔐 Embed watermark"):
                try:
                    wm_rgb = embed_spread_spectrum(
                        host_np, wm_bits, key=secret_key,
                        alpha=alpha, wavelet=wavelet, level=level, band=band
                    )
                    # Metrics
                    yp = psnr(host_np, wm_rgb)
                    # SSIM on luminance only for stability
                    s = ssim(cv2.cvtColor(host_np, cv2.COLOR_RGB2GRAY),
                             cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2GRAY))

                    st.success(f"Watermark embedded. PSNR: {yp:.2f} dB | SSIM (Y): {s:.4f}")
                    st.image(wm_rgb, caption="Watermarked image", use_column_width=True)

                    # Optional attacks
                    attacked = wm_rgb.copy()
                    if do_rot: attacked = small_rotation(attacked, deg)
                    if do_crop: attacked = small_crop(attacked, crop_ratio)
                    if do_jpeg: attacked = jpeg_compress(attacked, quality)

                    st.subheader("3) (Optional) Apply Attacks")
                    st.caption("Use sidebar toggles to simulate compression, small rotation, or small crop.")
                    st.image(attacked, caption="Attacked image (for extraction test)", use_column_width=True)

                    # Provide downloads
                    st.download_button(
                        "⬇️ Download watermarked (PNG)",
                        data=bytes_from_image(wm_rgb, fmt="PNG"),
                        file_name="watermarked.png",
                        mime="image/png"
                    )
                    st.download_button(
                        "⬇️ Download attacked (PNG)",
                        data=bytes_from_image(attacked, fmt="PNG"),
                        file_name="attacked.png",
                        mime="image/png"
                    )

                    st.subheader("4) Quick Extract Test (from attacked preview)")
                    nbits = wm_bits.size
                    bits_out = extract_spread_spectrum(attacked, nbits, key=secret_key,
                                                       wavelet=wavelet, level=level, band=band)
                    wm_rec = binary_image_from_bits(bits_out, wm_bin.shape)
                    # Normalized correlation
                    num = np.sum((wm_rec//255) * wm_bin)
                    den = max(1, np.sum(wm_bin))
                    ncc = num / den
                    st.image(wm_rec, clamp=True, caption=f"Extracted watermark (from attacked). NCC (hit-rate): {ncc:.4f}", use_column_width=True)

                    st.download_button(
                        "⬇️ Download extracted watermark (PNG)",
                        data=bytes_from_image(wm_rec, fmt="PNG"),
                        file_name="extracted_watermark.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Embedding failed: {e}")

else:
    with colL:
        st.subheader("Upload for Extraction")
        wmimg_file = st.file_uploader("Watermarked image (only this is needed)", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=False)
        st.caption("This is a *blind* extractor: original host is not required.")
        # Let user specify watermark shape if they know it (otherwise try to guess square)
        cols = st.columns(2)
        with cols[0]:
            wm_h = st.number_input("Watermark height (px)", min_value=1, value=64)
        with cols[1]:
            wm_w = st.number_input("Watermark width (px)", min_value=1, value=64)

        if wmimg_file:
            wm_rgb = ensure_color(pil_to_np_uint8(Image.open(wmimg_file).convert("RGB")))
            st.image(wm_rgb, caption=f"Loaded ({wm_rgb.shape[1]}×{wm_rgb.shape[0]})", use_column_width=True)

    with colR:
        st.subheader("Extract")
        if wmimg_file and st.button("🔎 Extract watermark"):
            try:
                nbits = int(wm_h * wm_w)
                bits = extract_spread_spectrum(wm_rgb, nbits, key=secret_key,
                                               wavelet=wavelet, level=level, band=band)
                wm_rec = binary_image_from_bits(bits, (int(wm_h), int(wm_w)))
                st.image(wm_rec, clamp=True, caption="Extracted watermark (binary)", use_column_width=True)
                st.download_button(
                    "⬇️ Download extracted watermark (PNG)",
                    data=bytes_from_image(wm_rec, fmt="PNG"),
                    file_name="extracted_watermark.png",
                    mime="image/png"
                )
                st.info("If your watermark looks noisy, confirm the exact watermark size and that the key, wavelet, level, and subband match those used during embedding.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# =============== Footer ===============
st.markdown("---")
st.caption("Tip: For best robustness, keep α modest (e.g., 5–8) and embed in HL/LH at level 1–2. Use the same key/params for extraction.")
