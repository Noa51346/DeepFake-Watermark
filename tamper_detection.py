import cv2
import numpy as np
from scipy.fftpack import dct

from watermark_logic import (
    _password_to_seed, _MID_FREQ, _adaptive_strength,
    _apply_geometric_correction, _prepare_bits, _extract_layer_a,
    _bits_to_message,
)


# ─────────────────────────────────────────
#  Tamper Detection Heatmap
# ─────────────────────────────────────────

def generate_tamper_map(image_path: str, password: str = "") -> dict:
    """
    Generate a tamper detection heatmap by comparing each watermarked
    DCT block's coefficient against its expected sign and magnitude.
    Only checks blocks that actually contain watermark data.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"overlay": None, "raw_map": None, "tamper_pct": 0}

    seed = _password_to_seed(password)
    corrected = _apply_geometric_correction(image, seed)
    ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float64)
    h, w = y.shape

    blocks_h, blocks_w = h // 8, w // 8
    strength = _adaptive_strength(h, w)
    max_bits = blocks_h * blocks_w

    # Step 1: extract bits to find how many were embedded
    bits = _extract_layer_a(ycrcb[:, :, 0], max_bits, seed)
    msg = _bits_to_message(bits)

    # Estimate watermark length: if we found a message, compute actual length
    # Otherwise check a generous portion of blocks
    if msg:
        expected_bits = _prepare_bits(msg)
        num_wm_blocks = len(expected_bits)
    else:
        num_wm_blocks = min(max_bits, 2000)

    # Step 2: for each watermarked block, measure deviation
    rng = np.random.RandomState(seed)
    block_order = rng.permutation(blocks_h * blocks_w)

    deviation_map = np.full((blocks_h, blocks_w), -1.0, dtype=np.float64)

    for i, block_idx in enumerate(block_order[:num_wm_blocks]):
        br = block_idx // blocks_w
        bc = block_idx % blocks_w
        r = br * 8
        c = bc * 8

        block = y[r:r+8, c:c+8]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
        pos = _MID_FREQ[i % len(_MID_FREQ)]
        coeff = D[pos]

        # Deviation: how far is |coeff| from expected strength?
        # Perfect watermark: |coeff| ≈ strength
        # Tampered: |coeff| << strength or sign flipped
        mag_deviation = abs(abs(coeff) - strength) / strength

        # Sign check: if we know expected bits, check sign consistency
        if msg and i < len(expected_bits):
            expected_sign = 1 if expected_bits[i] == 1 else -1
            actual_sign = 1 if coeff > 0 else -1
            sign_match = (expected_sign == actual_sign)
            # Sign mismatch is a strong tamper signal
            if not sign_match:
                deviation_map[br, bc] = min(1.0, 0.7 + mag_deviation * 0.3)
            else:
                deviation_map[br, bc] = min(1.0, mag_deviation * 0.5)
        else:
            deviation_map[br, bc] = min(1.0, mag_deviation)

    # Replace unchecked blocks (-1) with 0 (neutral)
    unchecked = deviation_map < 0
    deviation_map[unchecked] = 0.0

    # Upscale to image resolution
    heatmap_full = cv2.resize(deviation_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap: green (intact) → yellow → red (tampered)
    heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_color[:, :, 1] = ((1.0 - heatmap_full) * 255).astype(np.uint8)
    heatmap_color[:, :, 2] = (heatmap_full * 255).astype(np.uint8)
    mid_mask = (heatmap_full > 0.3) & (heatmap_full < 0.7)
    heatmap_color[mid_mask, 0] = 80

    # Alpha blend with original
    overlay = cv2.addWeighted(corrected, 0.55, heatmap_color, 0.45, 0)

    # Tamper percentage (only among watermarked blocks)
    wm_deviations = deviation_map[~unchecked]
    tamper_threshold = 0.4
    if len(wm_deviations) > 0:
        tamper_pct = float(np.sum(wm_deviations > tamper_threshold)) / len(wm_deviations) * 100
    else:
        tamper_pct = 0.0

    # Draw bounding boxes of tampered regions on overlay
    overlay_annotated = overlay.copy()
    summary = get_tamper_summary(deviation_map, tamper_threshold)
    for region in summary.get("regions", [])[:10]:
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(overlay_annotated, (rx, ry), (rx + rw, ry + rh),
                      (0, 0, 255), 2)
        label = f"{int(rw * rh / max(1, w * h) * 100)}%"
        cv2.putText(overlay_annotated, label, (rx + 4, ry + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return {
        "overlay": overlay,
        "overlay_annotated": overlay_annotated,
        "raw_map": deviation_map,
        "tamper_pct": round(tamper_pct, 1),
        "summary": summary,
    }


def get_tamper_summary(deviation_map: np.ndarray, threshold: float = 0.4) -> dict:
    """Get summary statistics and bounding boxes of tampered regions."""
    if deviation_map is None:
        return {"tamper_pct": 0, "severity": "none", "regions": []}

    # Only consider blocks that were checked (value > 0 or exactly 0 from watermark)
    checked = deviation_map >= 0
    if not np.any(checked):
        return {"tamper_pct": 0, "severity": "pristine", "regions": []}

    checked_vals = deviation_map[checked]
    tamper_pct = float(np.sum(checked_vals > threshold)) / max(1, len(checked_vals)) * 100

    # Find contours of tampered regions
    tampered_mask = (deviation_map > threshold).astype(np.uint8) * 255
    tampered_resized = cv2.resize(
        tampered_mask,
        (deviation_map.shape[1] * 8, deviation_map.shape[0] * 8),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(
        tampered_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > 200:
            regions.append({"x": x, "y": y, "w": bw, "h": bh})

    # Sort by area descending
    regions.sort(key=lambda r: r["w"] * r["h"], reverse=True)

    if tamper_pct < 5:
        severity = "pristine"
    elif tamper_pct < 15:
        severity = "minor"
    elif tamper_pct < 40:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "tamper_pct": round(tamper_pct, 1),
        "severity": severity,
        "regions": regions[:10],
        "num_regions": len(regions),
    }


# ─────────────────────────────────────────
#  Forensic Visualization — Frequency Spectrum
# ─────────────────────────────────────────

def generate_frequency_spectrum(image_path: str) -> np.ndarray:
    """Generate 2D FFT magnitude spectrum visualization."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    f = np.fft.fft2(image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log10(np.abs(fshift) + 1)

    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    return colored


def generate_before_after_spectrum(original_path: str,
                                   watermarked_path: str) -> np.ndarray:
    """Side-by-side frequency spectrum + difference map."""
    spec_orig = generate_frequency_spectrum(original_path)
    spec_wm = generate_frequency_spectrum(watermarked_path)

    if spec_orig is None or spec_wm is None:
        return None

    h = min(spec_orig.shape[0], spec_wm.shape[0])
    w = min(spec_orig.shape[1], spec_wm.shape[1])
    spec_orig = cv2.resize(spec_orig, (w, h))
    spec_wm = cv2.resize(spec_wm, (w, h))

    diff = cv2.absdiff(spec_orig, spec_wm)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff
    diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)

    label_h = 30
    result = np.zeros((h + label_h, w * 3 + 20, 3), dtype=np.uint8)

    result[label_h:label_h+h, 0:w] = spec_orig
    result[label_h:label_h+h, w+10:2*w+10] = spec_wm
    result[label_h:label_h+h, 2*w+20:3*w+20] = diff_colored

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Original", (10, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(result, "Watermarked", (w+20, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(result, "Difference", (2*w+30, 22), font, 0.6, (0, 100, 255), 1)

    return result


def generate_dct_block_viz(image_path: str, block_row: int = 10,
                           block_col: int = 10) -> np.ndarray:
    """Visualize DCT coefficients of a specific 8x8 block."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    r, c = block_row * 8, block_col * 8
    if r + 8 > image.shape[0] or c + 8 > image.shape[1]:
        return None

    block = image[r:r+8, c:c+8].astype(np.float64)
    D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

    cell_size = 40
    viz = np.zeros((8 * cell_size, 8 * cell_size, 3), dtype=np.uint8)

    max_val = max(np.abs(D).max(), 1)
    for i in range(8):
        for j in range(8):
            val = D[i, j]
            normalized = val / max_val

            if normalized > 0:
                color = (0, 0, int(normalized * 255))
            else:
                color = (int(-normalized * 255), 0, 0)

            y1, y2 = i * cell_size, (i+1) * cell_size
            x1, x2 = j * cell_size, (j+1) * cell_size
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (40, 40, 40), 1)

            if (i, j) in _MID_FREQ:
                cv2.circle(viz, (x1 + cell_size//2, y1 + cell_size//2),
                          3, (0, 255, 255), -1)

            text = f"{val:.0f}"
            cv2.putText(viz, text, (x1+4, y1+cell_size//2+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return viz


# ─────────────────────────────────────────
#  Error Level Analysis (ELA)
# ─────────────────────────────────────────

def generate_ela(image_path: str, quality: int = 90) -> np.ndarray:
    """
    Error Level Analysis — re-save image at given JPEG quality,
    then amplify the difference.  Tampered regions show brighter
    because they were saved at a different quality level.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Re-save as JPEG at specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", image, encode_param)
    resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Compute absolute difference and amplify
    diff = cv2.absdiff(image, resaved)
    scale = 255.0 / max(diff.max(), 1)
    ela = (diff.astype(np.float64) * scale).clip(0, 255).astype(np.uint8)

    # Convert to heatmap for visual clarity
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    # Add title bar
    label_h = 30
    h, w = ela_colored.shape[:2]
    result = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    result[label_h:, :] = ela_colored

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Error Level Analysis (JPEG Q={quality})",
                (10, 22), font, 0.55, (200, 200, 200), 1)

    return result


# ─────────────────────────────────────────
#  Noise Residual Analysis
# ─────────────────────────────────────────

def generate_noise_analysis(image_path: str) -> np.ndarray:
    """
    Extract and visualise the noise residual of an image.
    Tampered regions have different noise patterns than the original.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Denoise with median filter → subtract to get noise
    denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float64)
    noise = gray - denoised

    # Normalise to 0-255 range
    noise_norm = ((noise - noise.min()) / max(noise.max() - noise.min(), 1) * 255)
    noise_u8 = noise_norm.astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(noise_u8, cv2.COLORMAP_TWILIGHT_SHIFTED)

    label_h = 30
    h, w = colored.shape[:2]
    result = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    result[label_h:, :] = colored

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Noise Residual Analysis",
                (10, 22), font, 0.55, (200, 200, 200), 1)

    return result


# ─────────────────────────────────────────
#  Bit-plane Analysis
# ─────────────────────────────────────────

def generate_bitplane(image_path: str, bit: int = 0) -> np.ndarray:
    """
    Extract a specific bit-plane from the image.
    LSB (bit=0) often reveals hidden patterns or edits.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    plane = ((image >> bit) & 1) * 255
    colored = cv2.applyColorMap(plane, cv2.COLORMAP_BONE)

    label_h = 30
    h, w = colored.shape[:2]
    result = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    result[label_h:, :] = colored

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Bit-Plane {bit} (LSB)" if bit == 0 else f"Bit-Plane {bit}",
                (10, 22), font, 0.55, (200, 200, 200), 1)

    return result
