"""
VeriFrame — Invisible QR Code Layer (Layer C)

Embeds a QR code as a near-invisible high-frequency pattern
directly into the image pixels. The QR is:
  - Invisible to the human eye (alpha ~2-4 out of 255)
  - Survives mild JPEG compression and screenshot
  - Extractable by amplifying high-frequency residual

This satisfies requirement #4:
  "שימוש במניפולציות עיבוד תמונה עדינות המדמות QR Code
   או ברקוד ויזואלי שאינו נראה לעין"
"""

import cv2
import numpy as np
import json
import hashlib

try:
    import qrcode
    HAS_QR = True
except ImportError:
    HAS_QR = False


def _generate_qr_matrix(data: str, size: int) -> np.ndarray:
    """Generate a QR code as a binary matrix resized to (size, size)."""
    if not HAS_QR:
        return None
    qr = qrcode.QRCode(
        version=1, box_size=1, border=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
    )
    qr.add_data(data)
    qr.make(fit=True)
    matrix = qr.make_image(fill_color="black", back_color="white")
    arr = np.array(matrix.convert("L"))
    # Resize to target size
    resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)
    # Normalize to -1/+1
    return (resized.astype(np.float64) / 255.0) * 2.0 - 1.0  # -1 = black, +1 = white


def embed_invisible_qr(image: np.ndarray, secret: str,
                       password: str = "", alpha: float = 3.0) -> np.ndarray:
    """
    Embed an invisible QR code into the image.

    The QR pattern is applied as a very faint intensity modulation
    on the blue channel (least perceptible to human eyes).
    A pseudo-random scramble makes it harder to detect without the password.

    Args:
        image: BGR image (numpy array)
        secret: text to encode in the QR
        password: password for scrambling
        alpha: embedding strength (2-5 typical, higher = more robust but more visible)

    Returns:
        Modified image with invisible QR embedded
    """
    if not HAS_QR or image is None:
        return image

    h, w = image.shape[:2]
    qr_size = min(h, w)

    # Generate QR data payload
    qr_payload = json.dumps({
        "v": "VeriFrame",
        "msg": secret[:64],
        "h": hashlib.sha256(secret.encode()).hexdigest()[:12],
    }, ensure_ascii=False)

    qr_matrix = _generate_qr_matrix(qr_payload, qr_size)
    if qr_matrix is None:
        return image

    # Resize QR to image dimensions
    if qr_matrix.shape[0] != h or qr_matrix.shape[1] != w:
        qr_matrix = cv2.resize(qr_matrix, (w, h), interpolation=cv2.INTER_NEAREST)

    # Scramble with password-based permutation for security
    if password:
        seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed ^ 0xDEADBEEF)
        # Permute rows and columns
        row_perm = rng.permutation(h)
        col_perm = rng.permutation(w)
        qr_matrix = qr_matrix[row_perm, :][:, col_perm]

    # Embed in the blue channel (least visible to human eyes)
    result = image.copy().astype(np.float64)
    result[:, :, 0] += qr_matrix * alpha  # Blue channel

    # Also add a faint version to green for redundancy
    result[:, :, 1] += qr_matrix * (alpha * 0.3)

    return np.clip(result, 0, 255).astype(np.uint8)


def extract_invisible_qr(original: np.ndarray, watermarked: np.ndarray,
                          password: str = "",
                          amplify: float = 30.0) -> np.ndarray:
    """
    Extract the invisible QR code by computing the difference
    between original and watermarked images and amplifying it.

    Returns a visualization of the extracted QR pattern.
    """
    if original is None or watermarked is None:
        return None

    h, w = watermarked.shape[:2]
    original_resized = cv2.resize(original, (w, h))

    # Compute difference on blue channel
    diff_b = watermarked[:, :, 0].astype(np.float64) - original_resized[:, :, 0].astype(np.float64)

    # Un-scramble if password provided
    if password:
        seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed ^ 0xDEADBEEF)
        row_perm = rng.permutation(h)
        col_perm = rng.permutation(w)
        # Inverse permutation
        inv_row = np.argsort(row_perm)
        inv_col = np.argsort(col_perm)
        diff_b = diff_b[inv_row, :][:, inv_col]

    # Amplify
    amplified = diff_b * amplify
    amplified = ((amplified - amplified.min()) / max(amplified.max() - amplified.min(), 1) * 255)
    amplified = amplified.astype(np.uint8)

    # Threshold to make QR-like pattern
    _, binary = cv2.threshold(amplified, 128, 255, cv2.THRESH_BINARY)

    # Create colored visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    viz[:, :, 0] = amplified  # Blue = raw pattern
    viz[:, :, 1] = binary * 0.3  # Green hint for binary
    viz[:, :, 2] = binary * 0.1

    return viz


def extract_qr_blind(watermarked: np.ndarray, password: str = "") -> np.ndarray:
    """
    Blind extraction — attempt to reveal the QR without the original image.
    Uses high-pass filtering to isolate the embedded pattern.
    """
    if watermarked is None:
        return None

    h, w = watermarked.shape[:2]
    blue = watermarked[:, :, 0].astype(np.float64)

    # High-pass filter to isolate QR pattern
    blurred = cv2.GaussianBlur(blue, (7, 7), 0)
    residual = blue - blurred

    # Un-scramble if password provided
    if password:
        seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed ^ 0xDEADBEEF)
        row_perm = rng.permutation(h)
        col_perm = rng.permutation(w)
        inv_row = np.argsort(row_perm)
        inv_col = np.argsort(col_perm)
        residual = residual[inv_row, :][:, inv_col]

    # Normalize and amplify
    residual = ((residual - residual.min()) / max(residual.max() - residual.min(), 1) * 255)
    viz = residual.astype(np.uint8)

    # Apply colormap for visual impact
    colored = cv2.applyColorMap(viz, cv2.COLORMAP_INFERNO)

    # Add label
    label_h = 30
    result = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    result[label_h:, :] = colored
    cv2.putText(result, "Invisible QR — Extracted Pattern",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return result
