import cv2
import numpy as np
from scipy.fftpack import dct, idct
import hashlib

try:
    import reedsolo
    HAS_ECC = True
except ImportError:
    HAS_ECC = False


# ─────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────

def _password_to_seed(password: str) -> int:
    key = hashlib.sha256(password.encode()).digest()
    return int.from_bytes(key[:8], "big") % (2**31)


def _password_to_key(password: str) -> bytes:
    return hashlib.sha256(password.encode()).digest()


def _adaptive_strength(h: int, w: int, base: float = 25.0) -> float:
    ref_area = 512 * 512
    area = h * w
    return max(15.0, min(40.0, base * (ref_area / area) ** 0.3))


def _apply_ecc(data: bytes) -> bytes:
    if HAS_ECC:
        rs = reedsolo.RSCodec(16)
        return bytes(rs.encode(data))
    return data


def _remove_ecc(data: bytes):
    if HAS_ECC:
        rs = reedsolo.RSCodec(16)
        try:
            decoded, _, _ = rs.decode(data)
            return bytes(decoded)
        except Exception:
            return None
    return data


def _bits_to_message(bits: np.ndarray):
    pad = (8 - len(bits) % 8) % 8
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    raw_bytes = np.packbits(bits).tobytes()

    decoded = _remove_ecc(raw_bytes)
    if decoded is None:
        decoded = raw_bytes

    try:
        text = decoded.decode("utf-8", errors="ignore")
        if "####" in text:
            return text.split("####")[0]
    except Exception:
        pass
    return None


def _prepare_bits(secret: str) -> np.ndarray:
    data = (secret + "####").encode("utf-8")
    data_ecc = _apply_ecc(data)
    return np.unpackbits(np.frombuffer(data_ecc, dtype=np.uint8))


# ─────────────────────────────────────────
#  Layer A — DCT Mid-Frequency Embedding
#  Survives: JPEG/MP4 compression, re-encoding
# ─────────────────────────────────────────

_MID_FREQ = [(2, 1), (1, 2), (0, 3), (1, 3), (2, 2),
             (3, 1), (4, 0), (3, 0), (2, 0), (0, 4)]


def _embed_layer_a(y: np.ndarray, bits: np.ndarray, seed: int,
                   strength: float = 25.0) -> np.ndarray:
    h, w = y.shape
    y_f = y.astype(np.float64)
    rng = np.random.RandomState(seed)

    blocks_h, blocks_w = h // 8, w // 8
    block_order = rng.permutation(blocks_h * blocks_w)

    for i, block_idx in enumerate(block_order):
        if i >= len(bits):
            break
        r = (block_idx // blocks_w) * 8
        c = (block_idx % blocks_w) * 8

        block = y_f[r:r+8, c:c+8]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

        pos = _MID_FREQ[i % len(_MID_FREQ)]
        D[pos] = strength if bits[i] == 1 else -strength

        block_back = idct(idct(D, axis=1, norm="ortho"), axis=0, norm="ortho")
        y_f[r:r+8, c:c+8] = block_back

    return np.clip(y_f, 0, 255).astype(np.uint8)


def _extract_layer_a(y: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w = y.shape
    y_f = y.astype(np.float64)
    rng = np.random.RandomState(seed)

    blocks_h, blocks_w = h // 8, w // 8
    block_order = rng.permutation(blocks_h * blocks_w)

    bits = []
    for i, block_idx in enumerate(block_order[:num_bits]):
        r = (block_idx // blocks_w) * 8
        c = (block_idx % blocks_w) * 8

        block = y_f[r:r+8, c:c+8]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
        pos = _MID_FREQ[i % len(_MID_FREQ)]
        bits.append(1 if D[pos] > 0 else 0)

    return np.array(bits, dtype=np.uint8)


def _extract_layer_a_soft(y: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    """Return raw DCT coefficient values (floats) for temporal averaging."""
    h, w = y.shape
    y_f = y.astype(np.float64)
    rng = np.random.RandomState(seed)

    blocks_h, blocks_w = h // 8, w // 8
    block_order = rng.permutation(blocks_h * blocks_w)

    values = []
    for i, block_idx in enumerate(block_order[:num_bits]):
        r = (block_idx // blocks_w) * 8
        c = (block_idx % blocks_w) * 8

        block = y_f[r:r+8, c:c+8]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
        pos = _MID_FREQ[i % len(_MID_FREQ)]
        values.append(D[pos])

    return np.array(values, dtype=np.float64)


def temporal_average_decode(soft_bits_list: list) -> np.ndarray:
    """Average soft values across frames, then threshold to binary."""
    if not soft_bits_list:
        return np.array([], dtype=np.uint8)
    min_len = min(len(s) for s in soft_bits_list)
    stacked = np.array([s[:min_len] for s in soft_bits_list])
    averaged = np.mean(stacked, axis=0)
    return (averaged > 0).astype(np.uint8)


# ─────────────────────────────────────────
#  Layer B — Spread Spectrum
#  Survives: additive noise, brightness changes, screen recording
# ─────────────────────────────────────────

LAYER_B_BITS = 256


def _embed_layer_b(cr: np.ndarray, bits: np.ndarray, seed: int,
                   alpha: float = 1.2) -> np.ndarray:
    h, w = cr.shape
    cr_f = cr.astype(np.float64)

    for i, bit in enumerate(bits):
        bit_rng = np.random.RandomState((seed ^ (i * 2654435761)) % (2**31))
        pn = bit_rng.choice([-1.0, 1.0], size=(h, w))
        cr_f += alpha * (1 if bit == 1 else -1) * pn

    return np.clip(cr_f, 0, 255).astype(np.uint8)


def _extract_layer_b(cr: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    h, w = cr.shape
    cr_f = cr.astype(np.float64)

    bits = []
    for i in range(num_bits):
        bit_rng = np.random.RandomState((seed ^ (i * 2654435761)) % (2**31))
        pn = bit_rng.choice([-1.0, 1.0], size=(h, w))
        bits.append(1 if np.dot(cr_f.ravel(), pn.ravel()) > 0 else 0)

    return np.array(bits, dtype=np.uint8)


def _extract_layer_b_soft(cr: np.ndarray, num_bits: int, seed: int) -> np.ndarray:
    """Return raw correlation values (floats) for temporal averaging."""
    h, w = cr.shape
    cr_f = cr.astype(np.float64)

    values = []
    for i in range(num_bits):
        bit_rng = np.random.RandomState((seed ^ (i * 2654435761)) % (2**31))
        pn = bit_rng.choice([-1.0, 1.0], size=(h, w))
        values.append(np.dot(cr_f.ravel(), pn.ravel()))

    return np.array(values, dtype=np.float64)


# ─────────────────────────────────────────
#  Majority vote — fuse Layer A + Layer B
# ─────────────────────────────────────────

def _fuse_bits(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    length = min(len(bits_a), len(bits_b))
    fused = np.where(bits_a[:length] + bits_b[:length] >= 1, 1, 0).astype(np.uint8)
    if len(bits_a) > length:
        fused = np.concatenate([fused, bits_a[length:]])
    return fused


# ─────────────────────────────────────────
#  Sync Markers — geometric reference points
#  Survives: rotation, scaling, phone screen recording
# ─────────────────────────────────────────

_SYNC_SIZE = 16

def _generate_sync_pattern(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed ^ 0xCAFEBABE)
    pattern = np.zeros((_SYNC_SIZE, _SYNC_SIZE), dtype=np.float64)
    for r in range(4, 12):
        for c in range(4, 12):
            pattern[r, c] = rng.choice([-30.0, 30.0])
    return pattern


def _sync_marker_positions(h: int, w: int, margin: int = 32) -> list:
    return [
        (margin, margin),
        (margin, w - margin - _SYNC_SIZE),
        (h - margin - _SYNC_SIZE, margin),
        (h - margin - _SYNC_SIZE, w - margin - _SYNC_SIZE),
    ]


def _embed_sync_markers(y: np.ndarray, seed: int,
                        strength: float = 35.0) -> np.ndarray:
    h, w = y.shape
    if h < 128 or w < 128:
        return y
    y_f = y.astype(np.float64)
    pattern = _generate_sync_pattern(seed)
    positions = _sync_marker_positions(h, w)

    for (r, c) in positions:
        if r + _SYNC_SIZE > h or c + _SYNC_SIZE > w:
            continue
        block = y_f[r:r+_SYNC_SIZE, c:c+_SYNC_SIZE]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
        D += pattern * (strength / 30.0)
        block_back = idct(idct(D, axis=1, norm="ortho"), axis=0, norm="ortho")
        y_f[r:r+_SYNC_SIZE, c:c+_SYNC_SIZE] = block_back

    return np.clip(y_f, 0, 255).astype(np.uint8)


def _detect_sync_markers(y: np.ndarray, seed: int,
                         search_margin: int = 64) -> list:
    """Detect sync markers and return (expected, detected) position pairs."""
    h, w = y.shape
    if h < 128 or w < 128:
        return []

    y_f = y.astype(np.float64)
    pattern = _generate_sync_pattern(seed)
    expected_positions = _sync_marker_positions(h, w)
    pairs = []

    for (er, ec) in expected_positions:
        best_corr = -1e9
        best_pos = (er, ec)

        r_start = max(0, er - search_margin)
        r_end = min(h - _SYNC_SIZE, er + search_margin)
        c_start = max(0, ec - search_margin)
        c_end = min(w - _SYNC_SIZE, ec + search_margin)

        for r in range(r_start, r_end, 4):
            for c in range(c_start, c_end, 4):
                block = y_f[r:r+_SYNC_SIZE, c:c+_SYNC_SIZE]
                D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
                corr = np.sum(D * pattern)
                if corr > best_corr:
                    best_corr = corr
                    best_pos = (r, c)

        pairs.append(((er, ec), best_pos, best_corr))

    return pairs


def _apply_geometric_correction(image: np.ndarray, seed: int) -> np.ndarray:
    """Detect sync markers and correct geometric distortion."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    else:
        gray = image

    pairs = _detect_sync_markers(gray, seed)
    if len(pairs) < 3:
        return image

    src_pts = np.float32([p[1] for p in pairs])
    dst_pts = np.float32([p[0] for p in pairs])

    displacement = np.mean(np.abs(src_pts - dst_pts))
    if displacement < 2.0:
        return image

    if len(pairs) >= 4:
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    else:
        M, _ = cv2.estimateAffinePartial2D(src_pts[:3], dst_pts[:3])

    if M is None:
        return image

    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────
#  Trust Score — uses both layers
# ─────────────────────────────────────────

def compute_trust_score(image_path: str, password: str = "") -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"score": 0, "found": False, "message": "",
                "score_a": 0, "score_b": 0}

    seed = _password_to_seed(password)
    corrected = _apply_geometric_correction(image, seed)
    ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    h, w = y.shape
    max_bits = (h // 8) * (w // 8)

    bits_a = _extract_layer_a(y, max_bits, seed)
    msg = _bits_to_message(bits_a)

    if msg:
        # Layer A score: average coefficient strength
        rng = np.random.RandomState(seed)
        block_order = rng.permutation((h // 8) * (w // 8))
        strengths = []
        y_f = y.astype(np.float64)
        sample_count = min(200, max_bits)
        for i, block_idx in enumerate(block_order[:sample_count]):
            r = (block_idx // (w // 8)) * 8
            c = (block_idx % (w // 8)) * 8
            block = y_f[r:r+8, c:c+8]
            D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
            pos = _MID_FREQ[i % len(_MID_FREQ)]
            strengths.append(abs(D[pos]))
        strength_used = _adaptive_strength(h, w)
        score_a = min(100, int((float(np.mean(strengths)) / strength_used) * 100))

        # Layer B score: correlation confidence
        bits_b_soft = _extract_layer_b_soft(cr, min(64, LAYER_B_BITS), seed)
        avg_corr = float(np.mean(np.abs(bits_b_soft)))
        max_corr = float(h * w * 1.2)
        score_b = min(100, int((avg_corr / max_corr) * 300))

        score = int(0.6 * score_a + 0.4 * score_b)
        return {"score": score, "found": True, "message": msg,
                "score_a": score_a, "score_b": score_b}

    return {"score": 0, "found": False, "message": "",
            "score_a": 0, "score_b": 0}


# ─────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────

def encode_image(image_path: str, secret_data: str,
                 output_path: str, password: str = "") -> str:
    image = cv2.imread(image_path)
    if image is None:
        return "❌ Cannot read image"

    seed = _password_to_seed(password)
    bits = _prepare_bits(secret_data)

    h, w = image.shape[:2]
    max_bits = (h // 8) * (w // 8)
    if len(bits) > max_bits:
        return f"❌ Data too long ({len(bits)} bits, max {max_bits})"

    strength = _adaptive_strength(h, w)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Sync markers first (on Y channel)
    ycrcb[:, :, 0] = _embed_sync_markers(ycrcb[:, :, 0], seed, strength + 10)

    # Layer A on Y (luma) — DCT frequency domain
    ycrcb[:, :, 0] = _embed_layer_a(ycrcb[:, :, 0], bits, seed, strength)

    # Layer B on Cr (chroma) — Spread Spectrum
    short_bits = bits[:min(LAYER_B_BITS, len(bits))]
    ycrcb[:, :, 1] = _embed_layer_b(ycrcb[:, :, 1], short_bits, seed)

    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Layer C — Invisible QR code (spatial domain)
    try:
        from invisible_qr import embed_invisible_qr
        result = embed_invisible_qr(result, secret_data, password, alpha=3.0)
    except Exception:
        pass  # Layer C is optional — graceful fallback

    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return "✅ Watermark embedded successfully"


def decode_image(image_path: str, password: str = "") -> str:
    image = cv2.imread(image_path)
    if image is None:
        return "❌ Cannot read image"

    seed = _password_to_seed(password)

    # Geometric correction via sync markers
    corrected = _apply_geometric_correction(image, seed)
    ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    h, w = y.shape
    max_bits = (h // 8) * (w // 8)

    # Try Layer A alone
    bits_a = _extract_layer_a(y, max_bits, seed)
    msg = _bits_to_message(bits_a)
    if msg:
        return f"✅ {msg}"

    # Try fusing Layer A + Layer B
    bits_b = _extract_layer_b(cr, min(LAYER_B_BITS, len(bits_a)), seed)
    fused = _fuse_bits(bits_a[:len(bits_b)], bits_b)
    msg_fused = _bits_to_message(fused)
    if msg_fused:
        return f"✅ {msg_fused}"

    # Fallback: Layer B alone
    msg_b = _bits_to_message(bits_b)
    if msg_b:
        return f"✅ {msg_b}"

    return "❌ No watermark found (wrong password or not watermarked)"


def remove_watermark(image_path: str, output_path: str,
                     password: str = "") -> str:
    image = cv2.imread(image_path)
    if image is None:
        return "❌ Cannot read image"

    seed = _password_to_seed(password)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Remove Layer A (DCT)
    y = ycrcb[:, :, 0].astype(np.float64)
    h, w = y.shape
    rng = np.random.RandomState(seed)
    block_order = rng.permutation((h // 8) * (w // 8))

    for i, block_idx in enumerate(block_order):
        r = (block_idx // (w // 8)) * 8
        c = (block_idx % (w // 8)) * 8
        block = y[r:r+8, c:c+8]
        D = dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")
        pos = _MID_FREQ[i % len(_MID_FREQ)]
        D[pos] *= 0.05
        block_back = idct(idct(D, axis=1, norm="ortho"), axis=0, norm="ortho")
        y[r:r+8, c:c+8] = block_back
    ycrcb[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)

    # Remove Layer B (Spread Spectrum) — subtract estimated PN contribution
    cr = ycrcb[:, :, 1].astype(np.float64)
    bits_b = _extract_layer_b(ycrcb[:, :, 1], LAYER_B_BITS, seed)
    for i, bit in enumerate(bits_b):
        bit_rng = np.random.RandomState((seed ^ (i * 2654435761)) % (2**31))
        pn = bit_rng.choice([-1.0, 1.0], size=(h, w))
        cr -= 1.2 * (1 if bit == 1 else -1) * pn
    ycrcb[:, :, 1] = np.clip(cr, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, result)
    return "✅ Watermark removed"
