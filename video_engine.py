import os
import cv2
import numpy as np
import subprocess
import shutil

from watermark_logic import (
    _password_to_seed, _prepare_bits, _adaptive_strength,
    _embed_sync_markers, _embed_layer_a, _embed_layer_b,
    _extract_layer_a_soft, _extract_layer_b_soft,
    _apply_geometric_correction,
    _bits_to_message, temporal_average_decode,
    compute_trust_score,
    LAYER_B_BITS,
)


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except Exception:
        return False


HAS_FFMPEG = _has_ffmpeg()


# ─────────────────────────────────────────
#  Video Info
# ─────────────────────────────────────────

def _get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "duration": 0.0,
    }
    if info["fps"] > 0 and info["total_frames"] > 0:
        info["duration"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


# ─────────────────────────────────────────
#  Extract Frames
# ─────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str,
                   every_n: int = 1, progress_callback=None) -> list:
    """
    Extract frames from video to individual image files.
    Uses FFmpeg if available, otherwise OpenCV fallback.
    Returns list of frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    info = _get_video_info(video_path)
    total = max(1, info["total_frames"])
    frame_paths = []

    if HAS_FFMPEG and every_n > 1:
        # FFmpeg: faster for selective frame extraction
        pattern = os.path.join(output_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select=not(mod(n\\,{every_n}))",
            "-vsync", "vfr",
            "-q:v", "1",
            pattern,
            "-y", "-loglevel", "error",
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        # Collect output files
        for f in sorted(os.listdir(output_dir)):
            if f.startswith("frame_") and f.endswith(".png"):
                frame_paths.append(os.path.join(output_dir, f))
    else:
        # OpenCV fallback
        cap = cv2.VideoCapture(video_path)
        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                path = os.path.join(output_dir, f"frame_{saved:06d}.png")
                cv2.imwrite(path, frame)
                frame_paths.append(path)
                saved += 1
            idx += 1
            if progress_callback and total > 0:
                progress_callback(idx / total)
        cap.release()

    return frame_paths


# ─────────────────────────────────────────
#  Extract Audio
# ─────────────────────────────────────────

def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio track from video. Returns True if audio was found."""
    if not HAS_FFMPEG:
        return False
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "copy",
            output_path,
            "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


# ─────────────────────────────────────────
#  Reassemble Video
# ─────────────────────────────────────────

def reassemble_video(frame_dir: str, output_path: str, fps: float,
                     audio_path: str = None) -> str:
    """
    Reassemble frames into video.
    Uses FFmpeg if available (with optional audio), otherwise OpenCV.
    """
    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.startswith("frame_") and f.endswith(".png")
    ])
    if not frame_files:
        return "❌ No frames found"

    if HAS_FFMPEG:
        pattern = os.path.join(frame_dir, "frame_%06d.png")
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", pattern,
        ]
        if audio_path and os.path.exists(audio_path):
            cmd += ["-i", audio_path, "-c:a", "aac", "-shortest"]
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
            "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            return "✅ Video reassembled with FFmpeg"

    # OpenCV fallback (no audio support)
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fname in frame_files:
        frame = cv2.imread(os.path.join(frame_dir, fname))
        if frame is not None:
            writer.write(frame)
    writer.release()
    return "✅ Video reassembled with OpenCV"


# ─────────────────────────────────────────
#  Encode Video (full pipeline)
# ─────────────────────────────────────────

def encode_video(video_path: str, secret_data: str, output_path: str,
                 password: str = "", every_n: int = 1,
                 progress_callback=None) -> str:
    """
    Full video watermarking pipeline:
    1. Extract audio (if FFmpeg available)
    2. Read frames → watermark each → write output
    3. Temporal redundancy: same watermark on every frame in group
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "❌ Cannot open video"

    info = _get_video_info(video_path)
    fps = info["fps"]
    w, h = info["width"], info["height"]
    total = info["total_frames"]

    # Extract audio for later remuxing
    audio_path = None
    if HAS_FFMPEG:
        audio_path = output_path + ".audio.aac"
        has_audio = extract_audio(video_path, audio_path)
        if not has_audio:
            audio_path = None

    seed = _password_to_seed(password)
    bits = _prepare_bits(secret_data)
    strength = _adaptive_strength(h, w)
    short_bits = bits[:min(LAYER_B_BITS, len(bits))]

    # Write watermarked frames to temp video
    temp_out = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = _embed_sync_markers(ycrcb[:, :, 0], seed, strength + 10)
            ycrcb[:, :, 0] = _embed_layer_a(ycrcb[:, :, 0], bits, seed, strength)
            ycrcb[:, :, 1] = _embed_layer_b(ycrcb[:, :, 1], short_bits, seed)
            frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        writer.write(frame)
        frame_idx += 1

        if progress_callback and total > 0:
            progress_callback(frame_idx / total)

    cap.release()
    writer.release()

    # Remux with audio if available
    if audio_path and HAS_FFMPEG:
        cmd = [
            "ffmpeg",
            "-i", temp_out,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path,
            "-y", "-loglevel", "error",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        try:
            os.remove(temp_out)
            os.remove(audio_path)
        except OSError:
            pass
        if result.returncode == 0:
            return "✅ Video watermarked successfully (with audio)"

    # No FFmpeg or no audio — just rename temp to output
    if os.path.exists(temp_out):
        if os.path.exists(output_path) and output_path != temp_out:
            try:
                os.remove(output_path)
            except OSError:
                pass
        os.rename(temp_out, output_path)

    return "✅ Video watermarked successfully"


# ─────────────────────────────────────────
#  Decode Video (temporal averaging)
# ─────────────────────────────────────────

def decode_video(video_path: str, password: str = "",
                 num_sample_frames: int = 10,
                 progress_callback=None) -> dict:
    """
    Decode watermark from video using temporal averaging across frames.
    Temporal averaging gives sqrt(N) improvement in SNR.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"found": False, "message": "", "scores": [],
                "avg_score": 0, "frames_analyzed": 0}

    info = _get_video_info(video_path)
    total = max(1, info["total_frames"])
    seed = _password_to_seed(password)

    # Sample frames evenly across the video
    sample_indices = np.linspace(0, total - 1, num_sample_frames, dtype=int)
    sample_indices = np.unique(sample_indices)

    soft_bits_a = []
    soft_bits_b = []
    per_frame_scores = []

    for idx, frame_num in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ret, frame = cap.read()
        if not ret:
            continue

        corrected = _apply_geometric_correction(frame, seed)
        ycrcb = cv2.cvtColor(corrected, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        h, w = y.shape
        max_bits = (h // 8) * (w // 8)

        vals_a = _extract_layer_a_soft(y, max_bits, seed)
        vals_b = _extract_layer_b_soft(cr, min(64, LAYER_B_BITS), seed)

        soft_bits_a.append(vals_a)
        soft_bits_b.append(vals_b)

        # Per-frame hard decode for scoring
        bits_hard = (vals_a > 0).astype(np.uint8)
        msg_frame = _bits_to_message(bits_hard)
        strength_used = _adaptive_strength(h, w)
        avg_s = float(np.mean(np.abs(vals_a[:200]))) if len(vals_a) >= 200 else 0
        frame_score = min(100, int((avg_s / strength_used) * 100))
        per_frame_scores.append({
            "frame": int(frame_num),
            "score": frame_score,
            "found": msg_frame is not None,
        })

        if progress_callback:
            progress_callback((idx + 1) / len(sample_indices))

    cap.release()

    # Temporal averaging — the key innovation for screen recording survival
    if soft_bits_a:
        averaged_bits = temporal_average_decode(soft_bits_a)
        msg = _bits_to_message(averaged_bits)
    else:
        msg = None

    avg_score = int(np.mean([s["score"] for s in per_frame_scores])) if per_frame_scores else 0

    return {
        "found": msg is not None,
        "message": msg or "",
        "scores": per_frame_scores,
        "avg_score": avg_score,
        "frames_analyzed": len(per_frame_scores),
    }


# ─────────────────────────────────────────
#  Single Frame Extraction (for UI)
# ─────────────────────────────────────────

def extract_single_frame(video_path: str, frame_num: int = 0) -> np.ndarray:
    """Extract a single frame from video for display."""
    cap = cv2.VideoCapture(video_path)
    if frame_num > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_frame_trust_score(video_path: str, frame_num: int,
                          password: str = "") -> dict:
    """Get trust score for a specific frame (used by frame slider)."""
    frame = extract_single_frame(video_path, frame_num)
    if frame is None:
        return {"score": 0, "found": False, "message": ""}

    # Save frame temporarily for trust score computation
    tmp_path = f"tmp/_frame_check_{frame_num}.png"
    os.makedirs("tmp", exist_ok=True)
    cv2.imwrite(tmp_path, frame)
    result = compute_trust_score(tmp_path, password)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return result
