"""
VeriFrame Robustness Benchmark
Tests watermark survival across various attacks:
  - JPEG compression (multiple quality levels)
  - Resolution changes (downscale + upscale)
  - Gaussian noise addition
  - Brightness / contrast changes
  - Video codec re-encoding (mp4v)
  - Screenshot simulation (crop + resize)
  - Rotation
"""

import cv2
import numpy as np
import os
import tempfile

from watermark_logic import encode_image, decode_image, compute_trust_score


def _apply_attack(image_path: str, attack_name: str,
                  output_path: str, **kwargs) -> str:
    """Apply a specific attack to an image and save the result."""
    img = cv2.imread(image_path)
    if img is None:
        return ""

    h, w = img.shape[:2]

    if attack_name == "jpeg":
        quality = kwargs.get("quality", 75)
        cv2.imwrite(output_path, img,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        return f"JPEG Q={quality}"

    elif attack_name == "resize":
        scale = kwargs.get("scale", 0.5)
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h),
                              interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, restored)
        return f"Resize {int(scale*100)}% -> 100%"

    elif attack_name == "noise":
        sigma = kwargs.get("sigma", 15)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float64)
        noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, noisy)
        return f"Gaussian noise sigma={sigma}"

    elif attack_name == "brightness":
        delta = kwargs.get("delta", 30)
        bright = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, bright)
        return f"Brightness +{delta}"

    elif attack_name == "contrast":
        factor = kwargs.get("factor", 1.3)
        adjusted = np.clip(img.astype(np.float64) * factor, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, adjusted)
        return f"Contrast x{factor}"

    elif attack_name == "crop_resize":
        # Simulate screenshot: crop 10% margins, resize back
        margin = kwargs.get("margin", 0.1)
        m_h, m_w = int(h * margin), int(w * margin)
        cropped = img[m_h:h - m_h, m_w:w - m_w]
        restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, restored)
        return f"Crop {int(margin*100)}% + Resize"

    elif attack_name == "rotation":
        angle = kwargs.get("angle", 2)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(output_path, rotated)
        return f"Rotation {angle} deg"

    elif attack_name == "blur":
        ksize = kwargs.get("ksize", 3)
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        cv2.imwrite(output_path, blurred)
        return f"Gaussian blur k={ksize}"

    elif attack_name == "saturation":
        factor = kwargs.get("factor", 1.5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, result)
        return f"Saturation x{factor}"

    # Fallback: just copy
    cv2.imwrite(output_path, img)
    return attack_name


def run_robustness_benchmark(image_path: str, secret: str,
                             password: str = "",
                             progress_callback=None) -> dict:
    """
    Run a full robustness benchmark:
    1. Embed watermark
    2. Apply various attacks
    3. Attempt decode after each attack
    4. Return results table

    Returns dict with:
      - watermarked_path: path to the watermarked image
      - results: list of {attack, description, found, message, score}
      - survival_rate: percentage of attacks where watermark survived
    """
    os.makedirs("tmp", exist_ok=True)
    wm_path = "tmp/benchmark_wm.png"

    # Step 1: Embed
    r = encode_image(image_path, secret, wm_path, password)
    if "✅" not in r:
        return {"watermarked_path": "", "results": [], "survival_rate": 0}

    # Step 2: Define attacks
    attacks = [
        ("jpeg", {"quality": 95}),
        ("jpeg", {"quality": 75}),
        ("jpeg", {"quality": 50}),
        ("jpeg", {"quality": 30}),
        ("resize", {"scale": 0.75}),
        ("resize", {"scale": 0.5}),
        ("noise", {"sigma": 10}),
        ("noise", {"sigma": 25}),
        ("brightness", {"delta": 30}),
        ("brightness", {"delta": -30}),
        ("contrast", {"factor": 1.3}),
        ("contrast", {"factor": 0.7}),
        ("blur", {"ksize": 3}),
        ("blur", {"ksize": 5}),
        ("rotation", {"angle": 2}),
        ("rotation", {"angle": 5}),
        ("crop_resize", {"margin": 0.05}),
        ("crop_resize", {"margin": 0.1}),
        ("saturation", {"factor": 1.5}),
        ("saturation", {"factor": 0.5}),
    ]

    results = []
    total = len(attacks)

    for idx, (attack_name, params) in enumerate(attacks):
        attacked_path = f"tmp/benchmark_{attack_name}_{idx}.png"
        description = _apply_attack(wm_path, attack_name, attacked_path, **params)

        # Try to decode
        decode_result = decode_image(attacked_path, password)
        found = "✅" in decode_result
        extracted_msg = decode_result.replace("✅ ", "") if found else ""

        # Trust score
        trust = compute_trust_score(attacked_path, password)
        score = trust.get("score", 0)

        results.append({
            "attack": attack_name,
            "description": description,
            "found": found,
            "message": extracted_msg,
            "score": score,
            "correct": extracted_msg == secret if found else False,
        })

        # Cleanup
        try:
            os.remove(attacked_path)
        except OSError:
            pass

        if progress_callback:
            progress_callback((idx + 1) / total)

    # Cleanup watermarked image
    survived = sum(1 for r in results if r["found"])
    correct = sum(1 for r in results if r["correct"])

    return {
        "watermarked_path": wm_path,
        "results": results,
        "survival_rate": round(survived / max(1, total) * 100, 1),
        "accuracy_rate": round(correct / max(1, total) * 100, 1),
        "total_attacks": total,
        "survived": survived,
        "correct": correct,
    }
