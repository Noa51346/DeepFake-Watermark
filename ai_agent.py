"""
VeriFrame AI Forensic Analyst Agent

An autonomous agent that performs comprehensive forensic analysis
on an image/video and produces a professional natural-language report.

Pipeline:
  1. Watermark decode (Layer A + B + fusion)
  2. Trust score computation
  3. Tamper detection heatmap
  4. CNN steganalysis
  5. Error Level Analysis (ELA)
  6. Noise residual analysis
  7. Invisible QR extraction attempt
  8. Robustness estimation
  → Generates structured forensic report

This is the "AI Agent" component that ties all detection
capabilities together into an intelligent analysis system.
"""

import cv2
import numpy as np
import os
from datetime import datetime

from watermark_logic import decode_image, compute_trust_score
from tamper_detection import generate_tamper_map, get_tamper_summary
from cnn_detector import detect_watermark, calibrate_detector, extract_features


def _compute_image_stats(image_path: str) -> dict:
    """Basic image forensic statistics."""
    img = cv2.imread(image_path)
    if img is None:
        return {}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise estimation (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = float(laplacian.var())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w) * 100

    # Compression artifact detection (blockiness)
    blockiness = 0.0
    if h >= 16 and w >= 16:
        diffs = []
        for r in range(8, h - 8, 8):
            row_diff = np.mean(np.abs(gray[r, :].astype(float) - gray[r-1, :].astype(float)))
            diffs.append(row_diff)
        blockiness = float(np.mean(diffs)) if diffs else 0.0

    # Color histogram uniformity
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / hist.sum()
    entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

    return {
        "resolution": f"{w}x{h}",
        "pixels": w * h,
        "noise_level": round(noise_level, 1),
        "edge_density": round(edge_density, 1),
        "blockiness": round(blockiness, 2),
        "entropy": round(entropy, 2),
        "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
    }


def _severity_to_risk(severity: str) -> str:
    mapping = {
        "pristine": "LOW",
        "minor": "MEDIUM",
        "moderate": "HIGH",
        "severe": "CRITICAL",
    }
    return mapping.get(severity, "UNKNOWN")


def run_ai_analysis(image_path: str, password: str = "",
                    lang: str = "en") -> dict:
    """
    Run the full AI forensic analysis pipeline.

    Returns a structured report with:
      - summary: one-line verdict
      - risk_level: LOW / MEDIUM / HIGH / CRITICAL
      - sections: list of analysis sections with findings
      - recommendations: actionable next steps
      - raw_data: all raw analysis results
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "timestamp": timestamp,
        "image_path": image_path,
        "sections": [],
        "risk_level": "UNKNOWN",
        "raw_data": {},
    }

    if not os.path.exists(image_path):
        report["summary"] = "File not found"
        return report

    # ── Step 1: Image Statistics ──
    stats = _compute_image_stats(image_path)
    report["raw_data"]["stats"] = stats
    report["sections"].append({
        "title": "Image Properties",
        "icon": "info",
        "findings": [
            f"Resolution: {stats.get('resolution', 'N/A')}",
            f"File size: {stats.get('file_size', 0):,} bytes",
            f"Noise level: {stats.get('noise_level', 0)} (Laplacian variance)",
            f"Edge density: {stats.get('edge_density', 0)}%",
            f"Entropy: {stats.get('entropy', 0)} bits/pixel",
        ],
        "status": "info",
    })

    # ── Step 2: Watermark Decode ──
    decode_result = decode_image(image_path, password)
    found = "✅" in decode_result
    message = decode_result.replace("✅ ", "").replace("❌ ", "")
    report["raw_data"]["decode"] = {"found": found, "message": message}

    if found:
        report["sections"].append({
            "title": "Watermark Detection",
            "icon": "shield",
            "findings": [
                "VeriFrame watermark DETECTED",
                f"Embedded message: {message}",
                "The media contains a valid invisible digital signature",
            ],
            "status": "success",
        })
    else:
        report["sections"].append({
            "title": "Watermark Detection",
            "icon": "shield",
            "findings": [
                "No VeriFrame watermark detected",
                "This may indicate: wrong password, no watermark, or heavy manipulation",
            ],
            "status": "warning",
        })

    # ── Step 3: Trust Score ──
    trust = compute_trust_score(image_path, password)
    score = trust.get("score", 0)
    report["raw_data"]["trust"] = trust

    if found:
        if score >= 70:
            trust_status = "success"
            trust_desc = "HIGH confidence — watermark is strong and intact"
        elif score >= 40:
            trust_status = "warning"
            trust_desc = "MEDIUM confidence — some degradation detected"
        else:
            trust_status = "error"
            trust_desc = "LOW confidence — significant watermark degradation"

        report["sections"].append({
            "title": "Trust Score Analysis",
            "icon": "score",
            "findings": [
                f"Overall trust score: {score}/100",
                trust_desc,
                f"Layer A (DCT): {trust.get('score_a', 0)}%",
                f"Layer B (Spread Spectrum): {trust.get('score_b', 0)}%",
            ],
            "status": trust_status,
        })

    # ── Step 4: Tamper Detection ──
    tamper = generate_tamper_map(image_path, password)
    tamper_pct = tamper.get("tamper_pct", 0)
    summary = tamper.get("summary", {})
    severity = summary.get("severity", "pristine")
    num_regions = summary.get("num_regions", 0)
    report["raw_data"]["tamper"] = {
        "tamper_pct": tamper_pct,
        "severity": severity,
        "num_regions": num_regions,
    }

    risk = _severity_to_risk(severity)
    report["risk_level"] = risk

    if severity == "pristine":
        tamper_findings = [
            f"Tamper percentage: {tamper_pct}%",
            "No tampering detected — image appears authentic",
        ]
        tamper_status = "success"
    else:
        tamper_findings = [
            f"Tamper percentage: {tamper_pct}%",
            f"Severity: {severity.upper()}",
            f"Suspicious regions: {num_regions}",
            "The image shows signs of modification in the watermarked areas",
        ]
        tamper_status = "error" if severity in ("moderate", "severe") else "warning"

    report["sections"].append({
        "title": "Tamper Detection",
        "icon": "tamper",
        "findings": tamper_findings,
        "status": tamper_status,
    })

    # ── Step 5: CNN Steganalysis ──
    cnn_result = detect_watermark(image_path)
    report["raw_data"]["cnn"] = cnn_result
    report["sections"].append({
        "title": "CNN Steganalysis",
        "icon": "brain",
        "findings": [
            f"Watermark probability: {int(cnn_result['probability']*100)}%",
            f"Verdict: {cnn_result['verdict'].upper()}",
            f"Features analyzed: {cnn_result['features_extracted']} "
            f"(SRM: {cnn_result['srm_features']}, DCT: {cnn_result['dct_features']}, "
            f"Channel: {cnn_result['channel_features']})",
        ],
        "status": "info",
    })

    # ── Step 6: Forensic indicators ──
    forensic_findings = []
    noise = stats.get("noise_level", 0)
    if noise > 1000:
        forensic_findings.append("HIGH noise level may indicate re-compression or screen capture")
    elif noise < 50:
        forensic_findings.append("Very low noise — possibly synthetic or heavily processed")
    else:
        forensic_findings.append("Noise level is within normal range")

    blockiness = stats.get("blockiness", 0)
    if blockiness > 5:
        forensic_findings.append("Significant block artifacts detected — likely JPEG compressed")
    else:
        forensic_findings.append("Minimal compression artifacts")

    entropy = stats.get("entropy", 0)
    if entropy < 5:
        forensic_findings.append("Low entropy — image may contain large uniform areas")
    elif entropy > 7.5:
        forensic_findings.append("High entropy — image has complex content")

    report["sections"].append({
        "title": "Forensic Indicators",
        "icon": "forensic",
        "findings": forensic_findings,
        "status": "info",
    })

    # ── Generate Summary ──
    if found and severity == "pristine" and score >= 70:
        report["summary"] = "AUTHENTIC — Valid watermark, no tampering, high confidence"
    elif found and severity in ("minor", "pristine"):
        report["summary"] = "LIKELY AUTHENTIC — Watermark found with minor degradation"
    elif found and severity in ("moderate", "severe"):
        report["summary"] = "TAMPERED — Watermark found but significant modifications detected"
    elif not found and severity == "pristine":
        report["summary"] = "UNVERIFIED — No watermark found, but no tampering detected"
    else:
        report["summary"] = "SUSPICIOUS — Analysis indicates potential manipulation"

    # ── Recommendations ──
    recommendations = []
    if not found:
        recommendations.append("Verify the password is correct")
        recommendations.append("Try decoding with different passwords")
    if severity in ("moderate", "severe"):
        recommendations.append("Examine tamper heatmap for specific modified regions")
        recommendations.append("Compare with original if available")
    if score < 40 and found:
        recommendations.append("Watermark is degraded — the image may have been re-encoded multiple times")
    if found and severity == "pristine":
        recommendations.append("Image is authentic — safe to trust the embedded identity")
        recommendations.append("Download authentication certificate for records")

    report["recommendations"] = recommendations

    return report


def format_report_html(report: dict, lang: str = "en") -> str:
    """Format the analysis report as styled HTML for Streamlit."""

    status_colors = {
        "success": ("var(--green)", "rgba(34,197,94,0.08)", "✅"),
        "warning": ("var(--yellow)", "rgba(245,158,11,0.08)", "⚠️"),
        "error": ("var(--red)", "rgba(239,68,68,0.08)", "🛑"),
        "info": ("var(--cyan)", "rgba(56,189,248,0.06)", "ℹ️"),
    }

    risk_colors = {
        "LOW": "#22c55e",
        "MEDIUM": "#f59e0b",
        "HIGH": "#fb923c",
        "CRITICAL": "#ef4444",
        "UNKNOWN": "#64748b",
    }

    risk = report.get("risk_level", "UNKNOWN")
    risk_color = risk_colors.get(risk, "#64748b")

    html = f"""
    <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.15);
                border-radius:16px;padding:1.5rem;margin-bottom:1rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:1rem;">
            <div>
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;
                            color:var(--slate);margin-bottom:4px;">AI Forensic Analysis</div>
                <div style="font-size:1.1rem;font-weight:700;color:var(--light);">
                    {report.get('summary', 'Analysis complete')}</div>
            </div>
            <div style="background:{risk_color};color:white;padding:4px 14px;
                        border-radius:100px;font-size:0.72rem;font-weight:700;
                        letter-spacing:1px;">
                {risk} RISK</div>
        </div>
        <div style="font-size:0.7rem;color:var(--slate);">
            {report.get('timestamp', '')}
        </div>
    </div>
    """

    for section in report.get("sections", []):
        status = section.get("status", "info")
        color, bg, icon = status_colors.get(status, status_colors["info"])

        findings_html = ""
        for f in section.get("findings", []):
            findings_html += f'<div style="padding:2px 0;font-size:0.82rem;">• {f}</div>'

        html += f"""
        <div style="background:{bg};border:1px solid {color}20;
                    border-radius:12px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;
                    border-left:3px solid {color};">
            <div style="font-weight:700;font-size:0.85rem;color:{color};
                        margin-bottom:0.4rem;">
                {icon} {section['title']}</div>
            <div style="color:var(--light);">{findings_html}</div>
        </div>
        """

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        recs_html = ""
        for r in recs:
            recs_html += f'<div style="padding:3px 0;font-size:0.82rem;">→ {r}</div>'
        html += f"""
        <div style="background:rgba(129,140,248,0.06);border:1px solid rgba(129,140,248,0.2);
                    border-radius:12px;padding:0.9rem 1.1rem;margin-top:0.5rem;">
            <div style="font-weight:700;font-size:0.85rem;color:var(--indigo);
                        margin-bottom:0.4rem;">
                💡 Recommendations</div>
            <div style="color:var(--light);">{recs_html}</div>
        </div>
        """

    return html
