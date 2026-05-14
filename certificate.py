import os
import json
import hashlib
from datetime import datetime

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import qrcode
    HAS_QR = True
except ImportError:
    HAS_QR = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─────────────────────────────────────────
#  Colour palette
# ─────────────────────────────────────────
_BG         = (10, 14, 26)
_BG_CARD    = (17, 24, 39)
_CYAN       = (56, 189, 248)
_INDIGO     = (99, 102, 241)
_SLATE      = (100, 116, 139)
_LIGHT      = (226, 232, 240)
_GREEN      = (34, 197, 94)
_YELLOW     = (245, 158, 11)
_RED        = (239, 68, 68)
_DARK_BORDER = (30, 45, 74)


class VeriFrameCert(FPDF):
    """Dark-themed A4 certificate with gradient accents."""

    def header(self):
        # full page dark background
        self.set_fill_color(*_BG)
        self.rect(0, 0, 210, 297, "F")

    def _gradient_bar(self, y, h=4):
        """Cyan → indigo gradient stripe."""
        for x in range(211):
            ratio = x / 210
            r = int(_CYAN[0] + ratio * (_INDIGO[0] - _CYAN[0]))
            g = int(_CYAN[1] + ratio * (_INDIGO[1] - _CYAN[1]))
            b = int(_CYAN[2] + ratio * (_INDIGO[2] - _CYAN[2]))
            self.set_fill_color(r, g, b)
            self.rect(x, y, 1.5, h, "F")

    def _card(self, x, y, w, h, radius=0):
        """Dark card background."""
        self.set_fill_color(*_BG_CARD)
        self.rect(x, y, w, h, "F")
        # top border accent
        self.set_fill_color(*_DARK_BORDER)
        self.rect(x, y, w, 0.5, "F")

    def _score_gauge(self, cx, cy, score, size=28):
        """Draw a circular trust-score gauge using concentric arcs."""
        import math
        # Background circle
        self.set_fill_color(30, 41, 59)
        self.set_draw_color(30, 41, 59)
        self.ellipse(cx - size, cy - size, size * 2, size * 2, "FD")

        # Score arc — draw filled wedges
        if score >= 70:
            color = _GREEN
        elif score >= 40:
            color = _YELLOW
        else:
            color = _RED

        self.set_fill_color(*color)
        self.set_draw_color(*color)
        angle_end = score / 100 * 360
        steps = max(1, int(angle_end / 3))
        for i in range(steps):
            a1 = math.radians(-90 + (i * angle_end / steps))
            a2 = math.radians(-90 + ((i + 1) * angle_end / steps))
            px1 = cx + (size - 3) * math.cos(a1)
            py1 = cy + (size - 3) * math.sin(a1)
            px2 = cx + (size - 3) * math.cos(a2)
            py2 = cy + (size - 3) * math.sin(a2)
            # small triangle wedge from outer ring
            self.line(px1, py1, px2, py2)

        # Inner circle (creates ring look)
        self.set_fill_color(*_BG_CARD)
        self.set_draw_color(*_BG_CARD)
        inner = size - 7
        self.ellipse(cx - inner, cy - inner, inner * 2, inner * 2, "FD")

        # Score number
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*color)
        score_text = str(score)
        tw = self.get_string_width(score_text)
        self.text(cx - tw / 2, cy + 4, score_text)

        # "/100" label
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_SLATE)
        lw = self.get_string_width("/100")
        self.text(cx - lw / 2, cy + 11, "/100")


def _make_thumbnail(image_path: str, max_w: int = 300, max_h: int = 200) -> str:
    """Create a bounded thumbnail and return its temp path."""
    if not HAS_PIL or not os.path.exists(image_path):
        return image_path
    try:
        img = PILImage.open(image_path)
        img.thumbnail((max_w, max_h), PILImage.LANCZOS)
        thumb_path = image_path.rsplit(".", 1)[0] + "_thumb.png"
        img.save(thumb_path, "PNG")
        return thumb_path
    except Exception:
        return image_path


def generate_certificate(
    image_path: str,
    watermark_text: str,
    trust_score: int,
    password_hint: str = "",
    output_path: str = "certificate.pdf",
    tamper_pct: float = -1,
    severity: str = "",
) -> str:
    """
    Generate a professional VeriFrame authentication certificate.

    Returns the output PDF path on success, empty string on failure.
    """
    if not HAS_FPDF:
        return ""

    pdf = VeriFrameCert()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # ── Top gradient accent bar ──
    pdf._gradient_bar(0, 5)

    # ── Shield icon + Title ──
    pdf.set_y(14)
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(*_CYAN)
    pdf.cell(0, 14, "VeriFrame", ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*_SLATE)
    pdf.cell(0, 7, "Certificate of Digital Authenticity", ln=True, align="C")

    # Thin divider
    pdf.ln(3)
    pdf._gradient_bar(pdf.get_y(), 1.2)
    pdf.ln(6)

    # ══════════════════════════════════════════
    #  Two-column layout: Thumbnail | Trust Score Gauge
    # ══════════════════════════════════════════
    col_top_y = pdf.get_y()

    # ── LEFT: Image thumbnail in a card ──
    pdf._card(14, col_top_y, 90, 72)
    thumb_path = None
    if os.path.exists(image_path):
        try:
            thumb_path = _make_thumbnail(image_path)
            # Center the image in card
            pdf.image(thumb_path, x=19, y=col_top_y + 4, w=80)
        except Exception:
            pdf.set_xy(14, col_top_y + 30)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*_SLATE)
            pdf.cell(90, 7, "[Image preview unavailable]", align="C")

    # ── RIGHT: Trust score gauge + status ──
    pdf._card(110, col_top_y, 86, 72)

    # Score colour + status text
    if trust_score >= 70:
        score_color = _GREEN
        status = "AUTHENTIC"
    elif trust_score >= 40:
        score_color = _YELLOW
        status = "MODIFIED"
    else:
        score_color = _RED
        status = "SUSPICIOUS"

    # Draw gauge
    gauge_cx = 153
    gauge_cy = col_top_y + 30
    pdf._score_gauge(gauge_cx, gauge_cy, trust_score, size=22)

    # Status badge below gauge
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*score_color)
    sw = pdf.get_string_width(status)
    pdf.text(gauge_cx - sw / 2, col_top_y + 58, status)

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*_SLATE)
    lbl = "Trust Score"
    pdf.text(gauge_cx - pdf.get_string_width(lbl) / 2, col_top_y + 64, lbl)

    # ══════════════════════════════════════════
    #  Details card
    # ══════════════════════════════════════════
    details_y = col_top_y + 78
    pdf._card(14, details_y, 182, 68)

    pdf.set_xy(18, details_y + 4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*_CYAN)
    pdf.cell(0, 6, "VERIFICATION DETAILS", ln=True)

    def detail_row(label, value, color=_LIGHT):
        pdf.set_x(22)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_SLATE)
        pdf.cell(46, 6, label)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*color)
        pdf.cell(0, 6, str(value), ln=True)

    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
    file_hash = ""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

    pdf.ln(1)
    detail_row("Watermark Text:", watermark_text, _CYAN)
    detail_row("Verified On:", timestamp)
    detail_row("SHA-256 Hash:", file_hash[:32].upper() + "...")
    detail_row("Protection:", "DCT + Spread Spectrum + ECC + Sync Markers")

    if password_hint:
        masked = password_hint[:2] + "*" * min(6, len(password_hint) - 2) if len(password_hint) > 2 else "**"
        detail_row("Password:", masked)

    if tamper_pct >= 0:
        if tamper_pct < 5:
            tp_color = _GREEN
        elif tamper_pct < 15:
            tp_color = _YELLOW
        else:
            tp_color = _RED
        sev_label = f"  ({severity})" if severity else ""
        detail_row("Tamper Detected:", f"{tamper_pct}%{sev_label}", tp_color)

    # ══════════════════════════════════════════
    #  Digital signature section
    # ══════════════════════════════════════════
    sig_y = details_y + 74
    pdf._card(14, sig_y, 182, 26)

    pdf.set_xy(18, sig_y + 3)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*_CYAN)
    pdf.cell(0, 5, "DIGITAL SIGNATURE", ln=True)

    # Compute signature: hash of (file_hash + watermark + score + timestamp)
    sig_payload = f"{file_hash}|{watermark_text}|{trust_score}|{timestamp}"
    signature = hashlib.sha256(sig_payload.encode()).hexdigest()

    pdf.set_x(22)
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 4, f"SIG: {signature[:32]}", ln=True)
    pdf.set_x(22)
    pdf.cell(0, 4, f"     {signature[32:]}", ln=True)

    # ══════════════════════════════════════════
    #  QR Code section
    # ══════════════════════════════════════════
    qr_y = sig_y + 30
    qr_temp_path = None

    if HAS_QR:
        pdf._card(14, qr_y, 182, 58)

        pdf.set_xy(18, qr_y + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_CYAN)
        pdf.cell(0, 5, "SCAN TO VERIFY", ln=True)

        qr_data = json.dumps({
            "app": "VeriFrame",
            "ver": "2.0",
            "text": watermark_text,
            "score": trust_score,
            "status": status,
            "hash": file_hash[:16],
            "sig": signature[:16],
            "tamper": tamper_pct if tamper_pct >= 0 else None,
            "date": timestamp,
        }, ensure_ascii=False)

        qr = qrcode.QRCode(version=1, box_size=5, border=2)
        qr.add_data(qr_data)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="white", back_color="#111827")
        qr_temp_path = output_path.replace(".pdf", "_qr.png")
        qr_img.save(qr_temp_path)

        pdf.image(qr_temp_path, x=20, y=qr_y + 10, w=42)

        # QR description text on the right
        pdf.set_xy(68, qr_y + 14)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_LIGHT)
        pdf.multi_cell(120, 5,
            "Scan this QR code with any smartphone camera\n"
            "to view the full verification data including\n"
            "watermark text, trust score, file hash, and\n"
            "digital signature.\n\n"
            "This certificate is cryptographically linked\n"
            "to the watermarked media file.",
        )

    # ══════════════════════════════════════════
    #  Bottom section
    # ══════════════════════════════════════════

    # Bottom gradient bar
    pdf._gradient_bar(284, 5)

    # Footer
    pdf.set_y(277)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 4,
        "VeriFrame  |  Invisible Digital Watermarking  |  "
        "DCT Frequency + Spread Spectrum + Reed-Solomon ECC + Sync Markers",
        ln=True, align="C",
    )
    pdf.set_font("Helvetica", "", 6)
    pdf.set_text_color(40, 50, 70)
    pdf.cell(0, 3,
        f"Generated: {timestamp}  |  This document is for verification purposes only.",
        ln=True, align="C",
    )

    # ── Save ──
    pdf.output(output_path)

    # Cleanup temp files
    if qr_temp_path and os.path.exists(qr_temp_path):
        try:
            os.remove(qr_temp_path)
        except OSError:
            pass
    if thumb_path and thumb_path != image_path and os.path.exists(thumb_path):
        try:
            os.remove(thumb_path)
        except OSError:
            pass

    return output_path
