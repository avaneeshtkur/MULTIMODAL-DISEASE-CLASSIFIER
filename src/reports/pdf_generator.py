"""
pdf_generator.py — screening report PDF (English or Hindi) with FPDF2.

Fonts: bundled Noto Sans + Noto Sans Devanagari (SIL OFL, assets/fonts).
Hindi needs text shaping (uharfbuzz) for correct conjuncts; without it, or
without the fonts, Hindi reports are refused rather than rendered garbled
and English falls back to the built-in Helvetica font.

The document is explicitly labelled a screening aid, never a diagnostic
certificate. Simulated results carry a red banner on every page.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

import config
from src.clinical.translations import t

NAVY = (14, 42, 71)
TEAL = (13, 148, 136)
GREY = (90, 98, 110)
LIGHT = (241, 245, 249)
STATUS_COLOURS = {
    "SCREENING_FLAGGED": (180, 83, 9),
    "NO_FLAG_DETECTED": (21, 128, 61),
    "INCONCLUSIVE": (71, 85, 105),
    "REJECTED": (153, 27, 27),
}

FONT_FILES = {
    ("NotoSans", ""): "NotoSans-Regular.ttf",
    ("NotoSans", "B"): "NotoSans-Bold.ttf",
    ("NotoDeva", ""): "NotoSansDevanagari-Regular.ttf",
    ("NotoDeva", "B"): "NotoSansDevanagari-Bold.ttf",
}


class ReportError(RuntimeError):
    pass


def _shaping_available() -> bool:
    try:
        import uharfbuzz  # noqa: F401
        return True
    except ImportError:
        return False


def _fonts_available() -> bool:
    return all((config.FONTS_DIR / f).exists() for f in FONT_FILES.values())


class _ReportPDF(FPDF):
    def __init__(self, lang: str, simulated: bool):
        super().__init__(format="A4")
        self.lang = lang
        self.simulated = simulated
        self.unicode_fonts = _fonts_available()
        if lang == "hi" and not (self.unicode_fonts and _shaping_available()):
            raise ReportError("Hindi PDF requires assets/fonts Noto fonts and the 'uharfbuzz' package.")
        if self.unicode_fonts:
            for (family, style), fname in FONT_FILES.items():
                self.add_font(family, style, str(config.FONTS_DIR / fname))
            self.base = "NotoDeva" if lang == "hi" else "NotoSans"
            # Noto Sans Devanagari has no Latin letters, so each font falls back to the other.
            self.set_fallback_fonts(["NotoSans" if lang == "hi" else "NotoDeva"])
            if _shaping_available():
                self.set_text_shaping(True)
        else:
            self.base = "Helvetica"
        self.set_auto_page_break(True, margin=18)
        self.set_margins(15, 15, 15)

    def txt(self, s: str) -> str:
        if self.unicode_fonts:
            return s
        return s.encode("latin-1", "replace").decode("latin-1")   # core fonts are Latin-1 only

    def font(self, size: float, bold: bool = False, colour=(20, 24, 31)):
        self.set_font(self.base, "B" if bold else "", size)
        self.set_text_color(*colour)

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 24, "F")
        self.set_fill_color(*TEAL)
        self.rect(0, 24, 210, 1.6, "F")
        self.set_xy(15, 6)
        self.set_font("NotoSans" if self.unicode_fonts else "Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(60, 8, config.APP_NAME)
        self.set_xy(80, 5)
        self.font(11, True, (255, 255, 255))
        self.cell(115, 7, self.txt(t("pdf.title", self.lang)), align="R")
        self.set_xy(80, 12)
        self.font(8.5, False, (203, 213, 225))
        self.cell(115, 6, self.txt(t("pdf.subtitle", self.lang)), align="R")
        self.set_y(30)
        if self.simulated:
            self.set_fill_color(254, 226, 226)
            self.font(9, True, (153, 27, 27))
            self.multi_cell(0, 6, self.txt(t("simulated_banner", self.lang)), border=0, fill=True,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.font(7.5, False, GREY)
        self.cell(0, 6, self.txt(f"{t('pdf.footer', self.lang)} {self.page_no()}"), align="C")

    def section(self, title: str):
        self.ln(3)
        self.font(11.5, True, NAVY)
        self.cell(0, 7, self.txt(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)

    def kv_table(self, rows: list[tuple[str, str]]):
        for i, (k, v) in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(*LIGHT)
            y0 = self.get_y()
            self.font(9, True, GREY)
            self.multi_cell(55, 6.5, self.txt(k), fill=fill, new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.font(9.5)
            self.multi_cell(125, 6.5, self.txt(v), fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_y(max(self.get_y(), y0 + 6.5))

    def bullets(self, items: list[str]):
        self.font(9.5)
        for it in items:
            self.set_x(17)
            self.multi_cell(0, 5.6, self.txt("•  " + it), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(0.6)


def _fmt(v, nd=3) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def generate_report(report: dict, lang: str = "en", out_dir: str | Path | None = None,
                    gradcam: dict | None = None) -> Path:
    """
    report keys: patient_id, patient_ref, created_at, disease, status, score, threshold,
        uncertainty_band, score_kind, model_version, routing, quality, reason, recommendations,
        questionnaire_rules, referral_label, simulated, processing_time_ms, result_id
    gradcam: {"base": HxWx3 uint8, "overlay": HxWx3 uint8} or None
    """
    pdf = _ReportPDF(lang, bool(report.get("simulated")))
    # No total-page alias: fpdf2 substitutes it with a bytes.replace() over the whole page
    # stream, which can match shaped Devanagari glyph IDs and corrupt unrelated text.
    pdf.alias_nb_pages("")
    pdf.add_page()
    T = lambda k, **kw: t(k, lang, **kw)  # noqa: E731

    status = report["status"]
    # Result banner
    col = STATUS_COLOURS.get(status, GREY)
    pdf.set_fill_color(*col)
    pdf.font(12.5, True, (255, 255, 255))
    pdf.multi_cell(0, 9, pdf.txt(f"  {T('status.' + status)}"), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if report.get("reason"):
        pdf.font(9, False, GREY)
        pdf.multi_cell(0, 5.5, pdf.txt(report["reason"]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.section(T("pdf.result"))
    band = report.get("uncertainty_band")
    score_txt = _fmt(report.get("score"))
    if report.get("score") is not None and report.get("score_kind"):
        score_txt += f"  ({report['score_kind']})"
    rows = [
        (T("pdf.patient_ref"), report.get("patient_ref") or "—"),
        (T("pdf.internal_id"), f"#{report.get('patient_id')}" if report.get("patient_id") else "—"),
        (T("pdf.date"), report.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        (T("pdf.module"), T(f"disease.{report['disease']}") if report.get("disease") else "—"),
        (T("pdf.routing"), report.get("routing") or "—"),
        (T("pdf.quality"), report.get("quality") or T("pdf.quality_ok")),
        (T("pdf.score"), score_txt),
        (T("pdf.threshold"), _fmt(report.get("threshold")) + (f"  (± band {band[0]:.3f}–{band[1]:.3f})" if band else "")),
        (T("pdf.model_version"), report.get("model_version") or "—"),
        (T("pdf.processing_time"), f"{report['processing_time_ms']:.0f} ms" if report.get("processing_time_ms") else "—"),
    ]
    pdf.kv_table(rows)
    pdf.font(8, False, GREY)
    pdf.multi_cell(0, 4.5, pdf.txt(T("score_note")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.section(T("pdf.gradcam"))
    if gradcam and gradcam.get("overlay") is not None:
        if pdf.get_y() > 190:
            pdf.add_page()
        y = pdf.get_y()
        for i, key in enumerate(("base", "overlay")):
            arr = np.asarray(gradcam[key]).astype(np.uint8)
            pdf.image(Image.fromarray(arr), x=15 + i * 92, y=y, w=72, h=72)
        pdf.set_y(y + 74)
        pdf.font(8, False, GREY)
        pdf.multi_cell(0, 4.5, pdf.txt(T("gradcam_note")), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:
        pdf.font(9, False, GREY)
        msg = T("pdf.gradcam_unavailable") + (f": {gradcam['message']}" if gradcam and gradcam.get("message") else "")
        pdf.multi_cell(0, 5.5, pdf.txt(msg), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.section(T("pdf.recommendations") + (f" — {report['referral_label']}" if report.get("referral_label") else ""))
    pdf.bullets(report.get("recommendations") or [])

    pdf.section(T("pdf.questionnaire"))
    rules = report.get("questionnaire_rules")
    if rules is None:
        pdf.bullets([T("q.none")])
    else:
        pdf.bullets([f"[{r['rule_id']}] {r['text']}" for r in rules] or [T("q.no_rules")])

    pdf.section(T("pdf.limitations"))
    pdf.bullets([T(f"pdf.limit.{i}") for i in range(1, 5)])

    pdf.section(T("pdf.disclaimer"))
    pdf.set_fill_color(255, 251, 235)
    pdf.font(9.5, True, (120, 53, 15))
    pdf.multi_cell(0, 6, pdf.txt(T("disclaimer")), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(8)
    pdf.font(9, False, GREY)
    pdf.cell(0, 6, pdf.txt(T("pdf.reviewed_by") + ": ______________________________"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="mediscan_report_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    rid = report.get("result_id") or "na"
    path = out_dir / f"mediscan_report_{rid}_{lang}_{stamp}.pdf"
    pdf.output(str(path))
    return path
