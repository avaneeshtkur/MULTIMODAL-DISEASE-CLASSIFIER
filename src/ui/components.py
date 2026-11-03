"""
components.py — HTML fragments for the Gradio UI. All dynamic text is escaped.
"""
from __future__ import annotations

from html import escape

import config
from src.clinical.translations import t

STATUS_BG = {"SCREENING_FLAGGED": "#c2410c", "NO_FLAG_DETECTED": "#15803d",
             "INCONCLUSIVE": "#475569", "REJECTED": "#991b1b"}
MODEL_NAMES = {"tb": "TB · chest X-ray", "dr": "DR · fundus", "oral": "Oral cancer · histopathology",
               "gatekeeper": "Gatekeeper · modality"}


def disclaimer_html(lang: str = "en") -> str:
    return f'<div class="ms-disclaimer">{escape(t("disclaimer", lang))}</div>'


def header_html(registry) -> str:
    return (f'<aside class="ms-rail"><div class="ms-wordmark"><span class="ms-mark" aria-hidden="true">+</span>'
            f'<div><h1>{escape(config.APP_NAME)}</h1><span class="ms-rail-caption">CLINICAL WORKSPACE</span></div></div>'
            '<div class="ms-rail-label">WORKSPACE</div><div class="ms-rail-footer">'
            '<span class="ms-online"></span> Local workspace<br><small>Images processed on this device.<br>Screening aid. Not a diagnosis.</small></div></aside>')


def modalities_html(registry) -> str:
    """Existing sample images are visual references, never patient results."""
    import base64
    items = [("tb", "01", "Tuberculosis", "Chest X-ray", "sample_chest_xray.png"),
             ("dr", "02", "Diabetic retinopathy", "Retinal fundus", "sample_fundus.png"),
             ("oral", "03", "Oral cancer", "Histopathology", "sample_histopathology.jpg")]
    rows = []
    for key, number, title, modality, filename in items:
        path = config.SAMPLES_DIR / filename
        image = ""
        if path.exists():
            mime = "image/jpeg" if path.suffix == ".jpg" else "image/png"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            image = f'<img src="data:{mime};base64,{data}" alt="Sample {escape(modality)} image" />'
        status = "Model available" if registry.is_available(key) else "Model unavailable"
        rows.append(f'<article class="ms-modality"><div class="ms-modality-image">{image}'
                    f'<span class="ms-image-label">SAMPLE IMAGE</span><span class="ms-modality-no">{number}</span></div>'
                    f'<div class="ms-modality-info"><span>{modality}</span><h3>{title}</h3>'
                    f'<p class="ms-model-state">{status}</p></div></article>')
    return '<div class="ms-modalities">' + ''.join(rows) + '</div>'


def state_banner(registry) -> str:
    if config.SIMULATION_MODE:
        return ('<div class="ms-banner sim">SIMULATION MODE is on (MEDISCAN_SIMULATION=1): modules without an '
                'installed model return SIMULATED scores for interface rehearsal. They are labelled everywhere '
                'and are not screening results.</div>')
    missing = [MODEL_NAMES[k] for k in config.SPECIALIST_KEYS if not registry.is_available(k)]
    if not missing:
        return '<div class="ms-banner info">All models installed and validated.</div>'
    return ('<div class="ms-banner warn"><b>Incomplete-model state.</b> Not installed or invalid: '
            f'{escape(", ".join(missing))}. Those modules return a setup message instead of a result. '
            'Install trained artifacts with <code>python scripts/import_artifacts.py --archive trained_outputs.zip</code>.'
            '</div>')


def model_table_html(registry) -> str:
    rows = []
    for r in registry.status_rows():
        cls = {"available": "ok", "invalid": "bad"}.get(r["status"], "miss")
        thr = "—" if r["threshold"] is None else f'{r["threshold"] * 100:.2f}%'
        err = f'<div class="ms-small">{escape(r["errors"])}</div>' if r["errors"] else ""
        rows.append(f'<tr><td>{escape(MODEL_NAMES[r["key"]])}{err}</td><td class="{cls}">{escape(r["status"])}</td>'
                    f'<td>{escape(str(r["version"]))}</td><td>{escape(r["classes"])}</td><td>{thr}</td>'
                    f'<td>{"n/a" if r["key"] == "gatekeeper" else ("yes" if r["gradcam"] else "no")}</td><td><code>{escape(r["sha256"] or "—")}</code></td></tr>')
    return ('<div class="ms-table-wrap"><table class="ms-models"><tr><th>Model</th><th>Status</th><th>Version</th><th>Classes</th>'
            '<th>Threshold</th><th>Grad-CAM</th><th>SHA-256</th></tr>' + "".join(rows) + "</table></div>")


def cards_html(items: list[tuple[str, object]]) -> str:
    return '<div class="ms-cards">' + "".join(
        f'<div class="ms-card"><div class="k">{escape(k)}</div><div class="v">{escape(str(v))}</div></div>'
        for k, v in items) + "</div>"


def how_it_works_html() -> str:
    steps = [("1 · Validate", "File format, corruption, size, aspect ratio, blur and exposure checks."),
             ("2 · Duplicate check", "Exact SHA-256 match against earlier screenings (never deletes)."),
             ("3 · Select specialist", "Choose TB, diabetic retinopathy, or oral cancer; disease is not inferred by the gate."),
             ("4 · Screen", "The selected specialist produces the disease score; suitability checks do not change it."),
             ("5 · Explain", "Grad-CAM heatmap from the matching PyTorch checkpoint."),
             ("6 · Next steps", "Rule-based recommendations, kept separate from the model output, and a PDF report.")]
    return '<div class="ms-steps">' + "".join(
        f'<div class="ms-step"><b>{escape(a)}</b>{escape(b)}</div>' for a, b in steps) + "</div>"


def _score_bar(score: float, thr: float, band, colour: str) -> str:
    band_html = ""
    if band:
        band_html = f'<div class="band" style="left:{band[0] * 100:.1f}%;width:{(band[1] - band[0]) * 100:.1f}%"></div>'
    return (f'<div class="ms-scorebar">{band_html}<div class="thr" style="left:{thr * 100:.1f}%"></div>'
            f'<div class="dot" style="left:{score * 100:.1f}%;background:{colour}"></div></div>'
            '<div class="ms-small">0% ─ model score ─ 100% · dark line = threshold · grey zone = uncertainty band. '
            'The threshold is the cutoff for raising a flag. Scores near it are inconclusive; '
            'a percentage score is not the probability of disease.</div>')


def rejection_html(outcome, lang: str) -> str:
    r = outcome.rejection
    return (f'<div class="ms-result"><div class="head" style="background:{STATUS_BG["REJECTED"]}">'
            f'{escape(t("status.REJECTED", lang))}</div><div class="body"><table>'
            f'<tr><td class="k">Reason code</td><td><code>{escape(r.code)}</code></td></tr>'
            f'<tr><td class="k">Why</td><td>{escape(r.explanation(lang))}</td></tr>'
            f'<tr><td class="k">What to do</td><td>{escape(r.action(lang))}</td></tr>'
            f'<tr><td class="k">Time (UTC)</td><td>{escape(r.timestamp)}</td></tr>'
            + (f'<tr><td class="k">Heuristic hint (not a classifier)</td><td>{escape(outcome.modality.hint)}</td></tr>'
               if outcome.modality and outcome.modality.hint else "")
            + (f'<tr><td class="k">Setup</td><td>{escape(r.details["setup"])}</td></tr>' if "setup" in r.details else "")
            + "</table></div></div>")


def result_html(outcome, lang: str, routing: str) -> str:
    if outcome.stage == "rejected":
        return rejection_html(outcome, lang) + disclaimer_html(lang)
    r = outcome.result
    status = r["prediction_status"]
    parts = []
    if r.get("simulated"):
        parts.append(f'<div class="ms-banner sim">{escape(t("simulated_banner", lang))}</div>')
    reason = t(f"reason.{r['reason_code']}", lang) if r.get("reason_code") else ""
    detail = r.get("reason") or ""
    score_html = ""
    if r.get("model_score") is not None and r.get("threshold") is not None:
        score_html = _score_bar(r["model_score"], r["threshold"], r.get("uncertainty_band"), STATUS_BG[status])
    rows = [("Module", t(f"disease.{r['disease']}", lang)), ("Routing", routing),
            ("Model output label", r.get("class_label") or "— (no label: inconclusive)"),
            ("Model score", "—" if r.get("model_score") is None else f'{r["model_score"] * 100:.2f}% ({r.get("score_kind")})'),
            ("Threshold", "—" if r.get("threshold") is None else f'{r["threshold"] * 100:.2f}%'),
            ("Model version", r.get("model_version") or "—"),
            ("Inference source", "SIMULATED" if r.get("simulated") else r.get('provider') or "Specialist ONNX model"),
            ("Disease target", r.get('target_definition') or '—'),
            ("Provider outcome", r.get('provider_status') or '—'),
            ("Model fingerprint", (r.get("model_sha256") or "—")[:16]),
            ("Processing time", f'{r["processing_time_ms"]:.0f} ms'),
            ("Result ID", f"#{outcome.result_id}")]
    table = "".join(f'<tr><td class="k">{escape(k)}</td><td>{escape(str(v))}</td></tr>' for k, v in rows)
    extra = ""
    if reason:
        extra += f'<div class="ms-banner warn">{escape(reason)}{(" " + escape(detail)) if detail else ""}</div>'
    for w in outcome.quality_warnings:
        extra += f'<div class="ms-banner warn">{escape(w)}</div>'
    if outcome.duplicate_note:
        extra += f'<div class="ms-banner info">{escape(outcome.duplicate_note)}</div>'
    if outcome.modality and outcome.modality.note:
        extra += f'<div class="ms-banner info">{escape(outcome.modality.note)}</div>'
    parts.append(f'<div class="ms-result"><div class="head" style="background:{STATUS_BG[status]}">'
                 f'{escape(t("status." + status, lang))}</div><div class="body">'
                 f'{score_html}<table>{table}</table><div class="ms-small" style="margin-top:8px">'
                 f'{escape(t("score_note", lang))}</div>{extra}</div></div>')
    parts.append(disclaimer_html(lang))
    return "".join(parts)


def clinical_html(outcome, lang: str) -> str:
    if outcome.stage == "rejected":
        return (f'<div class="ms-box"><div class="ms-section-title">{escape(t("pdf.recommendations", lang))}</div>'
                f'<ul><li>{escape(t("rec.any.REJECTED.1", lang))}</li></ul></div>')
    c = outcome.clinical
    img = "".join(f"<li>{escape(x)}</li>" for x in c["image"]["recommendations"])
    q = c["questionnaire"]
    if not q["answers"]:
        qhtml = f'<p class="ms-small">{escape(t("q.none", lang))}</p>'
    elif not q["rules"]:
        qhtml = f'<p class="ms-small">{escape(t("q.no_rules", lang))}</p>'
    else:
        qhtml = "<ul>" + "".join(f'<li><code>{escape(x["rule_id"])}</code> {escape(x["text"])}</li>'
                                 for x in q["rules"]) + "</ul>"
    ref = c["referral"]
    return (f'<div class="ms-section-title">{escape(t("pdf.recommendations", lang))} · '
            f'<span class="ms-pill {ref["priority"]}">{escape(ref["label"])}</span></div>'
            f'<div class="ms-two"><div class="ms-box"><b>{escape(t("source.image", lang))}</b><ul>{img}</ul></div>'
            f'<div class="ms-box"><b>{escape(t("source.questionnaire", lang))}</b>{qhtml}'
            f'<p class="ms-small">Questionnaire rules are transparent referral rules, separate from the image model. '
            f'No combined probability is computed.</p></div></div>')
