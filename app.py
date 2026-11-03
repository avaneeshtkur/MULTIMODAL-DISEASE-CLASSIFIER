"""
app.py — MediScan AI Gradio 6 application.

    python app.py                      # http://127.0.0.1:7860
    MEDISCAN_SIMULATION=1 python app.py   # rehearse the UI without trained models (labelled SIMULATED)

Runs fully offline on CPU: ONNX Runtime inference, SQLite storage, local PDF reports.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

import config
from src.clinical.symptom_assessment import QUESTIONS
from src.clinical.translations import t
from src.core.pipeline import build_context, report_from_outcome, report_from_record, routing_text, run_screening
from src.reports.pdf_generator import ReportError, generate_report
from src.ui import charts
from src.ui.components import (cards_html, clinical_html, disclaimer_html, header_html, how_it_works_html,
                               model_table_html, modalities_html, result_html, state_banner)
from src.ui.theme import CSS, THEME_JS, TOGGLE_THEME_JS, build_theme
from src.utils.logging_config import get_logger, setup_logging
from src.utils.validators import AGE_RANGES, SEX_CHOICES, ValidationError

setup_logging()
log = get_logger("app")
ctx = build_context()

MODULE_CHOICES = [("Automatic — trained gatekeeper detects the image type", "auto"),
                  ("Tuberculosis — chest X-ray", "tb"),
                  ("Diabetic retinopathy — retinal fundus photograph", "dr"),
                  ("Oral cancer (OSCC) — histopathology", "oral")]
LANG_CHOICES = [("English", "en"), ("हिन्दी (Hindi)", "hi")]
SAMPLES = {"Chest X-ray": "sample_chest_xray.png", "Fundus": "sample_fundus.png",
           "Histopathology": "sample_histopathology.jpg", "Invalid (blurry)": "invalid_blurry.png",
           "Unsupported (non-medical)": "unsupported_synthetic.png"}


# ── helpers ──────────────────────────────────────────────────────
def patient_choices(include_demo: bool = True):
    out = []
    for p in ctx.patients.search("", include_demo=include_demo, limit=500):
        bits = [f"#{p['id']}", p["reference"] or "(no reference)", p["age_range"] or "", "DEMO" if p["is_demo"] else ""]
        out.append((" · ".join(b for b in bits if b), p["id"]))
    return out


def bool_choices(disease: str, lang: str):
    return [(t(f"q.{disease}.{qid}", lang), qid) for qid, kind, _ in QUESTIONS[disease] if kind == "bool"]


def answers_from_inputs(tb_cb, dr_type, dr_years, dr_comp, dr_last, oral_cb) -> dict:
    return {
        "tb": {k: True for k in (tb_cb or [])},
        "dr": {k: v for k, v in {"diabetes_type": dr_type, "diabetes_years": dr_years,
                                 "eye_complications": bool(dr_comp) or None, "last_eye_exam": dr_last}.items()
               if v not in (None, "")},
        "oral": {k: True for k in (oral_cb or [])},
    }


def overview_stats():
    s = ctx.analytics.summary(include_demo=True)
    return cards_html([("Screenings", s["results"]), ("TB", s["count_tb"]), ("DR", s["count_dr"]),
                       ("Oral cancer", s["count_oral"]), ("Flagged", s["SCREENING_FLAGGED"]),
                       ("No flag", s["NO_FLAG_DETECTED"]), ("Inconclusive", s["INCONCLUSIVE"]),
                       ("Rejected uploads", s["rejections"])])


# ── New screening ────────────────────────────────────────────────
def register_patient(ref, age, age_range, sex, is_demo):
    try:
        pid = ctx.patients.register(ref, age, age_range, sex, is_demo=is_demo)
    except ValidationError as e:
        return gr.update(), f'<div class="ms-banner warn">{e}</div>'
    return (gr.update(choices=patient_choices(), value=pid),
            f'<div class="ms-banner info">Registered patient #{pid}{" (demo record)" if is_demo else ""}.</div>')


def preview(file):
    return file if file else None


def on_module_change(module):
    show = {k: module in ("auto", k) for k in ("tb", "dr", "oral")}
    return gr.update(visible=show["tb"]), gr.update(visible=show["dr"]), gr.update(visible=show["oral"])


def run(file, patient_id, module, tb_cb, dr_type, dr_years, dr_comp, dr_last, oral_cb, allow_rescreen,
        do_gradcam, lang):
    empty = (None, None, None)
    if not patient_id:
        msg = '<div class="ms-banner warn">Select or register a patient first (step 1).</div>'
        return msg, "", *empty, "", {}, None, gr.update(value=None, visible=False)
    if not file:
        msg = '<div class="ms-banner warn">Upload an image first (step 4).</div>'
        return msg, "", *empty, "", {}, None, gr.update(value=None, visible=False)
    path = Path(file)
    answers = answers_from_inputs(tb_cb, dr_type, dr_years, dr_comp, dr_last, oral_cb)
    try:
        o = run_screening(ctx, path.read_bytes(), path.name, int(patient_id), module, answers, lang,
                          allow_rescreen=allow_rescreen, explain=do_gradcam)
    except Exception as e:  # noqa: BLE001 — surface any unexpected failure without crashing the UI
        log.exception("screening failed")
        return (f'<div class="ms-banner warn">Unexpected error: {type(e).__name__}: {e}</div>', "", *empty, "",
                {}, None, gr.update(value=None, visible=False))
    ex = o.explanation or {}
    imgs = (ex.get("base"), ex.get("heatmap"), ex.get("overlay")) if ex.get("ok") else empty
    gmsg = ex.get("message") or ("Grad-CAM not requested." if o.stage == "screened" else "No explanation: image rejected.")
    details = {"image_sha256": o.image_sha256, "quality_metrics": o.quality_metrics,
               "quality_warnings": o.quality_warnings,
               "modality": None if not o.modality else {"modality": o.modality.modality, "source": o.modality.source,
                                                        "confidence": o.modality.confidence, "probs": o.modality.probs,
                                                        "heuristic_hint": o.modality.hint},
               "rejection": o.rejection.to_dict(lang) if o.rejection else None,
               "result": {k: v for k, v in (o.result or {}).items() if k != "model_sha256"}}
    return (result_html(o, lang, routing_text(o.modality, lang)), clinical_html(o, lang), *imgs,
            f'<div class="ms-small">{gmsg}</div>', details, o, gr.update(value=None, visible=False))


def rerender(o, lang):
    if o is None:
        return gr.update(), gr.update()
    return result_html(o, lang, routing_text(o.modality, lang)), clinical_html(o, lang)


def make_report(o, lang):
    if o is None or o.stage != "screened":
        gr.Warning("Run a screening that produces a result first (rejected images have no report).")
        return gr.update(value=None, visible=False)
    try:
        path = generate_report(report_from_outcome(o, lang), lang, tempfile.mkdtemp(prefix="mediscan_"),
                               gradcam=o.explanation)
    except ReportError as e:
        gr.Warning(str(e))
        return gr.update(value=None, visible=False)
    ctx.db.audit("report_generated", "screening_result", o.result_id, {"lang": lang})
    return gr.update(value=str(path), visible=True)


def on_lang_change(lang, o):
    a, b = rerender(o, lang)
    return (a, b, disclaimer_html(lang), gr.update(choices=bool_choices("tb", lang), label=t("q.header", lang)),
            gr.update(choices=bool_choices("oral", lang)), gr.update(label=t("q.dr.diabetes_type", lang)),
            gr.update(label=t("q.dr.diabetes_years", lang)), gr.update(label=t("q.dr.eye_complications", lang)),
            gr.update(label=t("q.dr.last_eye_exam", lang)))


# ── Records ──────────────────────────────────────────────────────
PATIENT_COLS = ["id", "reference", "age", "age_range", "sex", "is_demo", "screenings", "last_screening", "created_at"]
HISTORY_COLS = ["id", "created_at", "disease", "screening_status", "model_score", "threshold", "model_version",
                "referral_priority", "simulated", "image_hash"]


def search_patients(q, include_demo):
    rows = ctx.patients.search(q, include_demo)
    return pd.DataFrame(rows, columns=PATIENT_COLS) if rows else pd.DataFrame(columns=PATIENT_COLS)


def load_history(pid):
    if not pid:
        return pd.DataFrame(columns=HISTORY_COLS), pd.DataFrame(), "Select a patient."
    p = ctx.patients.get(int(pid))
    if not p:
        return pd.DataFrame(columns=HISTORY_COLS), pd.DataFrame(), "Patient not found."
    h = ctx.patients.history(int(pid))
    rj = ctx.patients.rejections(int(pid))
    info = (f"**Patient #{p['id']}** · reference: {p['reference'] or '—'} · age: {p['age'] or '—'} "
            f"({p['age_range'] or '—'}) · sex: {p['sex'] or '—'} · {'DEMO record' if p['is_demo'] else 'clinic record'}"
            f" · {len(h)} screening(s), {len(rj)} rejected upload(s)")
    return (pd.DataFrame(h, columns=HISTORY_COLS) if h else pd.DataFrame(columns=HISTORY_COLS),
            pd.DataFrame(rj) if rj else pd.DataFrame(columns=["id", "created_at", "code", "stage", "requested_module"]),
            info)


def select_patient(evt: gr.SelectData, df):
    try:
        return int(df.iloc[evt.index[0]]["id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def show_result(rid, lang):
    if not rid:
        return "Select a screening."
    rec = ctx.screenings.get_result(int(rid))
    if not rec:
        return "Screening not found."
    rep = report_from_record(rec, lang)
    lines = [f"### Screening #{rec['id']} — {t('disease.' + rec['disease'], lang)}",
             f"**{t('status.' + rec['screening_status'], lang)}**" + ("  \n**SIMULATED — not a model output**" if rec["simulated"] else ""),
             f"- Date (UTC): {rec['created_at']}",
             f"- Model score / threshold: {rec['model_score'] if rec['model_score'] is not None else '—'} / "
             f"{rec['threshold'] if rec['threshold'] is not None else '—'} (model score is not a probability of disease)",
             f"- Model version: {rec['model_version'] or '—'} · processing {rec['processing_time_ms'] or 0:.0f} ms",
             f"- Routing: {rep['routing']}", f"- Image SHA-256: `{rec['image_hash'][:16]}…` (image itself not retained)",
             f"- Referral: **{rep['referral_label']}**", "", f"**{t('pdf.recommendations', lang)}**"]
    lines += [f"- {x}" for x in rep["recommendations"]]
    if rep["questionnaire_rules"] is not None:
        lines += ["", f"**{t('pdf.questionnaire', lang)}**"]
        lines += [f"- [{x['rule_id']}] {x['text']}" for x in rep["questionnaire_rules"]] or [f"- {t('q.no_rules', lang)}"]
    lines += ["", f"> {t('disclaimer', lang)}"]
    return "\n".join(lines)


def select_result(evt: gr.SelectData, df):
    return select_patient(evt, df)


def record_report(rid, lang):
    if not rid:
        gr.Warning("Select a screening first.")
        return gr.update(value=None, visible=False)
    rec = ctx.screenings.get_result(int(rid))
    if not rec:
        return gr.update(value=None, visible=False)
    try:
        path = generate_report(report_from_record(rec, lang), lang, tempfile.mkdtemp(prefix="mediscan_"),
                               gradcam={"message": "image not retained (privacy default)"})
    except ReportError as e:
        gr.Warning(str(e))
        return gr.update(value=None, visible=False)
    ctx.db.audit("report_generated", "screening_result", int(rid), {"lang": lang, "from": "record"})
    return gr.update(value=str(path), visible=True)


def export_csv(pid, include_demo):
    csv = ctx.patients.export_history_csv(int(pid) if pid else None, include_demo)
    if not csv:
        gr.Warning("Nothing to export.")
        return gr.update(value=None, visible=False)
    out = Path(tempfile.mkdtemp(prefix="mediscan_")) / (f"screening_history_patient_{int(pid)}.csv" if pid
                                                         else "screening_history_all.csv")
    out.write_text(csv, encoding="utf-8")
    return gr.update(value=str(out), visible=True)


def delete_patient(pid, confirm, include_demo):
    if not pid:
        return "Select a patient first.", gr.update()
    try:
        ctx.patients.delete_demo_patient(int(pid), confirm)
    except ValidationError as e:
        return f"{e}", gr.update()
    return f"Deleted demo patient #{int(pid)} and their screening records.", search_patients("", include_demo)


# ── Analytics ────────────────────────────────────────────────────
def analytics(include_demo, include_sim, lang):
    s = ctx.analytics.summary(include_demo, include_sim)
    cards = cards_html([("Screening sessions", s["sessions"]), ("Screening results", s["results"]),
                        ("TB screenings", s["count_tb"]), ("DR screenings", s["count_dr"]),
                        ("Oral cancer screenings", s["count_oral"]), ("Flagged", s["SCREENING_FLAGGED"]),
                        ("No flag", s["NO_FLAG_DETECTED"]), ("Inconclusive", s["INCONCLUSIVE"]),
                        ("Rejected uploads", s["rejections"]), ("Simulated (excluded)" if not include_sim else "Simulated",
                                                                s["simulated_results"])])
    rej = ctx.analytics.rejection_breakdown(include_demo)
    rej_df = pd.DataFrame(rej, columns=["code", "n", "pct"]) if rej else pd.DataFrame(columns=["code", "n", "pct"])
    return (cards, charts.disease_status_chart(ctx.analytics.status_by_disease(include_demo, include_sim), lang),
            charts.rejection_chart(rej), charts.trend_chart(ctx.analytics.daily_trend(include_demo, include_sim)),
            rej_df, model_table_html(ctx.registry))


# ── About ────────────────────────────────────────────────────────
def about_md() -> str:
    parts = ["## Architecture",
             "Upload → deterministic validation → SHA-256 duplicate check → gatekeeper (modality) → specialist "
             "EfficientNet-B0 (ONNX Runtime, CPU) → threshold + uncertainty band → Grad-CAM (PyTorch checkpoint) → "
             "rule-based clinical decision support → SQLite → PDF. See `docs/ARCHITECTURE.md`.",
             "## Installed models"]
    for key in config.MODEL_KEYS:
        h = ctx.registry.get(key)
        md = h.metadata
        parts.append(f"### {key.upper()} — {h.status}")
        if not h.available:
            parts.append(f"Not available: {'; '.join(h.errors) or 'not installed'}.")
            continue
        parts.append(f"- Version `{h.version}` · architecture `{md.get('architecture')}` · classes {md.get('class_labels')}"
                     f" · threshold {md.get('threshold', '—')} · input {md['preprocessing']['input_size']} px "
                     f"({md['preprocessing']['color_mode']})")
        m = md.get("metrics")
        if m:
            parts.append("- Metrics reported in the artifact metadata (measured by the training pipeline, "
                         "not by this app):\n\n```json\n" + json.dumps(m, indent=2)[:1500] + "\n```")
        else:
            parts.append("- **No measured evaluation metrics were shipped with this model.**")
    parts += ["## Datasets",
              "Bundled training data (150 images per class per disease): TB chest X-rays (Normal / Tuberculosis), "
              "DR fundus images that are already Gaussian-filtered 224 px crops (No_DR / DR — any grade, *not* "
              "referable DR), oral histopathology patches (Normal / OSCC). See `docs/DATASET_DOCUMENTATION.md`.",
              "## Limitations",
              "- Small public datasets; no external or prospective clinical validation.",
              "- Model scores are not calibrated probabilities. No-flag results do not exclude disease.",
              "- Grad-CAM shows model attention only. The DR model may not generalise to unfiltered fundus photos.",
              "- Questionnaire rules and Hindi translations still require clinician / native-speaker review.",
              "## Responsible AI",
              "Screening aid only — never a diagnosis or treatment recommendation. Abstains (INCONCLUSIVE) when "
              "unsure, rejects unusable images, keeps questionnaire logic separate from model output, stores no "
              "images by default, needs no internet, and never fabricates metrics. See `docs/CLINICAL_LIMITATIONS.md`."]
    return "\n\n".join(parts)


# ── Layout ───────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"{config.APP_NAME} — screening aid") as demo:
        outcome_state = gr.State(None)
        with gr.Row(equal_height=True, elem_id="ms-brand-row"):
            gr.HTML(header_html(ctx.registry))
        gr.HTML('<div class="ms-topline"><span>Workspace <span class="ms-slash">/</span> Screening & records</span><span class="ms-local">ON-DEVICE PROCESSING</span></div>')
        with gr.Row(elem_id="ms-toolbar"):
            lang = gr.Radio(LANG_CHOICES, value="en", label="Language / भाषा", scale=0, min_width=200, show_label=False, container=False, elem_id="ms-language")
            gr.HTML(state_banner(ctx.registry), scale=4)
            theme_btn = gr.Button("Light / dark mode", size="sm", elem_id="ms-theme", scale=0)
            theme_btn.click(fn=None, js=TOGGLE_THEME_JS, queue=False)

        with gr.Tabs(elem_id="ms-navigation", overflow_behavior="wrap") as navigation:
            # TAB 1 — Overview
            with gr.Tab("Overview", id="overview"):
                with gr.Row(elem_id="ms-page-heading"):
                    gr.HTML('<div class="ms-page-heading"><div class="ms-eyebrow">WORKSPACE OVERVIEW</div><h2>Screening workspace</h2><p>Review local activity and choose your next screening.</p></div>', scale=4)
                    start_btn = gr.Button("New screening", variant="primary", scale=0, min_width=170)
                    start_btn.click(lambda: gr.update(selected="screening"), outputs=navigation)

                stats = gr.HTML(overview_stats())
                gr.HTML('<div class="ms-section-heading"><h3>Screening modules</h3><span>Three imaging modalities · One workflow</span></div>')
                gr.HTML(modalities_html(ctx.registry))
                with gr.Accordion("Screening workflow · Validation to report", open=False):
                    gr.HTML(how_it_works_html())
                gr.Markdown("#### Model status")
                models_html = gr.HTML(model_table_html(ctx.registry))
                overview_disclaimer = gr.HTML(disclaimer_html("en"))
                gr.Button("Refresh statistics", size="sm").click(lambda: (overview_stats(), model_table_html(ctx.registry)),
                                                                 outputs=[stats, models_html])

            # TAB 2 — New screening
            with gr.Tab("New screening", id="screening"):
                gr.HTML('<div class="ms-page-heading"><div class="ms-eyebrow">NEW SCREENING</div><h2>Prepare a screening</h2><p>Patient details and image input on the left. Results and explanations on the right.</p></div>')
                with gr.Row():
                    with gr.Column(scale=5, elem_id="ms-screening-input"):
                        gr.Markdown("#### 1 · Patient")
                        with gr.Row():
                            patient = gr.Dropdown(patient_choices(), label="Select patient", scale=4)
                            gr.Button("Refresh", size="sm", scale=0, min_width=80).click(
                                lambda: gr.update(choices=patient_choices()), outputs=patient)
                        with gr.Accordion("Register a new patient (no names or phone numbers)", open=False):
                            with gr.Row():
                                ref = gr.Textbox(label="Clinic reference (optional)", placeholder="e.g. PHC-2026-014")
                                age = gr.Number(label="Age (optional)", precision=0, minimum=0, maximum=120)
                            with gr.Row():
                                age_range = gr.Dropdown(AGE_RANGES, label="…or age range")
                                sex = gr.Dropdown(SEX_CHOICES, value="Not recorded", label="Sex (optional)")
                            is_demo = gr.Checkbox(True, label="Demo record (kept separate from real patients)")
                            reg_btn = gr.Button("Register patient")
                            reg_msg = gr.HTML()
                        gr.Markdown("#### 2 · Screening module")
                        module = gr.Radio(MODULE_CHOICES, value="auto", label="Routing", elem_id="ms-module",
                                          info="Manual selection never skips image validation.")
                        gr.Markdown("#### 3 · Symptoms & risk factors (optional)")
                        with gr.Accordion("Symptom questionnaire", open=False):
                            tb_cb = gr.CheckboxGroup(bool_choices("tb", "en"), label="TB: " + t("q.header", "en"))
                            with gr.Row(visible=True) as dr_row:
                                dr_type = gr.Radio([(t("opt.type1"), "type1"), (t("opt.type2"), "type2"),
                                                    (t("opt.other"), "other")], label=t("q.dr.diabetes_type"))
                                dr_years = gr.Number(label=t("q.dr.diabetes_years"), minimum=0, maximum=100)
                                dr_comp = gr.Checkbox(label=t("q.dr.eye_complications"))
                                dr_last = gr.Radio([(t("opt.lt1y"), "lt1y"), (t("opt.gt1y"), "gt1y"),
                                                    (t("opt.never"), "never")], label=t("q.dr.last_eye_exam"))
                            oral_cb = gr.CheckboxGroup(bool_choices("oral", "en"), label="Oral cancer risk factors")
                        gr.Markdown("#### 4 · Image")
                        upload = gr.File(label="Upload image (PNG / JPEG / BMP / TIFF / WEBP)",
                                         file_types=["image"], type="filepath", elem_id="ms-upload")
                        with gr.Accordion("Try a sample image", open=False):
                            for name, fname in SAMPLES.items():
                                p = config.SAMPLES_DIR / fname
                                if p.exists():
                                    gr.Button(f"Sample: {name}", size="sm").click(lambda p=p: str(p), outputs=upload)
                        preview_img = gr.Image(label="Preview", height=220, interactive=False)
                        with gr.Row():
                            allow_rescreen = gr.Checkbox(False, label="Allow re-screening of an identical file")
                            do_gradcam = gr.Checkbox(True, label="Generate Grad-CAM")
                        run_btn = gr.Button("5 · Run screening", variant="primary", size="lg")
                    with gr.Column(scale=6, elem_id="ms-screening-result"):
                        gr.Markdown("#### Result")
                        result_out = gr.HTML('<div class="ms-empty"><div class="ms-scan-frame" aria-hidden="true"><span></span></div><div class="ms-eyebrow">RESULT INSPECTOR</div><strong>Awaiting an image</strong><p>Your screening result, model score, and recommended next steps will appear here.</p><div class="ms-empty-foot">01 Upload image <span>/</span> 02 Run screening <span>/</span> 03 Review</div></div>' + disclaimer_html("en"))
                        gr.Markdown("#### Grad-CAM explanation")
                        with gr.Row():
                            g_orig = gr.Image(label="Model input (after preprocessing)", height=210, interactive=False)
                            g_heat = gr.Image(label="Heatmap", height=210, interactive=False)
                            g_over = gr.Image(label="Overlay", height=210, interactive=False)
                        gradcam_msg = gr.HTML()
                        clinical_out = gr.HTML()
                        with gr.Row():
                            report_btn = gr.Button("Generate PDF report", variant="secondary")
                        report_file = gr.File(label="PDF report", visible=False)
                        with gr.Accordion("Validation & routing details (technical)", open=False):
                            details = gr.JSON()

                reg_btn.click(register_patient, [ref, age, age_range, sex, is_demo], [patient, reg_msg])
                upload.change(preview, upload, preview_img)
                module.change(on_module_change, module, [tb_cb, dr_row, oral_cb])
                run_btn.click(run, [upload, patient, module, tb_cb, dr_type, dr_years, dr_comp, dr_last, oral_cb,
                                    allow_rescreen, do_gradcam, lang],
                              [result_out, clinical_out, g_orig, g_heat, g_over, gradcam_msg, details,
                               outcome_state, report_file], show_progress="full")
                report_btn.click(make_report, [outcome_state, lang], report_file)

            # TAB 3 — Patient records
            with gr.Tab("Patient records"):
                gr.Markdown("## Patient records\nFind patients, review screening history, and export reports.")
                with gr.Row():
                    q = gr.Textbox(label="Search by patient ID or reference", scale=4)
                    rec_demo = gr.Checkbox(True, label="Include demo records", scale=1)
                    search_btn = gr.Button("Search", scale=1)
                patients_df = gr.Dataframe(search_patients("", True), interactive=False, label="Patients (click a row)")
                with gr.Row():
                    sel_pid = gr.Number(label="Selected patient ID", precision=0)
                    export_btn = gr.Button("Export history (CSV)")
                    export_all_btn = gr.Button("Export all screenings (CSV)")
                export_file = gr.File(label="Export", visible=False)
                patient_info = gr.Markdown()
                history_df = gr.Dataframe(interactive=False, label="Screening history (click a row)")
                rejections_df = gr.Dataframe(interactive=False, label="Rejected uploads for this patient")
                with gr.Row():
                    sel_rid = gr.Number(label="Selected screening ID", precision=0)
                    rec_report_btn = gr.Button("Regenerate PDF report")
                rec_report = gr.File(label="Report", visible=False)
                result_md = gr.Markdown()
                with gr.Accordion("Delete a demo patient", open=False):
                    gr.Markdown("Only records marked **demo** can be deleted here. Type the patient ID (or reference) "
                                "to confirm. Deletion removes their screenings and is written to the audit log.")
                    with gr.Row():
                        confirm = gr.Textbox(label="Confirm patient ID / reference")
                        del_btn = gr.Button("Delete demo patient", variant="stop")
                    del_msg = gr.Markdown()

                search_btn.click(search_patients, [q, rec_demo], patients_df)
                q.submit(search_patients, [q, rec_demo], patients_df)
                patients_df.select(select_patient, patients_df, sel_pid)
                sel_pid.change(load_history, sel_pid, [history_df, rejections_df, patient_info])
                history_df.select(select_result, history_df, sel_rid)
                sel_rid.change(show_result, [sel_rid, lang], result_md)
                rec_report_btn.click(record_report, [sel_rid, lang], rec_report)
                export_btn.click(export_csv, [sel_pid, rec_demo], export_file)
                export_all_btn.click(lambda d: export_csv(None, d), rec_demo, export_file)
                del_btn.click(delete_patient, [sel_pid, confirm, rec_demo], [del_msg, patients_df])

            # TAB 4 — Analytics
            with gr.Tab("Analytics"):
                gr.Markdown("## Screening analytics\nActivity and outcomes from locally stored records.")
                with gr.Row():
                    an_demo = gr.Checkbox(True, label="Include demo records")
                    an_sim = gr.Checkbox(False, label="Include simulated results")
                    an_btn = gr.Button("Refresh", variant="primary")
                an_cards = gr.HTML()
                with gr.Row():
                    p1 = gr.Plot(label="By module and result")
                    p2 = gr.Plot(label="Rejection reasons")
                p3 = gr.Plot(label="Trend")
                with gr.Row():
                    rej_table = gr.Dataframe(label="Rejection reasons (table view)", interactive=False)
                    an_models = gr.HTML()
                gr.Markdown("All figures are computed from records stored in the local SQLite database. "
                            "Aggregates contain no patient references, ages or image hashes.")
                an_outputs = [an_cards, p1, p2, p3, rej_table, an_models]
                an_btn.click(analytics, [an_demo, an_sim, lang], an_outputs)
                an_demo.change(analytics, [an_demo, an_sim, lang], an_outputs)
                an_sim.change(analytics, [an_demo, an_sim, lang], an_outputs)

            # TAB 5 — About
            with gr.Tab("About & models"):
                gr.Markdown(about_md())

        lang.change(on_lang_change, [lang, outcome_state],
                    [result_out, clinical_out, overview_disclaimer, tb_cb, oral_cb, dr_type, dr_years, dr_comp, dr_last])
        demo.load(analytics, [an_demo, an_sim, lang], an_outputs)
    return demo


if __name__ == "__main__":
    import os
    import uvicorn
    from src.ui.web import create_web_app

    web_app = create_web_app(ctx)
    web_app = gr.mount_gradio_app(
        web_app, build_ui(), path="/workspace", theme=build_theme(), css=CSS, js=THEME_JS,
        allowed_paths=[str(config.SAMPLES_DIR)], footer_links=[],
    )
    uvicorn.run(web_app, host=os.environ.get("MEDISCAN_HOST", "127.0.0.1"),
                port=int(os.environ.get("MEDISCAN_PORT", "7860")))
