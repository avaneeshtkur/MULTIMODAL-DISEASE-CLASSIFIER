"""
pipeline.py — end-to-end screening workflow.

    upload → Stage A image suitability → duplicate check → explicitly selected specialist
    → modality-specific quality → selected local/remote specialist → optional explanation
    → clinical decision support → persistence → automatic PDF in the web workflow

The legacy learned modality gatekeeper is not used for routing or scoring.
Every exit path is recorded: either a screening_results row or a
rejection_events row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.clinical.triage_engine import triage
from src.clinical.translations import t
from src.core.model_registry import ModelRegistry
from src.database.analytics_service import AnalyticsService
from src.database.db import Database
from src.database.patient_service import PatientService
from src.database.screening_service import ScreeningService
from src.providers import build_providers, load_provider_config
from src.gatekeeper import rejection_manager as rm
from src.gatekeeper.classifier import InputSuitabilityGate, ModalityDecision
from src.gatekeeper.duplicate_detector import check_duplicate, duplicate_rejection
from src.gatekeeper.image_quality import modality_blur_check, validate_image
from src.utils.image_hash import short_hash
from src.utils.logging_config import get_logger

log = get_logger("pipeline")

MODULE_TO_MODALITY = {v: k for k, v in config.MODALITIES.items()}


@dataclass
class ScreeningOutcome:
    stage: str                                 # "rejected" | "screened"
    lang: str = "en"
    session_id: int | None = None
    result_id: int | None = None
    rejection_id: int | None = None
    rejection: rm.Rejection | None = None
    image_sha256: str | None = None
    quality_metrics: dict = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)
    modality: ModalityDecision | None = None
    result: dict | None = None
    explanation: dict | None = None
    clinical: dict | None = None
    duplicate_note: str | None = None
    patient: dict | None = None

    @property
    def status(self) -> str:
        return "REJECTED" if self.stage == "rejected" else self.result["prediction_status"]


@dataclass
class AppContext:
    registry: ModelRegistry
    db: Database
    patients: PatientService
    screenings: ScreeningService
    analytics: AnalyticsService
    suitability_gate: InputSuitabilityGate
    modules: dict


def build_context(db_path=None, models_dir=None, provider_config=None) -> AppContext:
    settings = load_provider_config(provider_config)
    # Remote specialists never validate or load local disease weights.
    # The old modality classifier is not an informativity model and no longer participates.
    keys = [k for k, v in settings.items() if v['kind'] == 'local_onnx']
    registry = ModelRegistry(models_dir=models_dir, keys=keys)
    db = Database(db_path)
    modules = build_providers(registry, settings)
    db.sync_model_registry([m.status_row() for m in modules.values()])
    return AppContext(registry, db, PatientService(db), ScreeningService(db), AnalyticsService(db),
                      InputSuitabilityGate(), modules)


def routing_text_for(disease: str, source: str | None, confidence: float | None, lang: str = "en") -> str:
    modality = MODULE_TO_MODALITY.get(disease)
    if not modality or not source:
        return "—"
    text = f"{t(f'modality.{modality}', lang)} — {t(f'routing.{source}', lang)}"
    return text + (f" ({confidence:.2f})" if confidence is not None else "")


def routing_text(d: ModalityDecision | None, lang: str = "en") -> str:
    if d is None or d.modality is None:
        return "—"
    return routing_text_for(d.disease_key, d.source, d.confidence, lang)


def run_screening(ctx: AppContext, image_bytes: bytes, filename: str, patient_id: int,
                  module: str = "auto", answers: dict | None = None, lang: str = "en",
                  allow_rescreen: bool = False, explain: bool = True) -> ScreeningOutcome:
    patient = ctx.patients.get(patient_id)
    if patient is None:
        raise ValueError("Select or register a patient before screening.")
    is_demo = bool(patient["is_demo"])
    answers = {k: v for k, v in (answers or {}).items() if v}
    q_disease = module if module in config.SPECIALIST_KEYS else None
    session_id = ctx.screenings.create_session(patient_id, module, lang, q_disease,
                                               answers or None, is_demo)
    out = ScreeningOutcome("rejected", lang, session_id, patient=patient)

    def reject(rej: rm.Rejection) -> ScreeningOutcome:
        out.rejection = rej
        out.rejection_id = ctx.screenings.save_rejection(session_id, patient_id, rej, module,
                                                         out.image_sha256, is_demo)
        log.info("session %s rejected: %s (hash %s)", session_id, rej.code, short_hash(out.image_sha256 or ""))
        return out

    # Stage A — deterministic validation
    q = validate_image(image_bytes, filename)
    out.image_sha256, out.quality_metrics, out.quality_warnings = q.sha256, q.metrics, q.warnings
    if not q.ok:
        return reject(q.rejection)

    # Duplicate (exact file) check
    dup = check_duplicate(q.sha256, patient_id, ctx.screenings)
    rej = duplicate_rejection(dup, allow_rescreen)
    if rej:
        return reject(rej)
    if dup.is_duplicate:
        out.duplicate_note = ("Identical file screened before " +
                              ("for this patient." if dup.same_patient else "under another record.") +
                              " Re-screened deliberately; this does not establish patient identity.")
        ctx.db.audit("duplicate_rescreened", "screening_session", session_id, {"same_patient": dup.same_patient})

    # Disease selection is explicit. The old learned model classifies modality,
    # not informativeness, so it must not choose or influence the disease model.
    if module not in config.SPECIALIST_KEYS:
        return reject(rm.Rejection(rm.DISEASE_SELECTION_REQUIRED, rm.STAGE_MODALITY))

    # Stage B — deterministic suitability checks for the selected specialist.
    decision = ctx.suitability_gate.assess_selected(module, q.metrics)
    out.modality = decision
    if decision.rejection:
        return reject(decision.rejection)
    disease = module

    rej = modality_blur_check(q.metrics, decision.modality)
    if rej:
        return reject(rej)

    sharpness_floor = config.QUALITY["blur_min_by_modality"].get(decision.modality, config.QUALITY["blur_min_global"])
    if q.metrics.get("blur_laplacian_var", 0) < 2 * sharpness_floor:
        out.quality_warnings.append("Limited sharpness: close to the image rejection cutoff. Use a sharper original image; disease score is not image-quality confidence.")

    spec = ctx.modules[disease]
    if not spec.available and not config.SIMULATION_MODE:
        return reject(rm.Rejection(rm.MODEL_UNAVAILABLE, rm.STAGE_MODEL,
                                   {"module": disease, "setup": spec.setup_message()}))

    # Specialist inference
    result = spec.predict(q.image, q.metrics, image_sha256=q.sha256)
    out.result = result
    if explain:
        out.explanation = spec.explain(q.image, result)

    # Clinical decision support (does not modify `result`)
    out.clinical = triage(disease, result["prediction_status"], answers.get(disease) if answers else None, lang)
    rec_en = " | ".join(triage(disease, result["prediction_status"], None, "en")["image"]["recommendations"])

    out.result_id = ctx.screenings.save_result(
        session_id, patient_id, result, q.sha256, q.metrics, decision.source, decision.confidence,
        rec_en, out.clinical["referral"]["priority"], bool(out.explanation and out.explanation.get("ok")), is_demo)
    out.stage = "screened"

    if config.RETAIN_IMAGES:
        config.RETAINED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        (config.RETAINED_IMAGES_DIR / f"{out.result_id}_{q.sha256[:16]}.png").write_bytes(image_bytes)
    log.info("session %s → result %s: %s %s (%.0f ms)", session_id, out.result_id, disease,
             result["prediction_status"], result["processing_time_ms"])
    return out


def report_from_outcome(o: ScreeningOutcome, lang: str) -> dict:
    """Assemble report fields for a just-completed screening (re-translated for `lang`)."""
    r = o.result
    disease = r["disease"]
    clinical = triage(disease, r["prediction_status"], (o.clinical or {}).get("questionnaire", {}).get("answers"), lang)
    reason = t(f"reason.{r['reason_code']}", lang) if r.get("reason_code") else ""
    if r.get("simulated"):
        reason = t("simulated_banner", lang)
    return {
        "result_id": o.result_id, "patient_id": o.patient["id"], "patient_ref": o.patient.get("reference"),
        "created_at": None, "disease": disease, "status": r["prediction_status"], "score": r.get("model_score"),
        "threshold": r.get("threshold"), "uncertainty_band": r.get("uncertainty_band"),
        "score_kind": r.get("score_kind"), "model_version": r.get("model_version"),
        "routing": routing_text(o.modality, lang), "quality": t("pdf.quality_ok", lang), "reason": reason,
        "recommendations": clinical["image"]["recommendations"],
        "questionnaire_rules": clinical["questionnaire"]["rules"] if clinical["questionnaire"]["answers"] else None,
        "referral_label": clinical["referral"]["label"], "simulated": r.get("simulated"),
        "processing_time_ms": r.get("processing_time_ms"),
        "provenance": {k: r.get(k) for k in ('provider', 'provider_request_id', 'provider_status',
            'target_definition', 'positive_class', 'preprocessing_version', 'decision_policy_version', 'raw_positive_score')},
    }


def report_from_record(rec: dict, lang: str) -> dict:
    """Rebuild report fields from a stored screening (images are not retained → no Grad-CAM)."""
    disease = rec["disease"]
    answers = rec["questionnaire"].get(disease) if rec.get("questionnaire") else None
    clinical = triage(disease, rec["screening_status"], answers, lang)
    reason = t(f"reason.{rec['reason_code']}", lang) if rec.get("reason_code") else ""
    if rec.get("simulated"):
        reason = t("simulated_banner", lang)
    routing = routing_text_for(disease, rec.get("routing_source"), rec.get("modality_confidence"), lang)
    return {
        "result_id": rec["id"], "patient_id": rec["patient_id"], "patient_ref": rec.get("reference"),
        "created_at": rec["created_at"], "disease": disease, "status": rec["screening_status"],
        "score": rec.get("model_score"), "threshold": rec.get("threshold"), "uncertainty_band": rec.get('provenance', {}).get('uncertainty_band'),
        "score_kind": "SIMULATED (not a model output)" if rec.get("simulated") else rec.get('provenance', {}).get('score_kind'),
        "model_version": rec.get("model_version"), "routing": routing, "quality": t("pdf.quality_ok", lang),
        "reason": reason, "recommendations": clinical["image"]["recommendations"],
        "questionnaire_rules": clinical["questionnaire"]["rules"] if answers else None,
        "referral_label": clinical["referral"]["label"], "simulated": bool(rec.get("simulated")),
        "processing_time_ms": rec.get("processing_time_ms"),
        "provenance": rec.get('provenance', {}),
    }
