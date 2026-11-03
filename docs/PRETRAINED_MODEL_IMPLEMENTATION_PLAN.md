# Pretrained specialist models: evidence and implementation plan

Research date: 3 November 2026. Status: research-serving infrastructure implemented;
candidate medical-model replacements are pending verified artifacts and external evaluation.
The user's current choice is direct local inference in the existing server, supported
by `local_research`. Remote hosting recommendations below are historical alternatives.
See [local setup and implementation status](RESEARCH_DEPLOYMENT.md).

## 1. Decision and current state

Keep the application and specialist inference on the current local computer. Load verified research bundles on demand using `local_research`, with serialized execution and release after use. Require explicit disease-specialist selection; deterministic checks assess technical quality and coarse image-type suitability. The legacy modality gatekeeper is not called. Disease scores come from the selected disease specialist, with provider, version, input contract and positive-class definition recorded.

The screening pipeline checks suitability, then calls only `ctx.modules[selected_disease].predict(...)`. The legacy modality classifier is not part of score generation or routing. `src/core/inference_engine.py` obtains the specialist output. `src/core/external_adapter.py` adapts model artifacts; it is not a remote inference client.

The installed gatekeeper has three supported modality classes and lacks a learned unsupported-image class. Training on chest X-ray, fundus and histopathology images teaches modality recognition, not necessarily whether an image is diagnostically informative. A clear normal scan is informative too.

The prior audit of the supplied external TB image found a specialist score of 15.97%, including direct ONNX inference, below the existing 25.75% cutoff. This is not evidence that the gatekeeper suppressed the score. See `external-tb-audit.json` and `SCORING_AND_REPORTS.md`. Model generalization, preprocessing and training provenance remain issues to investigate.

Already implemented and preserved: named persistent demo cases, percentage displays, separate image-quality feedback, automatic named PDF screening reports and regression tests. This proposal does not replace working code or change thresholds.

## 2. Model shortlist and evidence

Benchmark values below describe particular studies, populations and model versions. Accuracy, AUROC, sensitivity and specificity are different measures and must not be ranked as interchangeable percentages. Checkpoint listings and API documentation were inspected; no commercial account was provisioned, no candidate weights were executed, and no patient images were submitted to external services.

### Tuberculosis: CAD4TB or qXR for a commercial integration

An independent 2026 study of 3,901 symptomatic adults across seven countries found CAD4TB specificity of 73.5% at 90% sensitivity; qXR and Lunit INSIGHT CXR also met 70% specificity at that sensitivity. Results varied by country and subgroup, with persistent difficulties in people living with HIV, people aged at least 50, and people with previous TB. These are stronger deployment-relevant results than a small curated image benchmark, but still do not imply detection of every positive. [Study](https://pubmed.ncbi.nlm.nih.gov/41973983/)

**Preferred first commercial adapter: CAD4TB.** Delft documents cloud API v2 with vendor-issued base URL, token and archive identifiers. Its workflow uploads frontal chest X-ray DICOM series, polls processing status and fetches results. Pricing, access, deployed version and exact score semantics require a vendor agreement. [API documentation](https://delft-cv-africa-static.s3.af-south-1.amazonaws.com/cv-af-south-001-ctb-001/help/service_manual/api/v2.html)

**Alternative: qXR.** Its documented input requirements include PA/AP chest radiographs, valid DICOM and minimum 1440 × 1440 resolution. The supplied 561 × 740 Google image does not satisfy that documented contract. Upscaling or wrapping it in DICOM does not restore acquisition quality or make it a validated input. General qXR age eligibility must not be confused with a specific TB-screening indication. [Input specification](https://documentation.qure.ai/api/platform-api/get-started/qxr)

WHO's 2025 CAD guidance covers TB screening in people aged 15 and above, with confirmatory testing after positive screening. Verify the selected product/version and intended population separately. [WHO announcement](https://www.who.int/news/item/11-06-2025-who-approves-six-software-products-for-computer-aided-detection-of-tb-on-chest-x-ray)

### Tuberculosis: TB-Net for a research prototype

The authors publish a pretrained 4.24-million-parameter TB-Net checkpoint and report 99.86% accuracy and 100% sensitivity on their benchmark. Those results do not establish comparable external clinical performance. Use it as an independently evaluated research candidate, not a guaranteed improvement over the current specialist. [Model table](https://github.com/darwinai/TuberculosisNet/blob/main/docs/models.md)

The reference repository uses an old TensorFlow stack. Isolate it in a remote container; do not downgrade this project's Python environment. Pin the checkpoint and preprocessing. If converting to ONNX, compare original and converted outputs before adoption. Resolve the repository's inconsistent license heading/body and checkpoint usage terms before redistribution. Check for overlap between its training data and this project's TB data. [Repository](https://github.com/darwinai/TuberculosisNet)

### Diabetic retinopathy: RETFound with an existing DR head

Use a disease-fine-tuned checkpoint, not the foundation encoder alone. The official repository lists checkpoints and splits for APTOS2019, IDRID and MESSIDOR-2. Artifact availability is documented; download integrity, class order and reproduction are implementation prerequisites. [Official checkpoints](https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md)

The original paper reports DR AUROC of 0.943 on APTOS2019, 0.822 on IDRID and 0.884 on MESSIDOR-2. Crucially, an APTOS-fine-tuned model achieved only 0.738 AUROC when externally tested on MESSIDOR-2. This supplies a concrete domain-shift problem for the project to investigate. These numbers are AUROC, not classification accuracy. [Original paper, author-hosted copy](https://pure-oai.bham.ac.uk/ws/files/208272053/s41586_023_06555_x.pdf)

Host RETFound remotely. Pin the exact backbone and checkpoint rather than treating every model added to the evolving repository as equivalent to the original paper. Check the noncommercial license and weight-specific conditions. Define whether the output is any DR or referable DR; a five-grade head can support a declared grade aggregation after class-order verification, but does not automatically establish diabetic macular edema detection. A fundus model detects retinal disease, not diabetes itself. [Repository and license](https://github.com/rmaphoh/RETFound)

### Diabetic retinopathy: EyeArt for a commercial alternative

The pivotal EyeArt study enrolled 942 people at 15 sites. Its observed undilated eye-level results for more-than-mild DR included 95.5% sensitivity, 85.0% specificity and 87.4% imageability. The dilate-if-needed pathway increased imageability to 97.4%. This illustrates why acquisition and handling ungradable images matter alongside classifier performance. These figures are not guarantees for arbitrary web images. [Pivotal study](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2786132)

Treat EyeArt as a vendor integration investigation. A currently usable public self-service API, price, raw scalar output and supported camera contract have not been verified. Do not promise a percentage if a licensed interface provides only a categorical decision.

### Oral squamous cell carcinoma: research candidate, conditional on reproduction

A 2026 Swin/ViT histopathology paper reports 99.25% accuracy and 99.21% sensitivity on its public-dataset experiment, with downloadable artifacts and a FastAPI demonstration. Independent clinical or multi-site performance was not established by this review. This is a candidate to audit, not an equivalent substitute for a clinically validated service. [Paper](https://www.mdpi.com/2227-7080/14/1/39), [model artifacts](https://huggingface.co/rudradcruze/oral-cancer)

Inspection of the public demo code identified concrete reproduction risks: its normal prediction path returns maximum-class confidence separately from OSCC probability, can retry checkpoint loading with `strict=False`, and omits SHAP feature selection used elsewhere in the project. These observations do not prove the weights are inaccurate, but prevent assuming that the deployed path reproduces the paper. Require strict compatible weight loading, verified preprocessing and head architecture, and independently reproduced outputs. Read the positive-class probability rather than maximum-class confidence. [Inspected API source](https://huggingface.co/spaces/rudradcruze/oral-cancer/blob/main/app.py)

Do not make this public demonstration endpoint a production dependency. If reproduction succeeds and licensing permits, deploy a pinned private copy. Keep it disabled if the pipeline cannot be reproduced.

Fallback: use a frozen histopathology encoder such as Phikon with a separately trained OSCC head. Phikon supplies tissue features, not an already validated oral-cancer classifier; this path explicitly requires labelled-data training and evaluation. Check its usage terms. [Official model card](https://huggingface.co/owkin/phikon)

## 3. Proposed architecture

```text
Patient + original scan
    → file/security checks and metadata extraction
    → supported modality / unsupported-image detection
    → modality-specific gradability and provider-input checks
    → one selected specialist provider
        chest X-ray     → CAD4TB / qXR / research TB-Net
        retinal fundus  → fine-tuned RETFound / licensed EyeArt
        oral H&E patch  → verified OSCC checkpoint
    → preserve specialist output + explicit decision policy
    → patient-linked result and automatic screening PDF
```

Maintain independent fields for modality confidence, image suitability, raw specialist score and screening decision. Quality checks may withhold an unreliable result; they must not artificially reduce the disease probability toward a reassuring negative. A clear healthy image should have good quality and a low disease score. A blurred positive should become ungradable when evidence is insufficient, not automatically negative.

Inputs supported initially: frontal chest radiographs, colour retinal fundus images, and oral H&E histopathology patches within the selected model's domain. CT, MRI, oral photographs, report screenshots and arbitrary graphs are unsupported unless a separately validated pipeline is later added. Whole-slide pathology requires explicit patch extraction and slide aggregation; a single-patch model must not claim a patient-level whole-slide diagnosis.

The initial quality gate combines deterministic corruption/exposure checks with modality recognition. Improve it with independently labelled supported, unsupported and ungradable examples. Include clear negatives, real acquisition defects, documents, graphs and unrelated medical modalities. Validate the gate's rejection of valid positives as carefully as its rejection of irrelevant inputs. Manual disease selection must not bypass provider eligibility checks.

## 4. Score and provider contract

Create `src/providers/base.py`, `local_onnx.py`, `remote_research.py`, `cad4tb.py` and, if selected, `qxr.py`. Retain current local models as explicitly selectable baselines, not silent substitutes during an outage.

Each provider declares input types, modality, acquisition requirements, disease endpoint, version, score meaning, positive-class mapping, preprocessing version, synchronous/asynchronous behavior and supported explanation outputs. Store:

```text
job_id, patient_id, image_sha256, provider_request_id
provider, model_version, checkpoint_sha256 (when available)
modality, gradability, eligibility_reason
target_definition, class_labels, raw_output, positive_class_score
score_kind, calibration_version, decision_policy_version
status, decision, timestamps, latency, report_token
```

Use statuses `queued`, `running`, `completed`, `ungradable`, `unsupported`, `failed` and `cancelled`. Missing scores remain null. Timeouts, authentication failures and malformed outputs never become zero scores or negative findings.

For a genuine 0–1 positive-class output, display `100 × score` with an honest label such as “uncalibrated model score.” A native 0–100 CAD index may need its own label; it is not automatically disease probability. Preserve categorical-only provider outputs. Never invent probabilities or multiply the disease score by routing confidence.

Threshold means the cutoff used to turn a numerical score into a flag. Lowering it changes sensitivity and specificity; it does not raise the model's score or improve its ability to distinguish disease. Choose research thresholds using validation data, then freeze them before testing. Keep licensed vendor decision policies within the agreed intended use. If calibration is evaluated, retain both raw and calibrated outputs and fit only on separate calibration data.

## 5. Concrete repository changes

| Area | Planned change | Acceptance check |
|---|---|---|
| `src/core/pipeline.py` | Split validation/routing from provider submission and result finalization | Exactly one eligible specialist; no score modification by router |
| `src/core/model_registry.py` | Provider configurations and lazy optional local loading | Remote mode loads no disease weights |
| `src/core/image_preprocessor.py` | Versioned per-model transforms; original DICOM ingestion path | Reference/adapter output parity; no double normalization |
| `src/gatekeeper/` | Unsupported and gradability support | Valid-positive rejection and irrelevant-image acceptance measured |
| `src/ui/web.py` | Job submission/status endpoints and bounded uploads | UI stays responsive; retries do not duplicate screening |
| `src/database/` | Versioned migration for jobs, provenance and provider results | Existing patient/demo/report records survive migration |
| `web/static/js/app.js` | Pending/error/ungradable states and provider labels | No false negative while waiting or on failure |
| `src/reports/automatic.py` | Completed-job report with provider, target and provenance | Patient and specialist output match persisted snapshot |
| `config.py` and environment | Explicit provider selection and server-side credentials | No tokens in browser, source or logs |
| New remote service directories | Isolated pinned model containers | Exact checkpoints, strict loading and health/version endpoints |
| New evaluation scripts/tests | Patient-grouped, external and corruption evaluations | Reproducible manifests and confidence intervals |

The current app primarily accepts ordinary image files. DICOM support is a real ingestion task: preserve the original, handle pixel interpretation correctly, de-identify appropriately, and retain required acquisition metadata. Never fake missing clinical metadata to satisfy a provider.

Keep patient names locally where the provider does not need them. Use pseudonymous request IDs, authenticated private transport, bounded retention and explicit remote-processing configuration. Public demo endpoints are for synthetic/nonpatient integration checks only.

## 6. Working within 8 GB RAM

Local machine: web UI, database, HTTP client and bounded image decoding. No gatekeeper model is needed for routing. Avoid importing PyTorch, TensorFlow or local Grad-CAM in remote-only mode. Cap worker concurrency and decoded-image dimensions; stream large files instead of retaining multiple copies.

Proposed local acceptance target: under 1 GB idle process memory and under 2 GB peak for one supported ordinary-image screening, measured on this laptop. These are targets, not measured guarantees. Whole-slide images require a separate remote storage/tiling design.

Research hosting can use a private remote worker with lazy loading and a bounded queue. Serverless scale-to-zero reduces idle compute but adds cold-start latency. Commercial APIs remove model hosting work but require a quote. No free, unlimited, clinically validated API for all three diseases was verified. Track cost per completed screening, warm/cold latency and retry charges before choosing a hosting tier.

## 7. Evaluation and the project's research contribution

**Research question:** can input-contract enforcement, improved gradability handling and measured domain adaptation improve pretrained specialists on unseen sources while keeping useful screening coverage?

1. Freeze current specialists and record hashes as the baseline. Preserve the supplied external TB failure as a regression example, not a threshold-tuning dataset.
2. Build manifests with source, patient/slide identifier, modality, label provenance, acquisition device and parent image. Split by patient and site before augmentation; keep both eyes together and all pathology patches from a slide together. Deduplicate exact and near-duplicate images across splits and known pretraining/evaluation collections.
3. Separate development, calibration/threshold validation and final external test sets. The existing demo collection is for UI demonstrations, not accuracy measurement. The local TB set and reconstructed DR/oral splits do not establish independence from pretrained model training.
4. Reproduce each candidate's reference inference on licensed fixtures. Verify positive-class ordering, RGB/grayscale handling, image size, normalization and activation. For converted models, predeclare numerical tolerances and verify decision agreement near the cutoff.
5. Test frozen candidates unchanged on the same eligible external cohort. Report patient-level sensitivity, specificity, AUROC, AUPRC, confusion matrices, calibration where meaningful, and confidence intervals. Report ungradable rate and the fraction of all reference-positive patients actually flagged end to end; do not hide rejected positives by excluding them from denominators.
6. Run paired corruption tests for blur, compression, contrast and exposure, including clean positives and negatives. Evaluate score instability and gradability handling. Do not demand that disease score monotonically decreases with degradation; that would conflate disease evidence and image quality.
7. Evaluate improvements separately: exact preprocessing; quality/unsupported gate; validation-set calibration; locally appropriate thresholds; then permitted open-model adaptation. For RETFound compare a frozen head with small-head or parameter-efficient tuning using remote compute. For oral histology test stain handling and slide-level grouping. For TB include old TB and other lung disease among negatives, subject to reliable reference labels.
8. Use ablations so each claimed improvement is attributable. Compare sensitivity at matched specificity, specificity at matched sensitivity, and coverage at matched error rates. Calibration should be judged with reliability/Brier-type measures, not advertised as improved discrimination.
9. Lock the chosen configuration before a final held-out evaluation. Bootstrap at patient/slide level, not augmented-image level. Report failures and subgroup uncertainty. An external study with no observed misses still does not prove perfect sensitivity.

Use a feasibility pilot first; calculate a larger sample size from desired confidence-interval precision and endpoint prevalence. Subgroup conclusions require enough cases. Published benchmarks guide hypotheses but do not set guaranteed project outcomes.

## 8. Execution sequence and release gates

Indicative engineering sequence, dependent on data and provider access:

1. **Contracts and baseline (week 1):** freeze manifests and current outputs; choose research versus commercial track; obtain checkpoint/API access and licenses; define each clinical target.
2. **Provider abstraction (week 2):** local adapter, mock remote responses, persisted jobs and failure handling. Preserve existing demo/report behavior.
3. **Remote TB and DR (weeks 3–4):** pin and reproduce research weights, or integrate contracted APIs. Add original-format support as required. Measure memory and latency.
4. **Oral reproduction and input gate (weeks 4–5):** audit the full oral checkpoint chain. If it fails, keep that provider disabled and use the separately planned encoder/head study; do not substitute unsupported predictions.
5. **External evaluation and adaptations (weeks 5–7):** baseline comparisons, subgroup analysis, calibration and targeted adaptation experiments. Data collection may take substantially longer.
6. **Integration release (week 8, conditional):** finalize provenance, reports, documentation and deployment tests. Release research functionality only with its limitations; clinical deployment requires its own evidence and intended-use review.

Required tests include wrong-modality rejection, normal-class high-confidence mapping, unavailable provider, malformed/NaN scores, repeated callbacks, retries, model version changes, report/patient association, database migration, and raw-score preservation. Quality rejection must not be represented as a negative diagnosis.

No candidate is promoted solely because its headline benchmark is higher. Promotion requires accessible licensed artifacts/API, reproducible inference, a matching input contract, successful engineering tests and an acceptable external sensitivity/specificity/coverage tradeoff. The report remains an automated screening result for review, not a definitive diagnosis.

## 9. Recommended starting configuration

For an academic prototype with minimal local hardware, start with private remote TB-Net and a published DR-fine-tuned RETFound checkpoint as evaluated candidates, retaining the current ONNX specialists for comparison. Treat the oral Swin/ViT model as conditional on reproduction. If that fails, explicitly scope the additional OSCC-head training work.

For a funded clinical integration, investigate CAD4TB first, qXR as an alternative, and EyeArt for DR. Oral histopathology remains a separate research-validation workstream in this review. Commercial access, licensing, dataset provenance and actual performance are unresolved dependencies—not completed integrations.
