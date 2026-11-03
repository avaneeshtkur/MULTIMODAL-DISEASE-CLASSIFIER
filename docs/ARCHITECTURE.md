# Architecture

## Request flow

`src/core/pipeline.py::run_screening` handles one upload. Every exit path writes exactly one row, to
either `screening_results` or `rejection_events`.

| # | Step | Module | Failure outcome |
|---|---|---|---|
| 1 | Patient lookup and session creation (questionnaire stored on the session) | `database/screening_service.py` | error: patient required |
| 2 | Stage A validation | `gatekeeper/image_quality.py` | `IMAGE_*` rejection |
| 3 | Exact duplicate check (SHA-256 of file bytes) | `gatekeeper/duplicate_detector.py` | `DUPLICATE_IMAGE` (unless "allow re-screening") |
| 4 | Stage B modality decision | `gatekeeper/classifier.py` | `UNSUPPORTED_MODALITY`, `UNCERTAIN_MODALITY`, `MODALITY_MISMATCH` |
| 5 | Modality-specific blur floor | `image_quality.modality_blur_check` | `IMAGE_TOO_BLURRY` |
| 6 | Model availability | `diseases/base.py` | `MODEL_UNAVAILABLE` |
| 7 | Inference: metadata preprocessing → ONNX → score → threshold/band | `core/inference_engine.py` | `INCONCLUSIVE` with reason code |
| 8 | Grad-CAM (optional) | `explainability/gradcam.py` | message only; the prediction is unaffected |
| 9 | Clinical decision support | `clinical/triage_engine.py` | — |
| 10 | Persist result + audit | `database/*` | — |

## Model registry and inference

- `ModelRegistry` validates every `artifacts/models/<key>/` folder at startup with
  `artifact_validator.validate_model_dir`. It checks the metadata schema, the preprocessing spec, the
  positive-class position, the threshold range, SHA-256 (metadata and installed manifest), the ONNX checker,
  the input shape against the preprocessing, the dtype, and runs a test inference to check output width and
  finiteness.
- Statuses are `available`, `missing` and `invalid`. Invalid or missing models never crash the app and never
  block other modules.
- An ONNX Runtime session is created lazily and once per model (thread-safe), then reused for every request.
- PyTorch is only imported for Grad-CAM. Checkpoints are loaded lazily and cached by (ONNX hash, checkpoint hash).

### Common prediction interface

```python
{
  "disease": "tb", "class_label": "Tuberculosis" | None,
  "model_score": 0.83, "score_kind": "uncalibrated model score",
  "threshold": 0.41, "uncertainty_band": [0.36, 0.46],
  "prediction_status": "SCREENING_FLAGGED" | "NO_FLAG_DETECTED" | "INCONCLUSIVE",
  "reason_code": None | "WITHIN_UNCERTAINTY_BAND" | "OUT_OF_DISTRIBUTION" | "PREPROCESSING_ERROR"
                 | "INFERENCE_ERROR" | "MODEL_OUTPUT_INVALID" | "MODEL_UNAVAILABLE",
  "processing_time_ms": 14.2, "model_version": "...", "model_sha256": "...", "simulated": False
}
```

Decision rule: `INCONCLUSIVE` if `threshold − margin < score < threshold + margin` (margin from metadata,
default 0.05). Otherwise `SCREENING_FLAGGED` if `score ≥ threshold`, else `NO_FLAG_DETECTED`. If metadata
declares temperature scaling, it is applied before the threshold.

### Preprocessing parity

`core/image_preprocessor.py` re-implements torchvision's PIL-backend eval transform in NumPy/PIL, so
inference needs no torch. A test (`test_preprocess_matches_torchvision_eval_transform`) checks that the
maximum difference is below 1e-6. During development it measured **0.0**. Unknown preprocessing keys
are rejected rather than ignored.

## Gatekeeper

| Situation | Behaviour |
|---|---|
| Trained gatekeeper, top score ≥ 0.80 and margin ≥ 0.20 | route to that modality |
| Trained gatekeeper, below either limit | `UNCERTAIN_MODALITY` |
| Trained gatekeeper predicts a non-supported class | `UNSUPPORTED_MODALITY` |
| Manual selection + trained gatekeeper disagrees | `MODALITY_MISMATCH` |
| No gatekeeper, automatic mode | `UNCERTAIN_MODALITY`. Automatic routing is disabled; a descriptive *heuristic hint* is shown but never used to route |
| No gatekeeper, manual mode | conservative consistency checks only (multi-coloured image submitted as X-ray; greyscale image submitted as fundus/histopathology), plus a "not verified by a trained classifier" note |

## Image quality thresholds

All live in `config.QUALITY`. They were calibrated on **all 900 bundled images** (300 per disease) so that
every valid training image passes:

| Check | Threshold | Bundled-data evidence |
|---|---|---|
| Extensions / real formats | png, jpg/jpeg, bmp, tif/tiff, webp; the decoded format must match an allowed one | — |
| Max file size | 25 MB, plus the Pillow decompression-bomb guard | — |
| Min side | 128 px | DR images are 224 px |
| Max aspect ratio | 3.0 | all bundled images are square |
| Blur (variance of 4-neighbour Laplacian, grey, longest side ≤ 512 px), global floor | 3.0 | X-ray minimum 5.0. Gaussian-blurred (σ≈3) X-rays max 2.7 |
| Blur floor after routing | X-ray 3.0 · fundus 30 · histopathology 30 | fundus minimum 473, histopathology minimum 148. Blurred σ≈3 versions max 2.8 / 6.9 |
| Underexposed | 99th grey percentile < 40 | X-ray p99 minimum 146 |
| Overexposed | 1st percentile > 235 or > 85% pixels ≥ 250 | histopathology white fraction maximum 0.18 |

Dark backgrounds alone never cause rejection, because exposure is judged on the brightest and darkest 1%.
16-bit images are scaled by bit depth (with a warning), never by per-image contrast stretching.

Manual-mode colour checks use *hue diversity*, the square root of the smaller eigenvalue of the per-pixel
(R−G, G−B) covariance. It is ≈0 for grey or single-tint images, including the cyan-tinted film photos found
in the TB data (max 15.8), and it is ≥ 30 for colourful natural images. The limit is 25.

## Clinical decision support

`triage(disease, image_status, answers)` returns three separate blocks:

- `image`: status and recommendations from `recommendations.py` (a fixed table per disease and status)
- `questionnaire`: rule hits with IDs (`TB-Q1` …) from `symptom_assessment.py`
- `referral`: `PRIORITY` if either source says so, otherwise `ROUTINE`, with each reason attributed

The image-model result is never modified, and no combined probability is computed.

## Database

SQLite, with one short-lived connection per operation, WAL mode, foreign keys and parameterised queries.

| Table | Purpose |
|---|---|
| `patients` | id, optional reference, age/age range, optional sex, `is_demo`, created_at |
| `screening_sessions` | patient, requested module, language, **questionnaire JSON** |
| `screening_results` | disease, image hash, model version/hash, score, threshold, status, reason, quality, routing, recommendation, referral, timing, `simulated`, `is_demo` |
| `rejection_events` | code, stage, module, hash, details |
| `model_registry` | status snapshot at each start |
| `audit_logs` | registration, screening, rejection, report, export, deletion events |

Images are not stored unless `MEDISCAN_RETAIN_IMAGES=1`. Logs contain IDs, codes, truncated hashes and
timings, never images or free text. Only `is_demo` patients can be deleted in the UI, and deletion requires
typing the ID or reference.

## Artifact import

See [MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md#importing). The steps are: archive safety checks →
staging extraction → manifest verification → per-model validation → atomic `os.replace` install with
backup → `installed_manifest.json`, which the registry re-checks at every start.

## Offline / CPU design

- No network calls at runtime. The UI uses system fonts, and PDF fonts are bundled in `assets/fonts`.
- ONNX Runtime CPU provider. Sessions are created once. Torch is loaded only if Grad-CAM is requested.
- The `src/core/pipeline.py` boundary (bytes in, outcome out) is where a future sync or queue layer would attach.
