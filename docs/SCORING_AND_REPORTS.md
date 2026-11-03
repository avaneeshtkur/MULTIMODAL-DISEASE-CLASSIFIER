# Scoring and automatic reports — 2026-11-03

## What changed

The studio automatically creates a PDF after each completed screening or rejection.
The download link appears beside results. Saved demo names are taken from their
patient record; other patients have a name field for the report. Reports preserve
a snapshot of patient name, result, quality measurements and model version, and
persist in SQLite across server restarts. A renderer failure does not discard the
screening; downloading retries PDF generation. PDF reports are automated screening
findings for clinician review, not diagnostic certificates.

Technical image quality is displayed separately from disease score. Existing
rejection limits remain: minimum side 128 px; maximum aspect ratio 3; sharpness
Laplacian variance minimum 3 for chest X-ray and 30 for fundus/histopathology;
underexposure when the 99th percentile is below 40/255; overexposure when the
1st percentile exceeds 235/255 or more than 85% of pixels are near white.
A new descriptive warning appears when sharpness is less than twice the relevant
rejection floor. This is a technical heuristic, not a calibrated quality score.
Quality warnings never rescale the specialist's output.

## Model audit and threshold decision

The user selects the disease specialist. The legacy modality gatekeeper is not
loaded or called by the screening pipeline. After deterministic suitability checks,
the selected specialist alone produces its disease score; the checks do not multiply,
lower or otherwise modify it. Direct ONNX inference matches the supplied external
image's pipeline score (see external-tb-audit.json).

All production thresholds remain unchanged. Removing the cutoff cannot create
better disease discrimination, and flagging every scan trivially flags all
positives while also flagging every negative. Clear healthy images must not receive
high disease scores just because they are clear. Neither calibration nor threshold
tuning guarantees a high score on every reference-positive internet image.

Run `MEDISCAN_ORT_THREADS=2 python scripts/evaluate_thresholds.py` to reproduce the
research-only comparison in threshold-evaluation.json. Candidate thresholds use
validation scores only; test scores evaluate candidates afterward. Existing TB
split membership was retained. DR and oral splits were reconstructed deterministically
because their split manifests are absent. Independence from the externally supplied
weights' training data is unverified for every local split. These are exploratory
checks, not evidence of clinical accuracy or reliable external generalization.

In the local TB test split (22 positive, 22 negative), the current 25.75% cutoff
flagged 22 positives and one negative. A 6.44% cutoff flagged the same positives
and seven negatives, with four additional negative cases inconclusive. Counting
inconclusive as nonflagged in the JSON is explicitly disclosed; it does not mean
inconclusive is clinically negative. A validation-selected 29.84% candidate left
one test positive inconclusive. No candidate was installed.

The supplied external TB image scored 15.97%; 50% darkening raised its score to
26.72% and mild Gaussian blur (radius 2) raised it to 47.85%. Severe darkening
and blur were rejected. See distortion-audit.json for exact metrics. These results
show model instability under image changes, not improved diagnostic performance.
The mild blur now triggers a limited-sharpness warning. No distorted image was
used to train, select a cutoff or claim improved accuracy.

## Work required before claiming an accuracy improvement

Obtain the original external training split manifests and exact preprocessing
code; preserve patient-level, source-independent validation and test sets. Include
external positives AND negatives across acquisition sources, and unsupported
modalities. Retrain/fine-tune with validated acquisition and degradation augmentations;
evaluate sensitivity, specificity, calibration and abstention on held-out sources.
If automatic unsupported-image detection is needed later, train and validate a
separate informativeness/gradability model with supported, unsupported, and
ungradable examples. Select any new screening cutoff on
validation data only, then test once on held-out data. Current weights have not
been retrained, and the external TB false negative remains unresolved.
