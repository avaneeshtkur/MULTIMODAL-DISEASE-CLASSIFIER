# Two-minute demo guide

## Preparation (before going on stage)

```bash
source .venv/bin/activate
python scripts/import_artifacts.py --archive trained_outputs.zip   # if the trained models have arrived
python scripts/make_demo_assets.py                                  # data/samples/
python scripts/seed_demo.py                                         # demo patient DEMO-0001 + real history
python app.py                                                       # open http://127.0.0.1:7860
```

- **Models installed:** results are real model outputs on held-out test images.
- **Models not installed:** the app shows *DEMO / INCOMPLETE-MODEL STATE*. You can still demo validation,
  rejection, routing logic, records, analytics and reports. To rehearse the screening screens, start with
  `MEDISCAN_SIMULATION=1 python app.py`. Every result is then marked **SIMULATED**; say so out loud and
  never present it as a screening result.
- Use a fresh database for the demo: `MEDISCAN_DB=data/demo.db python scripts/seed_demo.py` and
  `MEDISCAN_DB=data/demo.db python app.py`.

`seed_demo.py` runs the real pipeline. It never inserts hand-made results.

## Script (≈ 2 minutes)

| Time | Action | What to say |
|---|---|---|
| 0:00 | **Overview** tab | "Offline screening aid for TB, diabetic retinopathy and oral cancer. It runs on this laptop's CPU, with no internet." Point at the model-status table and the disclaimer. |
| 0:15 | **New screening** → *Register a new patient*: reference `PHC-DEMO-1`, age 52, Demo ✓ | "We store minimal data: no names, no images by default." |
| 0:25 | Routing **Automatic**. Tick *Cough ≥ 2 weeks* and *Fever*. Click **Sample: Chest X-ray** → **Run screening** | "Quality checks, then the gatekeeper detects a chest X-ray and routes it to the TB model." Point at the status, score bar, threshold and uncertainty band, and the timing. *(Without a gatekeeper, pick "Tuberculosis — chest X-ray" manually and point out the "not verified by a trained classifier" note.)* |
| 0:50 | Grad-CAM panel | "Where the model looked. That is not proof it's right, and we only show it when the checkpoint matches the deployed model." |
| 1:00 | Recommendations | "Image-based next steps, such as NAAT/CBNAAT sputum testing, sit beside questionnaire rules with rule IDs. They are never merged into a fake combined probability." |
| 1:10 | Switch language to **हिन्दी**, then **Generate PDF report** | "Bilingual, with a clinician sign-off line. It is a screening aid, not a diagnostic certificate." |
| 1:25 | **Sample: Invalid (blurry)** → Run | "Unusable images are rejected with a reason code, an explanation and a fix." |
| 1:35 | **Sample: Unsupported (non-medical)** → Run | "Unsupported images are refused, not forced into a disease model." |
| 1:45 | **Analytics** tab → Refresh | "Every number comes from the local database, including the rejection reasons, so a clinic can see where image quality fails." |
| 1:55 | Close | "It abstains when unsure, explains itself, and keeps the clinician in charge." |

## Backup answers

- **Accuracy?** Quote only the metrics on the *About & models* tab, which come from the imported
  `metadata.json` and the held-out test split. If it says *missing*, say there are no measured metrics yet.
- **Why INCONCLUSIVE?** The score was inside the uncertainty band, or the image was outside the training
  distribution. Abstaining is a feature.
- **Raw fundus photo gives INCONCLUSIVE?** The DR model was trained on Gaussian-filtered images (see the dataset docs).


## Saved demo patient catalog

Run `.venv/bin/python scripts/seed_demo_catalog.py` to create 21 reusable cases
(idempotent): two positive and two negative scans for each of TB, DR and oral
cancer, plus blurred, blank and underexposed quality tests for each modality.
The studio's Patient record and Medical sample dropdowns expose this catalog.
Selecting a saved patient loads its scan and disease module. Create a demo
patient opens a form for a fictional name, age, sex, details, disease, reference
label and scan. Saved scans persist locally in SQLite and reappear after restart.
Real upload retention settings are unchanged; demo uploads are intentionally saved.

Source labels come from the bundled dataset, not model predictions. Local test
split membership does not establish independence from externally supplied model
training. Fictional diabetes histories are record examples, not inferred from
X-rays or DR labels. A negative DR label does not establish absence of diabetes.

Scores and cutoffs display as percentages. API/database numeric values remain
in [0,1]. A threshold is the cutoff for raising a flag, with an inconclusive band
around it. Current cutoffs: TB 25.75%, DR 1.00%, oral 93.06%. None of these models
has supplied probability calibration. A score of 80% is not an 80% disease risk.

## Inference audit (2026-11-03)

See `inference-audit.json`: all 12 labeled cases reached the matching specialist
ONNX model, and pipeline scores matched direct model execution to the rounding
precision (0.00005). Six positive cases flagged; five negative cases had no flag;
one negative DR case scored 1.05% and remained inconclusive near the 1% cutoff.
All nine synthetic quality tests were rejected before specialist inference.
These checks verify wiring, not generalization or medical accuracy.

The gatekeeper only recognizes chest X-ray, fundus and histopathology, with no
other class. Its high confidence cannot certify that arbitrary internet inputs
are supported. The DR model expects Ben Graham filtered images. The oral model
cannot assess CT/MRI, mouth photos, documents or laboratory parameters. No
thresholds, weights, preprocessing or reference labels were altered to force
positive results. The supplied Google share link resolved to a Radiopaedia image from
`https://radiopaedia.org/cases/pulmonary-tuberculosis-78`. The downloaded image's
SHA-256 is `456119bb1fc5bafa2df9f5c038585603eb509ef8b0febe40eeeca4719f755c50`.
Both automatic and manual TB routing produced 15.97%, below the 25.75% threshold,
and NO_FLAG_DETECTED. Direct ONNX execution produced logit -1.6601216793 and
score 0.1597456636, confirming this is the specialist output rather than a UI or
routing substitution. See `external-tb-audit.json`. Against the supplied positive
reference this is a false negative; the source page's full clinical details were
not independently accessible. This remains an unresolved model generalization
issue. Independent representative validation, source-training preprocessing
verification and potentially retraining are required before claiming a fix;
changing a cutoff on this single example is not validation.

To audit another image without adding records to the working database:
`python scripts/audit_scan.py /path/to/scan.png --module auto --output audit.json`.
