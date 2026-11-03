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
