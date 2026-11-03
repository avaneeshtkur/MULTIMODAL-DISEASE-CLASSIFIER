# Clinical limitations and responsible use

> This is an experimental AI screening aid, not a medical diagnosis. Results must be reviewed by a
> qualified healthcare professional and confirmed through appropriate clinical testing.

## Intended use

Decision support for trained health workers in screening settings: deciding **who needs further
evaluation**. It is not for diagnosis, staging, treatment selection, or ruling disease out.

## What the outputs mean

| Status | Meaning | It does **not** mean |
|---|---|---|
| `SCREENING_FLAGGED` | the model score is at or above the validated threshold, outside the uncertainty band | the patient has the disease |
| `NO_FLAG_DETECTED` | the score is below the threshold, outside the band | the patient is healthy; disease is ruled out |
| `INCONCLUSIVE` | near the threshold, out of distribution, invalid output, or model unavailable | anything about the patient |
| Rejected | the image was not analysed | — |

The **model score** is the network's output (after sigmoid/softmax), not a calibrated probability.
Thresholds target ≥ 95% sensitivity *on a small validation split*, so real-world sensitivity may be lower.

## Known limitations

- Trained on about 150 images per class from public datasets, with no external, prospective or
  multi-site validation.
- **TB:** a single frontal CXR. TB can be present with a normal-appearing CXR (e.g. early, HIV co-infection).
  Microbiological confirmation (NAAT/CBNAAT, sputum) is required.
- **DR:** trained on Gaussian-filtered images labelled *any DR*. It is not validated on raw clinic fundus
  photos, does not grade severity, does not detect macular oedema, and ungradable images are possible.
- **Oral:** classifies histopathology patches, so it does not replace a pathologist's read of the whole
  slide. Pre-augmented training data may inflate metrics.
- Grad-CAM shows model attention, not lesions. Attention on irrelevant regions (labels, borders) is a
  warning sign.
- Quality checks are heuristic and can miss subtle problems such as rotation, wrong projection or poor focus
  in one region.
- No subgroup (age/sex/device) performance analysis is possible with the available data.

## Clinical decision support

- Recommendation text is fixed per disease and status. It names referral and confirmatory tests (e.g.
  NAAT/CBNAAT, dilated eye exam, OCT when indicated, pathologist review or biopsy when advised) and **never
  medicines or treatment**.
- Questionnaire rules (`TB-Q1…`, `DR-Q1…`, `ORAL-Q1…`) are derived from public screening guidance (WHO
  TB symptom screen and India NTEP presumptive-TB criteria, routine diabetic eye examination intervals, and
  the 2-week red flag for oral lesions). **They have not yet been reviewed by a clinician for this
  deployment**, and that review must happen before clinical use.
- The questionnaire never changes the model output, and no combined probability is computed.

## Language

Hindi strings are hand-written in `src/clinical/translations.py`; no machine translation is used at
runtime. **They are pending review by a native-speaking clinician.**

## Privacy and safety

- Minimal patient data: an internal ID, an optional reference code (names and phone numbers are rejected by
  format), age or age range, and optional sex.
- Images are not stored by default. Only the SHA-256 is kept, for duplicate detection. A hash match means
  identical files, never the same patient.
- Demo records are flagged separately. Only demo records can be deleted in the app, and deletions are audited.
- No external APIs. All processing happens locally.
- Rejected images are never used for retraining.

## Before any clinical pilot

1. Train on larger, representative data. Use group-aware splits for oral data.
2. Validate externally on local images from the target devices and calibrate thresholds locally.
3. Have clinicians review the recommendations, questionnaire rules and Hindi text.
4. Obtain ethics, regulatory (e.g. CDSCO SaMD) and data-protection (DPDP Act 2023) review.
5. Plan human oversight, incident reporting and performance monitoring.
