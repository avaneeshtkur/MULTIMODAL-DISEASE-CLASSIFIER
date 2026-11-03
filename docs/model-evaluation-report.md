# Model and dataset evaluation report

Audit: 3–4 November 2026. No threshold tuning, score inflation, retraining or operational
model replacement was performed. Software parity is distinct from medical performance.

## TB: actual additional evaluation

Downloaded 12 original images directly from the [NLM Montgomery collection](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html).
The source describes 138 PA chest radiographs, 80 normal and 58 with TB-consistent
manifestations. This radiographic endpoint is not automatically microbiologically
confirmed active TB. The resource permits research use and requests citation;
no broader commercial/redistribution licence is assumed.

Six images per class were selected before inference by sorting
`SHA256("mediscan-audit-20261103" + filename)`. Original source filenames identify
cases; hashes, URLs and labels are retained in `output/external/montgomery/manifest.json`.
No exact duplicates were allowed. Patient identity beyond the source case identifier,
near duplicates, and overlap with model pretraining remain unverified.

All 12 originals exceeded the app's 16-million-pixel input limit: zero coverage,
no disease scores. Those outcomes are retained in `montgomery-*-evaluation.json`.
For a separate experiment, the same images were resized offline with PIL LANCZOS
to a maximum side of 2048, retaining original hashes. This does not change the app
limit, restore image detail, or make the cohort independent.

| Model, unchanged cutoff | TP | FN | TN | FP | Sensitivity | Specificity | AUROC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline EfficientNet | 0 | 6 | 6 | 0 | 0% | 100% | 0.4444 | 0.5234 |
| BioMedCLIP-distilled DenseNet | 0 | 6 | 6 | 0 | 0% | 100% | 0.8333 | 0.8000 |
| Additional public ViT | 3 | 3 | 5 | 1 | 50% | 83.33% | 0.5833 | 0.7202 |

All adapted cases reached a decision. Wilson 95% sensitivity intervals: baseline
and DenseNet 0–39.03%; ViT 18.76–81.24%. ViT specificity interval: 43.65–96.99%.
These are very small-sample exploratory results, not independent validation.
Full outputs are `montgomery-adapted-{baseline,densenet,vit}.json`.
The earlier local DenseNet pilot remains 7 TP / 15 FN / 21 TN / 1 FP.

The [additional ViT author's repository](https://huggingface.co/sukhmani1303/tuberculosis-vit-model)
provides TorchScript, explicit normal/TB mapping and CLAHE/blur/z-score preprocessing,
under an Apache-2.0 model-card declaration. Pinned revision:
`4388ed11fbfd0b0bc5f554920a9c9708485bd1d2`; downloaded model SHA256:
`45055328533628af9e42371ffa4462c5817ca756954ea64488b12e10a19c5105`.
The reviewed author handler and the separate RGB-to-BGR adapter reproduce tensors
within 1e-6 and logits within 1e-5 on two fixtures. Its already exported TorchScript
was used; original eager-training/export parity is not independently established.
Author benchmark values are not published in its README. It is an evaluation-only
bundle at `output/research-bundles/tb-vit`, not a validated replacement.

TB-Net's official checkpoint link remains unavailable (404). The original baseline
also shows a serious generalization problem; retaining it preserves operation and
does not certify its accuracy.

## DR

The APTOS-fine-tuned RETFound head has five rows, strictly matching its backbone.
Input is RGB, Resize shorter side 256 with bicubic interpolation, CenterCrop 224,
ImageNet normalization. Original eager versus TorchScript outputs were checked.
The declared endpoint is any DR (grades 1–4), not referable DR or DME.
Four raw mirrored APTOS examples scored 94.82%, 97.05% (mild) and 5.57%, 8.09%
(normal). Possible training overlap prevents accuracy claims. The conventional
0–4 grade map still requires authoritative training-folder/label provenance before
reporting full five-class or referable-DR performance.

Independent per-class confusion matrices, sensitivity, specificity, AUROC, average
precision and confidence intervals are NOT available. They were not fabricated.
[Messidor-2's official distributor](https://www.adcis.net/en/third-party/messidor2/)
lists 874 paired-eye examinations / 1748 images, research/education-only use,
no redistribution, a personal-email/institution access form, and no included DR
ground-truth labels. Pairing is provided; separately sourced labels and overlap
review are required. No identity or access request was submitted on the user's
behalf. The IDRiD challenge page returned HTTP 403 in this audit.

## Oral

The 20-image local demo pilot remains 10 TP, 0 FN, 0 TN, 10 FP. These are augmented
tiles without patient/slide independence. Strict loading and demo-path parity pass;
useful disease discrimination does not.

Inspected [author code](https://github.com/rudradcruze/oral-cancer) at commit
`9a9d814153f82f97b4e4005045c4220580fd6975`:
`3_Model_Comparison.py` consumes BEiT feature CSVs; `4_Feature_Selection.py` consumes
Swin features and trains a distinct 500-feature MultiScaleCNN. The HF demo combines
a Swin extractor with its 1024-feature attention-CNN head, omitting SHAP selection.
Its StandardScaler object is unused in the main prediction path. The reviewed
training scripts do not establish that simply adding a scaler would repair it.
Matching feature CSVs, head lineage, training label mapping and complete paper-path
artifacts are absent from the inspected published file inventory. No scaler,
feature subset, missing weight, or compatibility fallback was invented.

The [Mendeley source dataset](https://data.mendeley.com/datasets/ftmp4cvtmb/1)
contains 1224 images from 230 patients under CC BY 4.0: at 100×, 89 normal and
439 OSCC; at 400×, 201 normal and 495 OSCC. This is the paper's source data,
not an independent external cohort. Patient/slide mapping and original split
manifests have not been obtained. Downloading another copy would not solve independence.

## Reproducible evaluation inputs

Actual measured CSVs: `output/external/montgomery/{baseline,densenet}.csv` and
`output/external/montgomery/resized2048/{baseline,densenet,vit}.csv`.
Their columns exactly match `scripts/evaluate_external.py`:
`patient_id,source,split,image_sha256,label,status,score,model_version`.
An empty header-only template is `deploy/external-patients.template.csv` (not data).
One row per patient is required; aggregate eyes/tiles using a declared rule before
evaluation. Preserve missing scores for rejected/failed cases. Exact image hashes
and patient uniqueness are checked; near-duplicate/pretraining overlap needs review.

```sh
.venv/bin/python scripts/screen_evaluation_manifest.py \
  output/external/montgomery/resized2048/manifest.json \
  --source NLM-Montgomery-adapted2048-overlap-unknown --disease tb \
  --config output/research-bundles/tb-vit/client-identity.json --out /tmp/vit-new.csv
.venv/bin/python scripts/evaluate_external.py /tmp/vit-new.csv --out /tmp/vit-new-results.json
```

Use new output paths: these commands refuse overwriting evaluation evidence.
All model promotions remain withheld pending acceptable independent evidence.
