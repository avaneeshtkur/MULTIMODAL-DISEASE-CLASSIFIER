# Local research inference setup

## Verified progress: 3 November 2026

All three public candidate bundles are now downloaded, strictly loaded, exported
where necessary, and verified through local inference, database persistence and
automatic PDF generation. No external inference service is used. The existing
server and baseline models remain unchanged because two candidates failed pilots.

| Candidate | Actual check | Replacement status |
|---|---|---|
| BioMedCLIP-distilled DenseNet-121 TB v11 | Author preprocessing/ONNX parity passed; local test: 7 true positives, 15 missed positives, 21 true negatives, 1 false positive | Failed pilot; evaluation only |
| APTOS-fine-tuned RETFound | Strict five-grade head, eager/export parity; two raw mild-DR images scored 94.82% and 97.05%, two normal images 5.57% and 8.09% | Runnable research candidate; four examples do not validate accuracy |
| Published oral Swin + attention-CNN demo | Strict extractor/head loading and export parity; all 20 sampled images flagged, including 10 normal images | Failed pilot; evaluation only; paper's full SHAP/scaling pipeline not reproduced |

TB-Net's official checkpoint folder returned 404, also reported in upstream
[issue 6](https://github.com/darwinai/TuberculosisNet/issues/6). The downloaded
[DenseNet alternative](https://huggingface.co/Ressshh/tb-conformal-distilled-densenet121)
is a different model; its author-reported benchmark claims are not independently
verified. It scored the supplied external TB image at 99.91%, but its 15 local
misses prevent claiming it improves accuracy. No threshold was fitted to that image.

The raw DR images are mirrored APTOS examples, potentially overlapping training;
they are software smoke tests, not external validation. Existing filtered fundus
demos do not establish performance on raw fundus images or vice versa. The oral
pilot uses local augmented images without verified slide independence. Detailed
outputs are in `pretrained-tb-pilot.json`, `pretrained-dr-pilot.json`,
`pretrained-oral-pilot.json` and `pretrained-integration-audit.json`.

For explicit **research evaluation of all three candidates**, stop the existing
server before running:

```sh
.venv/bin/python run_pretrained.py
```

This uses the same local address, `http://127.0.0.1:7860`, displays an evaluation
notice, and defaults to a separate `output/research/mediscan.db` database. It refuses
the standard operational `data/mediscan.db` path even if explicitly selected.
It does not promote these candidates as better diagnostic models. Restart
`.venv/bin/python app.py` without `MEDISCAN_PROVIDER_CONFIG` to use the preserved
baseline. No model is silently substituted during a failure.

Weights and bundles are ignored under `output/pretrained/` and
`output/research-bundles/`. `deploy/providers.pretrained.local.json` selects the
three bundles. Build dependency versions are recorded in
`requirements-research-build.txt`; the existing environment already has them.
`scripts/verify_local_pretrained.py` repeats the real application
and PDF checks against an isolated temporary database, without modifying patient
records. The measured three-model check took 12.03 seconds, with maximum RSS
1.36 GB and peak memory footprint 1.54 GB; these are single-run measurements, not
an upper bound. Models run sequentially and are released after each request.

The remaining accuracy work is substantive: reproduce the oral training-time
feature selection/scaling artifacts or choose another reproducible specialist;
find a TB candidate that passes a meaningful evaluation; and evaluate DR on an
independent patient-grouped cohort. Increasing displayed scores does not fix this.

## Current choice: everything runs on this computer

The subsequent audit is recorded in `implementation-status.md`,
`model-evaluation-report.md` and `testing-report.md`: 144 tests passed, both actual
launchers passed disposable-database HTTP/restart checks, and an additional TB ViT
was reproduced but not promoted. Research mode now defaults to a separate database.

The website and specialist inference run in the existing local server at
`http://127.0.0.1:7860`. No cloud host, domain, SSH account, API subscription,
Docker installation or separate model service is required. The remote-service
instructions below are retained as an optional alternative, not a prerequisite.

The default configuration already uses the installed local ONNX specialists.
To use a verified replacement bundle directly in the same server, create a provider
configuration such as:

```json
{
  "tb": {"kind": "local_research", "bundle": "research-bundles/tb"},
  "dr": {"kind": "local_onnx"},
  "oral": {"kind": "local_onnx"}
}
```

Bundle paths are relative to the configuration file. Start the existing application
with `MEDISCAN_PROVIDER_CONFIG=providers.local.json MEDISCAN_ORT_THREADS=2 .venv/bin/python app.py`.
Use `server.py` instead of `app.py` for the lighter studio without the optional Gradio
workspace. Stop the existing instance before starting another on port 7860.

`scripts/package_research_bundle.py` now generates a **local** `client-identity.json`
configuration automatically. It requires neither a URL nor a token. You can point
`MEDISCAN_PROVIDER_CONFIG` directly at that generated file. Omitted disease keys use
their current ONNX specialists.

Local research models load on demand, run serially, and release their model objects
after each request. This increases load latency but avoids caching all research
weights simultaneously. Runtime libraries can retain allocated memory, so fitting
large RETFound checkpoints within 8 GB still requires measurement; no memory limit
or accuracy improvement is guaranteed. TensorFlow is absent from the current Python
environment; a TB-Net bundle needs a compatible local runtime or verified conversion.

Only replacement model files, their reference preprocessing/class mapping, and
reference outputs are needed to integrate new models. Independently labelled data
are needed to evaluate improvement. These can be obtained from the published sources;
you do not need to provide server credentials or a hosting budget.

Implemented locally: explicit provider adapters; bounded background jobs; durable status,
restart recovery and idempotency; model/scan identity checks; percentage scores and
separate quality outcomes; persisted raw outputs and provenance; automatic PDFs;
private model service; packaging/export tools; external-cohort evaluation.

The selected track is local research inference. Downloaded candidates can now run
in explicit evaluation mode; none has established improved external accuracy.
The installed ONNX models remain the operational baseline. Commercial vendor integrations and DICOM acquisition
are deferred for this track; the application still accepts ordinary supported images.

## Run the application

Existing installation with the optional Gradio workspace:

```sh
MEDISCAN_ORT_THREADS=2 .venv/bin/python app.py
```

For a separate lightweight installation, install `requirements-runtime.txt` and run
`python server.py`. This entry point omits Gradio, training dependencies and local
PyTorch explanations. Use one application process per database; a file lock enforces
one queue owner. `MEDISCAN_PROVIDER_CONFIG` selects a JSON file, with omitted disease
keys using the current local ONNX baseline. Remote failures never fall back locally.

`deploy/providers.local.example.json` illustrates an all-local configuration. Runtime
secrets belong in environment variables. `providers.local.json` and `.env*` are ignored
by Git. The loader does not automatically read `.env`; export variables through your
shell or secret manager. Keep the local UI bound to loopback unless separately secured.

## Prepare a verified model bundle locally

1. Obtain the official checkpoint and review its usage terms. Record its source and
   upstream commit. Do not use a foundation encoder without the disease classifier.
2. Run the original implementation on at least two distinct, licensed reference images
   and save its raw logits. Include positive and negative examples and cases near the
   decision cutoff when available. This is reproduction evidence, not clinical validation.
3. Reproduce exactly the preprocessing, class order, target and decision policy. Choose
   the policy on separate validation data. Templates deliberately contain null thresholds
   so they cannot silently install an arbitrary cutoff.
4. Package the files and independent reference outputs. The service checks all hashes
   and reruns every reference fixture before becoming ready. A failure keeps it offline.

Example reference file format (values must come from the original model, not this example):

```json
[
  {"file": "negative.png", "raw_output": [2.1, -1.4]},
  {"file": "positive.png", "raw_output": [-0.7, 1.9]}
]
```

For a verified ONNX export:

```sh
python scripts/package_research_bundle.py --settings model-settings.json \
  --model model.onnx --references reference-logits.json --out verified-bundle
```

Settings have the same provenance/class/preprocessing/policy fields as the templates,
with `runtime` set to `onnx`. External ONNX weight files must be included explicitly
using `--extra-file`. For a TorchScript export, use `runtime: torchscript` and the
exported `.pt` file. Bundle creation refuses an existing output directory.

TB-Net uses its reviewed, pinned `preprocessing.py`, `model_eval.meta`, `model.index`
and `model.data-00000-of-00001`. Include the latter three supporting files with
`--extra-file`; set `--model model_eval.meta`. Its corner masking and cropping are
materially different from generic EfficientNet preprocessing. The container uses
TensorFlow compatibility APIs; equivalence to the old reference environment must
pass the real-image fixture checks. Compatibility has not been established for the
published checkpoint in this session.

For RETFound, export in the reviewed upstream environment using:

```sh
python scripts/export_retfound.py --reference-dir /path/to/RETFound \
  --revision FULL_REVIEWED_COMMIT_SHA --checkpoint /path/to/dr-finetuned.pth \
  --constructor RETFound_mae --num-classes 5 --out /path/to/dr-export.pt
```

Select the constructor matching the pinned repository revision. The exporter requires
a matching classifier head and strict weight loading. A tensor sanity check is followed
by the independent reference-image checks during packaging. Grade order must be
verified before summing classes 1–4 for any DR, or 2–4 for a separately defined referable
DR endpoint. No DME claim is implied. The settings example uses the documented evaluation
transform; verify it against the exact selected checkpoint's training code.

The oral demo path has been exported and verified, but the full paper pipeline is
not reproduced. The comparison training script consumes BEiT features whereas the
demo uses Swin; the SHAP study uses a separate 500-feature MultiScaleCNN. Do not
equate successful demo execution with reproduction of the paper's accuracy.
`deploy/candidates.json` records the failed pilot.

## Optional alternative: separate private services (not needed locally)

On a Linux server with Docker, choose `MODEL_RUNTIME=onnx`, `torchscript` or `tbnet_tf`.
Install only that runtime. Set the bundle path and a private random token of at least
32 characters in the server environment, then run:

```sh
docker compose -f deploy/compose.research.yml up --build -d
```

Required environment: `MEDISCAN_BUNDLE_PATH` (absolute path),
`MEDISCAN_SERVICE_TOKEN`, and optionally `MODEL_RUNTIME` / `MEDISCAN_SERVICE_PORT`.
The service listens on server loopback, uses a read-only model mount, has a bounded
temporary filesystem and a 4 GB container memory limit. Measure actual model memory;
large exports may require a different limit. Run each specialist as a separate Compose
project/port. Configure an HTTPS reverse proxy with authentication forwarded and a
26 MB maximum request body, or use a private SSH tunnel to the loopback port. No public
patient-upload endpoint is provisioned by this repository.

The server authenticates requests, accepts one inference at a time, strips patient
information from the client payload, and never logs uploaded images or API keys. The
API is `POST /v1/predict` with multipart fields `disease`, `request_id`, `image` and a
Bearer token. `GET /health` is authenticated and exposes the exact model identity.
The service checks basic image quality and conservative modality consistency; these
checks are not a trained diagnostic gradability model.

For this optional alternative only, construct a `remote_research` configuration with
the identity returned by authenticated `/health`, a URL and a token environment name.
The packager's generated `client-identity.json` uses direct local inference instead.
Set the named token environment variable and start the local application with:

```sh
MEDISCAN_PROVIDER_CONFIG=providers.local.json MEDISCAN_ORT_THREADS=2 .venv/bin/python server.py
```

The API token must already be exported. A remote configuration means “configured”,
not “verified reachable.” Successful inference requires matching model version, bundle
hash, preprocessing identity, target and decision-policy identity. `model_sha256` for
a remote provider identifies the complete pinned bundle and output mapping, not only
one weights file. Wrong-model, malformed, redirected, oversized and timed-out responses
produce no score and an inconclusive result.

## Job and storage behavior

The UI submits `POST /api/jobs` with an `Idempotency-Key`, polls `GET /api/jobs/{id}`,
and reconnects to the pending job after a same-tab refresh. Reusing a key with different
input returns 409. One running plus two waiting requests bounds memory. Queued jobs
can be cancelled through `POST /api/jobs/{id}/cancel`; running remote requests cannot
be safely cancelled after submission. The compatibility `/api/screen` endpoint uses
the same queue.

Uploads remain in memory while queued and are released after processing. Job status,
result and PDF references persist in SQLite. On restart, unfinished jobs become failed;
they are not automatically replayed. A result saved just before interruption may remain
in patient history. Completed jobs and their PDF snapshots remain accessible. Demo scans
retain their existing explicit local storage behavior. Database schema v2 adds tables
without rewriting existing patient, screening or report records.

The legacy gatekeeper checkpoint may remain on disk, but the screening pipeline no
longer loads or calls it. Users select the disease specialist explicitly. Deterministic
quality and coarse modality-compatibility checks determine whether to proceed; they do
not establish clinical gradability. A true informative/uninformative classifier would
need new labelled data and independent evaluation.

## Measure performance before promotion

Use `scripts/evaluate_external.py` on one fixed disease endpoint/model version. Its CSV
columns are documented in `--help`. Supply one row per patient, original source and
split identifiers, labels, statuses and scores; retain ungradable/unsupported/failed
rows with empty scores. Define eye/slide aggregation before creating the file.

```sh
python scripts/evaluate_external.py external-patients.csv --out external-results.json
```

The evaluator rejects duplicated patients/images, reports coverage, reports the fraction
of all reference-positive patients flagged, and separately reports sensitivity/specificity
among decided cases with Wilson intervals. AUROC/average precision use scored cases
with their coverage disclosed. Source overlap is reported. Pretraining overlap,
near-duplicate detection, subgroup sufficiency and clinical ground truth still require
dataset provenance review. Do not interpret synthetic integration tests as medical
accuracy tests. Existing demo cases remain excluded from accuracy claims.

Checkpoints, local measurements and a 12-image official-source Montgomery sample
are now available. Training independence remains unverified; original images exceed
the app pixel limit, and a separately recorded 2048-pixel adaptation was evaluated.
See `model-evaluation-report.md`. Retraining and improved clinical accuracy remain
unestablished. Local tests cover the real HTTP contract,
synthetic ONNX execution, score separation, failure behavior, jobs, database and reports.

## Verification in this workspace

- See `testing-report.md` for the latest executed suite and launcher checks;
  the previous checkpoint was 139 passing tests.
- New tests exercise remote multipart transport through actual synthetic ONNX inference,
  positive-class mapping, five-grade aggregation, wrong model/version responses, timeouts,
  redirects, oversized responses, reference mismatches and immutable report provenance.
- Queue tests cover duplicate submissions, capacity, cancellation, restart recovery,
  persisted results and additive database migration.
- The lightweight entry point imports neither PyTorch nor Gradio. JavaScript passes
  the Node syntax check. Docker is not installed here, so container builds are unverified.
- Before the live database migration, an integrity-checked SQLite backup was saved under
  `output/mediscan-before-providers.bTysCC`. Existing patient/demo/report data were retained.
