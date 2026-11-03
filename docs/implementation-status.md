# Implementation status and acceptance checklist

Audit: 3–4 November 2026. Existing dirty Git work was preserved. Backup:
`output/audit-backup-20261103-235143/`, including important files and a consistent
SQLite backup whose integrity check returned `ok`. No patient data was used for
research tests. Operational server remained on port 7860; real subprocess tests
used port 7861 with temporary databases and logs.

| Work item | Status | Evidence / remaining work |
|---|---|---|
| Repository, entry points, schemas, providers, tests, audits inspected | COMPLETED AND VERIFIED | `app.py`, `server.py`, `run_pretrained.py`; core/database/gatekeeper/reports/providers; docs and test inventory |
| Baseline startup, HTTP uploads, three specialist outputs | COMPLETED AND VERIFIED | `launcher-audit.json`, actual `app.py` subprocess |
| Persistent results, patient associations, PDFs, restart recovery | COMPLETED AND VERIFIED | Real HTTP/restart audit plus automated tests |
| Research runtime and three existing bundles | COMPLETED AND VERIFIED | Hash/reference checks, real local specialists, isolated HTTP audit |
| Research database separation | COMPLETED AND VERIFIED | Launcher defaults to `output/research/mediscan.db`; refuses standard operational path; three regression tests |
| Queue bounds, idempotency, cancellation and interrupted-job handling | COMPLETED AND VERIFIED | `test_remote_providers.py` and live idempotency/restart checks |
| TB accuracy improvement | FAILED | Existing candidate failed pilot; baseline and DenseNet missed 6/6 adapted Montgomery positives; additional ViT missed 3/6 |
| New TB candidate acquisition and reproduction | COMPLETED AND VERIFIED | Pinned public TorchScript, hash, author preprocessing and independent adapter tensor/logit parity; not promoted |
| DR five-output execution and export | COMPLETED AND VERIFIED | Strict five-row head; eager/export fixture parity, raw-image smoke checks |
| DR independent patient-level metrics and full grade-label validation | BLOCKED BY EXTERNAL DEPENDENCY | Need labelled, licensed, grouped cohort and authoritative label-map/split provenance; no five-class accuracy claim |
| Complete oral paper pipeline | BLOCKED BY EXTERNAL DEPENDENCY | Demo Swin/head execution differs from BEiT comparison and SHAP 500-feature training; matching artifacts absent from inspected published inventory |
| Oral demo accuracy | FAILED | All 10 local normal samples flagged |
| External-source dataset comparison | PARTIALLY COMPLETED | 12 official NLM images downloaded, hashes and source IDs retained; training overlap unknown |
| Independent clinical validation / retraining | BLOCKED BY EXTERNAL DEPENDENCY | No verified independent multi-disease cohort or original training manifests |
| Candidate promotion | COMPLETED AND VERIFIED | None promoted; original baseline files/configuration preserved |
| Memory and latency measurements | COMPLETED AND VERIFIED | Separate sequential processes in `research-performance.json`; HTTP timings in `launcher-audit.json` |
| Regression and syntax checks | COMPLETED AND VERIFIED | 144 passed; syntax, compilation and dependency checks passed; no accuracy inferred from software tests |

## Architecture and state boundaries

`app.py` provides the studio and Gradio workspace. `server.py` provides the lighter
studio. Both use validation → modality gatekeeper → selected specialist → separate
decision policy → persistence → patient-linked automatic PDF. The router never
scales the disease score. Technical quality rejection is not a negative diagnosis.
`run_pretrained.py` uses direct local research providers and a separate database.
Remote adapters remain optional and are not used by the default or research launchers.

SQLite schema v2 adds provenance and jobs without replacing existing patient/results
tables. Demo and report services maintain their own additive tables. Queue uploads
remain in memory; saved demo scans and report snapshots remain in SQLite. Models
load serially for research requests; runtime allocators may retain memory.

## Concrete changes during this audit

- Fixed research launcher database isolation; preserved explicit disposable DB support.
- Added real process/HTTP/restart verification and sequential per-bundle measurements.
- Added hash-checked labelled-manifest screening into the existing evaluator's CSV schema.
- Added optional, versioned CLAHE preprocessing for the newly investigated TB ViT.
- Added launcher and preprocessing regression tests; no clinical thresholds changed.
- Corrected stale README/research documentation and recorded failed experiments.
- Ignored local dataset downloads, research databases and backups to avoid accidental commits.

## Acceptance assessment

The operational software and local research workflow are verified within the tested
paths. The research objective remains incomplete: models are demonstrably fallible,
and independence, full oral reproduction and full DR cohort evaluation remain unresolved.
This is not a claim of clinical readiness or 100% completion. Docker/cloud deployment
is outside the selected local track and remains unverified.
