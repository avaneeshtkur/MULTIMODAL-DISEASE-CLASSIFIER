# Executed testing report

Audit completed 4 November 2026 (Asia/Kolkata). Existing operational server stayed
on port 7860. All write/restart/research testing used isolated databases.

## Commands and actual results

| Command / check | Status | Actual result |
|---|---|---|
| `.venv/bin/python -m pytest -q` | COMPLETED AND VERIFIED | 144 passed in 15.01 seconds; one existing Pillow deprecation warning |
| Launcher regression subset | COMPLETED AND VERIFIED | 3 passed; default research isolation, operational-path refusal, explicit disposable path |
| New TB preprocessing tests | COMPLETED AND VERIFIED | 2 passed; constant-image finite output and deterministic standardized grayscale channels |
| Launcher + provider subset | COMPLETED AND VERIFIED | 22 passed before the two preprocessing tests were added |
| `.venv/bin/python scripts/verify_launchers.py` | COMPLETED AND VERIFIED | Both actual entry points passed HTTP/inference/PDF/idempotency/restart checks; disposable DBs, port 7861 |
| `.venv/bin/python scripts/verify_local_pretrained.py --out output/local-pretrained-reverification.json` | COMPLETED AND VERIFIED | All three existing candidates passed real inference, persistence and PDF checks; no accuracy assertion |
| `.venv/bin/python scripts/benchmark_research.py` | COMPLETED AND VERIFIED | Four separate sequential CPU processes measured |
| `node --check web/static/js/app.js` | COMPLETED AND VERIFIED | Exit 0 |
| `.venv/bin/python -m compileall -q src services scripts server.py run_pretrained.py app.py` | COMPLETED AND VERIFIED | Exit 0 |
| `.venv/bin/python -m pip check` | COMPLETED AND VERIFIED | No broken requirements |
| `git diff --check` | COMPLETED AND VERIFIED | Exit 0 |
| Production preservation | COMPLETED AND VERIFIED | Current logical SQL dump SHA256 exactly matches audit backup; both integrity checks `ok` |

The full suite covers score separation, positive-class mapping, five-grade score
aggregation, missing/corrupt/mismatched models, preprocessing, image validation,
clinical translations, migrations, jobs, capacity, idempotency, cancellation,
interrupted-job recovery, wrong remote identities, timeouts, malformed responses,
PDF errors/retries, and report snapshots. Synthetic graph tests verify engineering
behavior, not medical performance. New TB reference-source comparison was also
executed on two real research fixtures: tensor tolerance 1e-6 and logits 1e-5.

## Real launcher timings

| Entry point | First startup | Restart | TB screening + PDF | DR screening + PDF | Oral screening + PDF |
|---|---:|---:|---:|---:|---:|
| `app.py` baseline | 3.905 s | 4.522 s | 0.704 s | 0.527 s | 0.375 s |
| `run_pretrained.py` research | 1.045 s | 4.301 s | 0.583 s | 8.874 s | 3.386 s |

Cached PDF downloads were 0.002–0.008 seconds. The screening figures include PDF
creation, persistence and HTTP polling; standalone PDF generation latency was not
isolated. All six outputs were actual specialist results with matching model
identity. Corrupted uploads returned rejection without a disease score. Completed
jobs and exact PDF bytes survived process restarts. No queue-capacity or concurrency
claims beyond the tested bounded single-worker behavior are made.

## Model resource measurements

From `research-performance.json`: each model used its own fresh process; load time
includes hash verification and reference-image checks. A single subsequent image
was timed. These are observations, not distributions or hard memory bounds.

| Model | Load + verification | One inference | Peak RSS (decimal GB) |
|---|---:|---:|---:|
| TB DenseNet | 0.385 s | 0.047 s | 0.190 |
| RETFound | 4.712 s | 0.262 s | 1.323 |
| Oral Swin/head | 2.799 s | 0.152 s | 1.079 |
| Additional TB ViT | 1.339 s | 0.024 s | 0.399 |

The repeated integrated three-candidate check took 12.33 seconds, maximum RSS
1.307 GB and peak footprint 1.544 GB (`/usr/bin/time -l`). No simultaneous large-model
loading or speculative optimization was introduced. Repeated requests still reload
research weights; allocator retention remains possible.

## Preserved data and limitations

`database-preservation-audit.json` records the matching logical SQL hashes. The
backup is `output/audit-backup-20261103-235143/mediscan.db`. No production patient,
demo, report, or screening rows changed during this audit. The running baseline
status endpoint remains available with all four installed models.

Independent DR patient-level performance: BLOCKED BY EXTERNAL DEPENDENCY.
Complete oral paper reproduction: BLOCKED BY EXTERNAL DEPENDENCY.
Accuracy improvement / promotion: FAILED for the tested candidates; no promotion.
External training independence: not established. The source-derived 12-case TB
comparison is exploratory. Automated browser visual interaction was not exercised;
real HTTP flows, static syntax and existing UI tests were. Docker and cloud inference
were not part of this local-only verification.
