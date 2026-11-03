"""
Create a demo patient and a screening history by running the REAL pipeline on
data/samples/. Nothing is inserted by hand: every row is the genuine outcome
(model result, INCONCLUSIVE, or rejection) for the models installed right now.
With MEDISCAN_SIMULATION=1, modules without models produce rows flagged
simulated=1, which analytics exclude by default.

    python scripts/make_demo_assets.py && python scripts/seed_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.core.pipeline import build_context, run_screening  # noqa: E402

PLAN = [
    ("sample_chest_xray.png", "tb", {"tb": {"cough_2w": True, "fever": True}}),
    ("sample_fundus.png", "dr", {"dr": {"diabetes_type": "type2", "diabetes_years": 12, "last_eye_exam": "never"}}),
    ("sample_histopathology.jpg", "oral", {"oral": {"tobacco": True, "ulcer_2w": True}}),
    ("invalid_blurry.png", "tb", None),
    ("unsupported_synthetic.png", "tb", None),
    ("unsupported_synthetic.png", "auto", None),
]


def main():
    ctx = build_context()
    existing = ctx.patients.search("DEMO-0001")
    pid = existing[0]["id"] if existing else ctx.patients.register("DEMO-0001", age=52, sex="Female", is_demo=True)
    print(f"Demo patient #{pid} (DEMO-0001)")
    for fname, module, answers in PLAN:
        path = config.SAMPLES_DIR / fname
        if not path.exists():
            print(f"  missing {path}; run scripts/make_demo_assets.py first")
            continue
        o = run_screening(ctx, path.read_bytes(), fname, pid, module, answers, allow_rescreen=True, explain=False)
        what = o.rejection.code if o.rejection else o.result["prediction_status"] + \
            (" (SIMULATED)" if o.result.get("simulated") else "")
        print(f"  {fname:28s} module={module:5s} → {o.stage}: {what}")


if __name__ == "__main__":
    main()
