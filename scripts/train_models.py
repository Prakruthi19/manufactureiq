"""
ManufactureIQ — Train everything, in order
============================================
    python scripts/download_data.py        # once: SECOM + WM-811K (~1.3 GB download, ~40 MB kept)
    python scripts/build_wafer_features.py # once: 26 features × 172,950 wafers (~10 min)
    python scripts/train_models.py         # this file

Steps:
    1. train_process_monitor.py  — SECOM process-risk model + SPC limits
    2. train_wafer_classifier.py — WM-811K defect-pattern classifier
    3. seed_chromadb.py          — similar-wafer index (training lots only)
    4. build_demo_samples.py     — held-out wafers + runs for the dashboard

Each step prints its own evaluation; metrics land in models/*_metrics.json.
The previous AI4I version of this project lives in scripts/legacy_ai4i/.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["train_process_monitor.py", "train_wafer_classifier.py",
         "seed_chromadb.py", "build_demo_samples.py"]

for step in STEPS:
    print(f"\n{'#' * 70}\n# {step}\n{'#' * 70}", flush=True)
    if subprocess.run([sys.executable, str(HERE / step)]).returncode:
        sys.exit(f"{step} failed")
print("\nAll models trained.")
