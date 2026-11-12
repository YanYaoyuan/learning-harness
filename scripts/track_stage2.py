#!/usr/bin/env python3
"""Reuse the independent progress and archival observer for the Stage 2 policy study."""
from pathlib import Path
import track_stage1 as observer


if __name__ == "__main__":
    observer.ROOT = Path("runs/variance-diagnostic-v1/stage2")
    observer.OUT = Path("output/variance-diagnostic-v1/stage2")
    observer.main(stage="Stage 2", planned=68, study="stage2",
                  report_script="scripts/stage2_sampling_report.py", timeout_hours=18)
