"""Read-only operational counts for frozen scoring and environment pilots."""

import json
import math
from pathlib import Path

from .common import file_sha
from .webshop_pilot import recover_aggregate
from .webshop_requests import summarize_requests

STUDIES = {
    "nll": {
        "root": "runs/nll-scoring-diagnostic-v1",
        "freeze_sha256": "d07f9cfdaa743984b09503c74810c68c82c0342df0da9d09f5657ca731872e43",
        "controller": "run_nll_scoring_probe.py",
        "cap_gpu_seconds": 600,
    },
    "webshop": {
        "root": "runs/webshop-development-pilot-v1",
        "freeze_sha256": "3e5759a36d5e35370a5c930daa68ab8e7326b8ba43c2e2288b1bba8a5270e316",
        "controller": "run_webshop_pilot.py",
        "cap_gpu_seconds": 3600,
    },
}
TERMINAL = {"completed", "failed", "interrupted", "stopped_at_cap"}
STATES = TERMINAL | {
    "not_started",
    "running",
    "evaluation_completed",
    "preparing_cpu_backend",
    "starting_dedicated_base_server",
}
PHASES = ("fp32", "direct_checks", "stress", "legacy", "uniform")
USAGE = (
    "attempted_requests",
    "response_records",
    "error_records",
    "pending_requests",
    "unknown_usage_requests",
    "known_input_tokens",
    "known_output_tokens",
)
PUBLIC_FIELDS = (
    "state",
    "freeze_sha256",
    "started",
    "terminal",
    "audited_complete",
    "gpu_seconds_estimate",
    "cap_gpu_seconds",
    "needs_attention",
    "live_verified_processes",
    "raw_records_uploaded",
    "progress_only",
)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def safe_counts(values):
    """Only named finite nonnegative integer counters can cross the cloud boundary."""
    result = {}
    for key, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError("Invalid operational count")
        result[key] = value
    return result


def cloud_metrics(observations):
    """Flat W&B history metrics; local process records cannot enter this projection."""
    metrics = {}
    for kind in STUDIES:
        if kind not in observations:
            continue
        public = observations[kind]["public"]
        for key in PUBLIC_FIELDS:
            metrics[f"progress/{kind}/{key}"] = public[key]
        for key, value in public["counts"].items():
            metrics[f"progress/{kind}/counts/{key}"] = value
    return metrics


def snapshot(kind, spec, now, process):
    root = Path(spec["root"])
    if file_sha(root / "freeze.json") != spec["freeze_sha256"]:
        raise ValueError("Frozen follow-up identity changed")
    frozen = read(root / "freeze.json")
    ledger = read(root / "execution.json")
    if ledger and ledger.get("freeze_sha256") != spec["freeze_sha256"]:
        raise ValueError("Execution and freeze identities disagree")
    state = ledger.get("status", "not_started")
    if state not in STATES:
        raise ValueError("Unrecognized execution state")
    handles = {}
    if kind == "nll":
        controller = read(root / "job.json").get("pipeline", {})
        if controller.get("pid"):
            handles["controller"] = process(controller["pid"], [spec["controller"]])
        for index, item in enumerate(ledger.get("open", {}).values()):
            handles[f"worker{index}"] = process(
                item["pid"], [spec["controller"], "base-model-readonly-scoring"]
            )
        elapsed = ledger.get("gpu_seconds", 0) + sum(
            max(0, now - item["started_unix"])
            for item in ledger.get("open", {}).values()
        )
        journal = root / "cases/base-model-readonly-scoring/forwards.private"
        usage = summarize_requests(journal)
        counts = {phase: 0 for phase in PHASES}
        if journal.exists():
            for line in journal.read_bytes().splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                if event["kind"] == "response":
                    phase = event["response"]["phase"]
                    if phase not in counts:
                        raise ValueError("Unknown scoring phase")
                    counts[phase] += len(event["response"]["scores"])
        counts = {f"{phase}_observations": count for phase, count in counts.items()}
        counts.update({key: usage[key] for key in USAGE})
        report = read(
            Path("output") / root.relative_to("runs") / "aggregate-summary.json"
        )
        audited_complete = (
            report.get("freeze_sha256") == spec["freeze_sha256"]
            and report.get("complete") is True
            and report.get("base_model_files_after_verified") is True
        )
    elif kind == "webshop":
        if ledger.get("pid"):
            handles["controller"] = process(ledger["pid"], [spec["controller"]])
        for role, item in ledger.get("children", {}).items():
            scripts = [Path(s).name for s in item["argv"] if s.endswith(".py")]
            handles[role] = process(item["pid"], scripts)
        elapsed = ledger.get("gpu_seconds", 0)
        if ledger.get("gpu_started_unix") and not ledger.get("stopped_unix"):
            elapsed = max(elapsed, max(0, now - ledger["gpu_started_unix"]))
        report = recover_aggregate(root, frozen)
        counts = {
            "completed_episodes": report["completed_episodes"],
            "planned_episodes": 64,
        }
        for key in USAGE:
            counts[key] = sum(group[key] for group in report["groups"].values())
        audited_complete = report["complete"] is True and bool(
            ledger.get("stopped_unix")
        )
    else:
        raise ValueError("Unknown follow-up study")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid charged-time estimate")
    terminal = state in TERMINAL
    missing = [
        name
        for name, value in handles.items()
        if not value["alive"] or not value["identity_verified"]
    ]
    needs_attention = (
        bool(missing and not terminal)
        or bool(ledger and not handles and not terminal)
        or (state == "completed" and not audited_complete)
    )
    # No source dictionary, process argv, task key, prompt, NLL or result text is copied.
    public = dict(
        state=state,
        freeze_sha256=spec["freeze_sha256"],
        started=bool(ledger),
        terminal=terminal,
        audited_complete=audited_complete,
        gpu_seconds_estimate=float(elapsed),
        cap_gpu_seconds=spec["cap_gpu_seconds"],
        needs_attention=needs_attention,
        live_verified_processes=sum(
            v["alive"] and v["identity_verified"] for v in handles.values()
        ),
        counts=safe_counts(counts),
        raw_records_uploaded=False,
        progress_only=True,
    )
    return dict(public=public, local_processes=handles)
