#!/usr/bin/env python3
"""Check the other frozen primary contrast and all conditional full-pool repeats."""

import argparse
import json
from pathlib import Path

import run_stage3_task_controls as study
from learning_harness.common import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    frozen = study.load_freeze()
    summary_path = args.snapshot / "summary.json"
    summary = json.loads(summary_path.read_text())
    snapshot = json.loads((args.snapshot / "inputs.json").read_text())
    out = (
        Path("output/variance-diagnostic-v1/stage3/interim-blocks")
        / args.snapshot.name
        / "full-reference-check.json"
    )
    if out.exists():
        raise ValueError("Preserve existing reference comparison")

    def historical(source, init, sample, repeat):
        p = (
            study.stage2.ROOT
            / "cases"
            / f"full_pass-{source}-i{init}-s{sample}-r{repeat}"
            / "worker-result.json"
        )
        if file_sha(p) != frozen["files"][str(p)]:
            raise ValueError("Historical result differs from frozen reference")
        return json.loads(p.read_text()), str(p), file_sha(p)

    rows, provenance = [], {}
    for r in summary["paired_rows"]:
        old, path, sha = historical(
            r["source"], r["initialization_seed"], r["sampling_seed"], 0
        )
        current = next(
            v
            for v in snapshot["records"]
            if v["case"]["arm"] == "random"
            and v["case"]["repeat"] == 0
            and all(
                v["case"][k] == r[k]
                for k in ("source", "initialization_seed", "sampling_seed")
            )
        )
        recipe = json.loads(Path(current["case"]["schedule"]).read_text())
        meta = json.loads((Path(path).parent / "A/experiment.json").read_text())
        if (
            not old["complete"]
            or old["control"]["initial_parameter_sha256"]
            != current["control"]["initial_parameter_sha256"]
            or old["case"]["gpu"] != current["case"]["gpu"]
            or old["control"]["schedule_sha256"] != recipe["reference_schedule_sha256"]
            or meta["training_data_sha256"] != recipe["data_sha256"]
            or old["exposure"]["processed_examples"]
            != current["exposure"]["processed_examples"]
        ):
            raise ValueError("Historical pairing, data or schedule mismatch")
        rows.append(
            dict(
                source=r["source"],
                initialization_seed=r["initialization_seed"],
                sampling_seed=r["sampling_seed"],
                historical_full_success=old["mean_success"],
                random_success=r["random_success"],
                random_minus_full_pp=100 * (r["random_success"] - old["mean_success"]),
            )
        )
        provenance[path] = sha
    effects, repeats = {}, {}
    for source in ("api", "generic"):
        group = [r for r in rows if r["source"] == source]
        effects[source] = dict(
            full_success_pct=100
            * sum(r["historical_full_success"] for r in group)
            / len(group),
            random_success_pct=100
            * sum(r["random_success"] for r in group)
            / len(group),
            random_minus_full_pp=sum(r["random_minus_full_pp"] for r in group)
            / len(group),
        )
        values = []
        for repeat in (0, 1):
            old, path, sha = historical(source, 101, 101, repeat)
            values.append(
                dict(study="stage2", repeat=repeat, success=old["mean_success"])
            )
            provenance[path] = sha
        fresh = next(
            v
            for v in snapshot["records"]
            if v["case"]["arm"] == "full_reference" and v["case"]["source"] == source
        )
        values.append(dict(study="stage3", repeat=2, success=fresh["mean_success"]))
        repeats[source] = values
    record = dict(
        interim=True,
        complete=False,
        source_snapshot=str(args.snapshot),
        source_summary_sha256=file_sha(summary_path),
        historical_result_sha256=provenance,
        random_minus_historical_full=effects,
        paired_cells=rows,
        conditional_full_pool_repeats=repeats,
        note="Historical full references retained as frozen; three conditional repeats do not isolate systematic drift. Contemporary low-NLL/random comparison is separate.",
    )
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(dict(effects=effects, conditional_full_pool_repeats=repeats)),
        flush=True,
    )


if __name__ == "__main__":
    main()
