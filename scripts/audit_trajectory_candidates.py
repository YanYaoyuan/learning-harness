#!/usr/bin/env python3
"""Read-only feasibility audit for complete-trajectory NLL selection baselines."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import time

from learning_harness.common import digest, file_sha, read_jsonl
from learning_harness.support import SUPPORT_PREFIX


def index_unique(rows, key, label):
    result = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError("Ambiguous duplicate in " + label)
        result[value] = row
    return result


def audit(source):
    collection = Path("runs/api-harness-v1/collection") / source
    raw_path = Path(f"runs/api-harness-v1/corpora/raw-{source}.jsonl")
    full_path = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
    score_path = Path(f"runs/variance-diagnostic-v1/stage3/base-nll/{source}.json")
    inputs = [
        collection / name for name in ("episodes.jsonl", "steps.jsonl", "run.json")
    ]
    inputs += [raw_path, full_path, score_path, Path(str(raw_path) + ".meta.json")]
    episodes = read_jsonl(inputs[0])
    successes = [
        e
        for e in episodes
        if e["partition"] == "train" and e["status"] == "completed" and e["success"]
    ]

    def key(row):
        return (
            row["task_id"],
            row.get("source_arm", row.get("arm")),
            row["rollout_seed"],
        )

    winners = index_unique(successes, key, "successful task/arm/seed identities")
    steps = defaultdict(list)
    for row in read_jsonl(inputs[1]):
        steps[row["attempt_id"]].append(row)
    raw, full = read_jsonl(raw_path), read_jsonl(full_path)
    raw_index = index_unique(raw, lambda r: (*key(r), r["step"]), "raw corpus steps")
    full_index = index_unique(
        full, lambda r: (*key(r), r["step"]), "eligible corpus steps"
    )
    meta = json.loads(inputs[-1].read_text())
    run = json.loads((collection / "run.json").read_text())
    if (
        meta["source_run_fingerprint"] != run["fingerprint"]
        or meta["successful_episodes"] != len(successes)
        or meta["examples"] != len(raw)
    ):
        raise ValueError("Raw corpus provenance differs from collection")
    score = json.loads(score_path.read_text())
    if score["data_sha256"] != file_sha(full_path) or len(score["scores"]) != len(full):
        raise ValueError("Cached NLL does not cover this eligible corpus")
    nll = {}
    for index, (row, value) in enumerate(zip(full, score["scores"])):
        if value["index"] != index or value["task_id"] != row["task_id"]:
            raise ValueError("Cached NLL row alignment changed")
        nll[(*key(row), row["step"])] = value["mean_target_nll"]
    records, raw_seen = [], set()
    for identity, episode in winners.items():
        trace = sorted(steps[episode["attempt_id"]], key=lambda r: r["step"])
        if [r["step"] for r in trace] != list(range(episode["steps"])):
            raise ValueError("Original successful trace is not complete")
        expected_steps = []
        for row in trace:
            if key(row) != identity or row["partition"] != "train":
                raise ValueError("Source trace identity/split changed")
            if row["invalid_action"]:
                continue
            row_key = (*identity, row["step"])
            expected = [
                m
                for m in row["messages"]
                if not (m["role"] == "user" and m["content"].startswith(SUPPORT_PREFIX))
            ]
            expected += [dict(role="assistant", content=row["response"])]
            if row_key not in raw_index or raw_index[row_key]["messages"] != expected:
                raise ValueError(
                    "Raw target view does not match its actual source step"
                )
            raw_seen.add(row_key)
            expected_steps.append(row_key)
        eligible = [k for k in expected_steps if k in full_index]
        if any(full_index[k] != raw_index[k] for k in eligible):
            raise ValueError("Eligible corpus changed source examples")
        missing = [k for k in expected_steps if k not in full_index]
        lookup = {r["step"]: r for r in trace}
        if any(
            lookup[k[-1]]["observation"].strip() != "Nothing happens." for k in missing
        ):
            raise ValueError("An excluded label has an unexplained removal reason")
        complete_raw = len(expected_steps) == episode["steps"]
        records.append(
            dict(
                task_id=identity[0],
                source_arm=identity[1],
                rollout_seed=identity[2],
                attempt_id=episode["attempt_id"],
                original_steps=episode["steps"],
                raw_step_labels=len(expected_steps),
                eligible_step_labels=len(eligible),
                raw_trace_complete=complete_raw,
                eligible_trace_complete=complete_raw and not missing,
                excluded_literal_no_effect_labels=len(missing),
                target_view_trajectory_sha256=digest(
                    [raw_index[k]["messages"] for k in expected_steps]
                ),
                full_trajectory_macro_nll=statistics.mean(
                    nll[k] for k in expected_steps
                )
                if complete_raw and not missing
                else None,
            )
        )
    if raw_seen != set(raw_index) or not set(full_index).issubset(raw_seen):
        raise ValueError("A corpus row has no unique successful training provenance")
    by_task = defaultdict(list)
    for row in records:
        by_task[row["task_id"]].append(row)
    complete = {
        task: [r for r in rows if r["eligible_trace_complete"]]
        for task, rows in by_task.items()
    }
    summary = dict(
        source=source,
        successful_episodes=len(successes),
        task_count=len(by_task),
        raw_step_labels=len(raw),
        eligible_step_labels=len(full),
        raw_complete_successful_traces=sum(r["raw_trace_complete"] for r in records),
        eligible_complete_successful_traces=sum(
            r["eligible_trace_complete"] for r in records
        ),
        traces_with_removed_labels=sum(
            r["excluded_literal_no_effect_labels"] > 0 for r in records
        ),
        removed_literal_no_effect_labels=sum(
            r["excluded_literal_no_effect_labels"] for r in records
        ),
        raw_candidates_per_task_histogram=dict(
            Counter(len(rows) for rows in by_task.values())
        ),
        tasks_with_two_or_more_raw_candidates=sum(
            len(rows) >= 2 for rows in by_task.values()
        ),
        tasks_with_two_or_more_distinct_raw_candidates=sum(
            len({r["target_view_trajectory_sha256"] for r in rows}) >= 2
            for rows in by_task.values()
        ),
        tasks_with_two_or_more_complete_eligible_candidates=sum(
            len(rows) >= 2 for rows in complete.values()
        ),
        tasks_with_ten_raw_candidates=sum(len(rows) >= 10 for rows in by_task.values()),
        cached_nll_covers_all_raw_steps=len(raw) == len(full),
        full_trajectory_nll_available=sum(
            r["full_trajectory_macro_nll"] is not None for r in records
        ),
        all_source_target_views_verified=True,
    )
    return summary, records, {str(p): file_sha(p) for p in inputs}


def main():
    out = Path("output/trajectory-baseline-feasibility/20261113")
    if (out / "aggregate-summary.json").exists():
        raise ValueError("Preserve the existing feasibility audit")
    out.mkdir(parents=True, exist_ok=True)
    summaries, inputs = {}, {}
    for source in ("api", "generic"):
        summary, records, hashes = audit(source)
        # Task identities and episode-level scores remain local, never upload.
        local = Path("runs/trajectory-baseline-feasibility/20261113")
        local.mkdir(parents=True, exist_ok=True)
        (local / (source + ".json")).write_text(json.dumps(records, indent=2) + "\n")
        summaries[source] = summary
        inputs.update(hashes)
    result = dict(
        status="source_lineage_and_trajectory_candidate_audit_passed",
        checked_unix=time.time(),
        sources=summaries,
        inputs_sha256=inputs,
        script_sha256=file_sha(__file__),
        actual_new_model_calls=0,
        gpu_used=False,
        teacher_calls=0,
        official_test_evaluations=0,
        raw_records_uploaded=False,
        scope="Old API/generic pools, not the newer API/ReAct source comparison. Metadata/NLL-cache feasibility only; no new performance result or SmartAD reproduction.",
    )
    (out / "aggregate-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], sources=summaries), indent=2))


if __name__ == "__main__":
    main()
