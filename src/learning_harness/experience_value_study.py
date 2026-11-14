"""Development study: deterministic selectors, paired cells and raw-outcome audits."""

from collections import Counter, defaultdict
import hashlib
import json
import math

from .action_matched_sampling import action_matched_schedule
from .nll_range_sampling import range_schedule

POLICIES = (
    "random",
    "gradient_high",
    "high",
    "low",
    "middle",
    "gradient_low",
    "gradient_global",
)
SOURCES = ("api", "generic")
INITIALIZATIONS = (101, 202, 303, 404, 505, 606, 707, 808)
SAMPLING = (101, 202)
NODES = (1, 2, 4, 8, 16)


def clean_zero_branch(tensors):
    """Retain the active adapter without foreign named-adapter export keys."""
    active = {
        k: v
        for k, v in tensors.items()
        if k.endswith((".lora_A.weight", ".lora_B.weight"))
    }
    foreign = set(tensors) - set(active)
    if not active or any(not k.endswith(".lora_B.A.weight") for k in foreign):
        raise ValueError("Unrecognized mixed-adapter export")
    a = {
        k.removesuffix(".lora_A.weight") for k in active if k.endswith(".lora_A.weight")
    }
    b = {
        k.removesuffix(".lora_B.weight") for k in active if k.endswith(".lora_B.weight")
    }
    if (
        not a
        or a != b
        or any(
            v.count_nonzero().item()
            for k, v in active.items()
            if k.endswith(".lora_B.weight")
        )
    ):
        raise ValueError("Active zero-branch projections are incomplete or nonzero")
    return active, sorted(foreign)


def content_sha(row):
    return hashlib.sha256(
        json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def matrix():
    cases = []
    block = 0
    for initialization in INITIALIZATIONS:
        for sampling in SAMPLING:
            block += 1
            for policy in POLICIES:
                for source in SOURCES:
                    cases.append(
                        dict(
                            name=f"b{block:02d}-{policy}-{source}-i{initialization}-s{sampling}",
                            block=block,
                            source=source,
                            policy=policy,
                            initialization_seed=initialization,
                            sampling_seed=sampling,
                            gpu=0,
                        )
                    )
    # Train one student immediately; measure the untrained baseline before any
    # balanced result is reported. Neither result changes the registered plan.
    cases.insert(
        1,
        dict(
            name="base",
            block=0,
            source="base",
            policy="base",
            initialization_seed=None,
            sampling_seed=None,
            gpu=0,
        ),
    )
    return cases


def arrays(rows):
    if not rows or any(r["partition"] != "train" for r in rows):
        raise ValueError("Only existing train supervision may enter selection")
    tasks = [r["task_id"] for r in rows]
    commands = [json.loads(r["messages"][-1]["content"])["action"] for r in rows]
    families = ["go" if a.startswith("go to ") else a.split()[0] for a in commands]
    return tasks, families, [content_sha(r) for r in rows]


def select(rows, nll, gradient, reference, sampling_seed, policy):
    tasks, families, content = arrays(rows)
    if (
        policy not in POLICIES
        or len(nll) != len(rows)
        or len(gradient) != len(rows)
        or not all(math.isfinite(v) for v in gradient)
    ):
        raise ValueError("Unknown selector or misaligned scores")
    if policy in ("random", "low", "middle", "high"):
        result = range_schedule(
            tasks, families, content, reference, sampling_seed, policy, nll
        )
    elif policy in ("gradient_high", "gradient_low"):
        values = [-v for v in gradient] if policy == "gradient_high" else gradient
        result = action_matched_schedule(
            tasks, families, content, reference, sampling_seed, "min_nll", values
        )
    else:
        from random import Random

        first = {}
        for i, key in enumerate(content):
            first.setdefault(key, i)
        chosen = sorted(first.values(), key=lambda i: (-gradient[i], i))[:256]
        if len(chosen) != 256:
            raise ValueError("Insufficient distinct content for global selection")
        rng, result = Random(sampling_seed), []
        while len(result) < 768:
            order = list(range(256))
            rng.shuffle(order)
            result.extend(chosen[i] for i in order[: 768 - len(result)])
    if len(result) != 768 or len({(tasks[i], content[i]) for i in result}) != 256:
        raise ValueError("Presentation or distinct-content budget changed")
    if policy != "gradient_global":
        strata = list(zip(tasks, families))

        def quota(indices):
            return Counter(s for s, key in {(strata[i], content[i]) for i in indices})

        if [strata[i] for i in result] != [strata[i] for i in reference] or quota(
            result
        ) != quota(reference):
            raise ValueError("Task/action position or content quota changed")
    return result


def validate_queries(queries, calibration, confirmation):
    if len(queries) != 128 or Counter(r["task_type"] for r in queries) != Counter(
        pick_and_place_simple=64, pick_clean_then_place_in_recep=64
    ):
        raise ValueError("All 128 balanced development tasks are required")
    groups = [r["task_group"] for r in queries]
    forbidden = {r["task_group"] for r in calibration + confirmation}
    if (
        len(set(groups)) != 128
        or set(groups) & forbidden
        or len({r["task_id"] for r in queries}) != 128
        or any(
            r["fresh_role"] != "development"
            or r["partition"] != "fresh_development"
            or r["official_split"] != "train"
            for r in queries
        )
    ):
        raise ValueError("Development role/group isolation failed")


def audit_episodes(rows, events, usage, queries):
    expected = {r["task_id"]: r for r in queries}
    if (
        len(rows) != len(expected)
        or {r["task_id"] for r in rows} != set(expected)
        or len({r["run_id"] for r in rows}) != len(rows)
        or not usage["complete_usage"]
        or usage["error_records"]
    ):
        raise ValueError("Incomplete/duplicate evaluation or request accounting")
    by_run = defaultdict(list)
    for event in events:
        by_run[event["run_id"]].append(event)
    if set(by_run) != {r["run_id"] for r in rows}:
        raise ValueError("Orphan events")
    from .json_demo_executor import actor_messages

    for row in rows:
        own = by_run[row["run_id"]]
        task = expected[row["task_id"]]
        if (
            row["task_group"] != task["task_group"]
            or row["task_type"] != task["task_type"]
            or row["seed"] != 11
            or row["method"] != "json_direct"
            or len(row["trials"]) != 1
            or not 1 <= row["calls"] <= 40
            or len(own) != row["calls"]
            or [e["call"] for e in own] != list(range(1, row["calls"] + 1))
            or any(
                e["task_id"] != row["task_id"]
                or e["seed"] != 11
                or e["method"] != "json_direct"
                or e["trial"] != 1
                or e["phase"] != "act"
                or e["thought"]
                for e in own
            )
            or row["success"] != own[-1]["success"]
            or row["first_success"] != row["success"]
            or any(
                sum(e[k] for e in own) != row[k]
                for k in ("input_tokens", "output_tokens")
            )
        ):
            raise ValueError("Raw episode and event accounting differs")
        system = actor_messages(own[0]["messages"][1]["content"], [])[0]
        if any(e["messages"][0] != system for e in own):
            raise ValueError("The common public-example executor changed")
    if (
        sum(r["calls"] for r in rows) != usage["attempted_requests"]
        or len(events) != usage["attempted_requests"]
        or sum(r["input_tokens"] for r in rows) != usage["known_input_tokens"]
        or sum(r["output_tokens"] for r in rows) != usage["known_output_tokens"]
    ):
        raise ValueError("Global request/event totals differ")
    return dict(
        episodes=len(rows),
        successes=sum(r["success"] for r in rows),
        mean_success=sum(r["success"] for r in rows) / len(rows),
        families={
            f: dict(
                tasks=sum(r["task_type"] == f for r in rows),
                successes=sum(r["success"] for r in rows if r["task_type"] == f),
            )
            for f in sorted({r["task_type"] for r in queries})
        },
        invalid_format_decisions=sum(bool(e["invalid_format"]) for e in events),
        usage=usage,
    )


def aggregate_cells(records, block_ids):
    selected = [r for r in records if r["case"]["block"] in block_ids]
    keys = [
        (r["case"]["block"], r["case"]["source"], r["case"]["policy"]) for r in selected
    ]
    wanted = {(b, s, p) for b in block_ids for s in SOURCES for p in POLICIES}
    if len(keys) != len(set(keys)) or set(keys) != wanted:
        raise ValueError(
            "Only complete balanced source/selector blocks may be aggregated"
        )
    result = {}
    for source in SOURCES:
        result[source] = {}
        for policy in POLICIES:
            items = [
                r
                for r in selected
                if r["case"]["source"] == source and r["case"]["policy"] == policy
            ]
            by_init = defaultdict(list)
            for r in items:
                by_init[r["case"]["initialization_seed"]].append(r["mean_success"])
            result[source][policy] = dict(
                mean_success=sum(r["mean_success"] for r in items) / len(items),
                per_initialization={
                    str(k): sum(v) / len(v) for k, v in sorted(by_init.items())
                },
                seed_cells=len(items),
            )
    contrasts = {}
    for alternative in ("random", "high", "gradient_low"):
        differences = []
        cells = {
            (r["case"]["block"], r["case"]["source"], r["case"]["policy"]): r[
                "mean_success"
            ]
            for r in selected
        }
        for block in block_ids:
            differences.append(
                sum(
                    cells[(block, s, "gradient_high")] - cells[(block, s, alternative)]
                    for s in SOURCES
                )
                / 2
            )
        contrasts["gradient_high_minus_" + alternative] = dict(
            mean_difference=sum(differences) / len(differences),
            per_paired_block=differences,
            wins=sum(v > 0 for v in differences),
            ties=sum(v == 0 for v in differences),
            losses=sum(v < 0 for v in differences),
        )
    return dict(
        sources=result,
        contrasts=contrasts,
        complete_blocks=len(block_ids),
        inferential_tests_performed=False,
        scope="Descriptive development results; no calibrated confirmatory p-values or equivalence claim.",
    )
