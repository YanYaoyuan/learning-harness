"""Plan and paired-result contracts shared by future action-control studies."""

from collections import defaultdict
import math

SOURCES = ("api", "generic")
POLICIES = ("random", "min_nll")


def validate_plan(frozen):
    if (
        frozen["protocol"] != "task-action-controls-v1"
        or frozen["teacher_calls"] != 0
        or frozen["official_tests_used"]
        or frozen["primary_metric"] != "mean_success"
        or frozen["primary_contrast"] != "min_nll_minus_random"
    ):
        raise ValueError("Unexpected study question, outcome or execution scope")
    gpus = frozen["gpus"]
    if (
        not gpus
        or len(set(gpus)) != len(gpus)
        or any(type(g) is not int or g < 0 for g in gpus)
    ):
        raise ValueError("Unique physical GPU indices required")
    if (
        not math.isfinite(frozen["cap_gpu_hours"])
        or frozen["cap_gpu_hours"] <= 0
        or not 0 < frozen["minimum_allocation_seconds"] <= frozen["allocation_seconds"]
        or not math.isfinite(frozen["allocation_seconds"])
    ):
        raise ValueError("Finite positive accounting limits required")
    prep = frozen.get("preparatory_gpu_seconds", 0)
    if not math.isfinite(prep) or prep < 0:
        raise ValueError("Preparatory GPU time cannot be negative or nonfinite")
    cases = frozen["cases"]
    if not cases or frozen["planned_episodes"] != 64 * len(cases):
        raise ValueError("Two 32-task families required for every planned model")
    names, indexed = set(), {}
    for c in cases:
        name = c["name"]
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or name in names
        ):
            raise ValueError("Unique plain case names required")
        names.add(name)
        if (
            c["source"] not in SOURCES
            or c["policy"] not in POLICIES
            or type(c["gpu"]) is not int
            or c["gpu"] not in gpus
            or not isinstance(c["schedule"], str)
            or type(c["repeat"]) is not int
            or c["repeat"] < 0
            or any(
                type(c[k]) is not int for k in ("initialization_seed", "sampling_seed")
            )
        ):
            raise ValueError("Invalid source/policy, seed, repeat or GPU")
        key = (
            c["source"],
            c["initialization_seed"],
            c["sampling_seed"],
            c["repeat"],
            c["policy"],
        )
        if key in indexed:
            raise ValueError("Duplicate experimental cell")
        indexed[key] = c
    source_keys = {}
    for source in SOURCES:
        source_keys[source] = {k[1:3] for k in indexed if k[0] == source and k[3] == 0}
    if not source_keys["api"] or source_keys["api"] != source_keys["generic"]:
        raise ValueError("Sources must share the same primary seed cells")
    for key, c in indexed.items():
        mate_key = (*key[:4], "min_nll" if key[4] == "random" else "random")
        mate = indexed.get(mate_key)
        if mate is None or mate["gpu"] != c["gpu"]:
            raise ValueError("Policy pairs must be complete and share physical GPU")
        if c["repeat"]:
            base = indexed.get((key[0], key[1], key[2], 0, key[4]))
            if (
                base is None
                or base["schedule"] != c["schedule"]
                or base["gpu"] != c["gpu"]
            ):
                raise ValueError(
                    "Conditional repeat must reuse schedule and physical GPU"
                )
    keys = sorted(source_keys["api"])
    return dict(
        primary_seed_cells=keys,
        planned_models=len(cases),
        repeated_models=sum(c["repeat"] > 0 for c in cases),
    )


def paired_effects(rows):
    indexed = {}
    for row in rows:
        if row["repeat"] != 0:
            continue
        key = (
            row["source"],
            row["initialization_seed"],
            row["sampling_seed"],
            row["policy"],
        )
        if key in indexed:
            raise ValueError("Duplicate primary result cell")
        indexed[key] = row
    keys = sorted({k[1:3] for k in indexed})
    if not keys or any(
        (s, *key, p) not in indexed for key in keys for s in SOURCES for p in POLICIES
    ):
        raise ValueError("Every primary result must have all source/policy mates")
    result = {}
    for metric in ("mean_success", "mean_nll"):
        effects = {}
        for source in SOURCES:
            per_init = defaultdict(list)
            cells = []
            for init, sample in keys:
                random_value = indexed[(source, init, sample, "random")][metric]
                selected_value = indexed[(source, init, sample, "min_nll")][metric]
                if not math.isfinite(random_value) or not math.isfinite(selected_value):
                    raise ValueError("Nonfinite result cannot enter a contrast")
                if (
                    metric == "mean_success"
                    and not all(0 <= v <= 1 for v in (random_value, selected_value))
                ) or (metric == "mean_nll" and min(random_value, selected_value) < 0):
                    raise ValueError(
                        "Success probability or NLL outside its valid range"
                    )
                delta = selected_value - random_value
                cells.append(
                    dict(
                        initialization_seed=init, sampling_seed=sample, difference=delta
                    )
                )
                per_init[init].append(delta)
            better = [
                r["difference"] if metric == "mean_success" else -r["difference"]
                for r in cells
            ]
            effects[source] = dict(
                mean_difference=sum(r["difference"] for r in cells) / len(cells),
                cells=cells,
                wins=sum(v > 0 for v in better),
                ties=sum(v == 0 for v in better),
                losses=sum(v < 0 for v in better),
                per_initialization=[
                    dict(initialization_seed=i, mean_difference=sum(v) / len(v))
                    for i, v in sorted(per_init.items())
                ],
            )
        result[metric] = dict(nll_minus_random=effects)
    return result


def audit_episodes(episodes, manifest):
    expected = {row["task_id"]: row for row in manifest}
    if (
        len(expected) != len(manifest)
        or not expected
        or any(
            r["official_split"] != "train" or r["purpose"] != "development"
            for r in manifest
        )
    ):
        raise ValueError(
            "Expected frozen development tasks from the official training pool"
        )
    if len(episodes) != len(expected) or {r["task_id"] for r in episodes} != set(
        expected
    ):
        raise ValueError("Episode count/identity does not match the frozen manifest")
    for row in episodes:
        target = expected[row["task_id"]]
        if (
            row["arm"] != "none"
            or row["rollout_seed"] != 11
            or row["status"] != "completed"
            or row["partition"] != target["partition"]
            or row["task_group"] != target["task_group"]
            or type(row["success"]) is not bool
        ):
            raise ValueError(
                "Wrong execution condition, role, grouping or invalid episode"
            )
    return sum(r["success"] for r in episodes) / len(episodes)


def validate_reservation(ledger, name, pid, gpu, seconds, freeze_sha):
    item = ledger.get("open", {}).get(name)
    if (
        ledger.get("status") != "running"
        or ledger.get("freeze_sha256") != freeze_sha
        or item is None
        or item["pid"] != pid
        or item["gpu"] != gpu
        or item["allocation"] != seconds
    ):
        raise ValueError("Worker must own a matching durable GPU/time reservation")
