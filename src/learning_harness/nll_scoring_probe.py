"""Train-only scoring geometry, fixed stress selection and aggregate diagnostics."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .action_matched_sampling import action_matched_schedule
from .common import digest, file_sha, read_jsonl
from .task_matched_sampling import matched_schedule

CONDITIONS = (
    ("canonical", 0, True, "fixed", 0),
    ("canonical_repeat", 0, True, "fixed", 1),
    ("default_unpadded", 0, False, "fixed", 0),
    ("dynamic_window", 0, True, "dynamic", 0),
    ("pad32_default", 32, False, "fixed", 0),
    ("pad32_explicit", 32, True, "fixed", 0),
    ("pad128_default", 128, False, "fixed", 0),
    ("pad128_explicit", 128, True, "fixed", 0),
)


def validate_encoded(row):
    ids, labels, mask = (row[k] for k in ("input_ids", "labels", "attention_mask"))
    if (
        not ids
        or len(ids) > 2048
        or len(labels) != len(ids)
        or mask != [1] * len(ids)
        or any(type(x) is not int or x < 0 for x in ids)
    ):
        raise ValueError("Invalid unpadded training-token record")
    start = next((i for i, x in enumerate(labels) if x != -100), len(ids))
    if not 0 < start < len(ids) or labels != [-100] * start + ids[start:]:
        raise ValueError("Assistant labels must be a nonempty contiguous suffix")
    return len(ids) - start


def make_batch(rows, pad_token, extra_padding=0, explicit_positions=False):
    if (
        not rows
        or type(pad_token) is not int
        or pad_token < 0
        or type(extra_padding) is not int
        or extra_padding < 0
    ):
        raise ValueError("Explicit valid batch and padding required")
    for row in rows:
        validate_encoded(row)
    length = max(len(row["input_ids"]) for row in rows) + extra_padding
    if length > 2048:
        raise ValueError("Padding would exceed the frozen context limit")
    batch = dict(input_ids=[], attention_mask=[], labels=[])
    if explicit_positions:
        batch["position_ids"] = []
    for row in rows:
        padding = length - len(row["input_ids"])
        batch["input_ids"].append([pad_token] * padding + row["input_ids"])
        batch["attention_mask"].append([0] * padding + row["attention_mask"])
        batch["labels"].append([-100] * padding + row["labels"])
        if explicit_positions:
            batch["position_ids"].append(
                [0] * padding + list(range(len(row["input_ids"])))
            )
    return batch


def prediction_positions(labels, logits_length):
    length = len(labels)
    if type(logits_length) is not int or not 0 < logits_length <= length:
        raise ValueError("Actual logits length is invalid")
    indices, targets = [], []
    for index, token in enumerate(labels):
        if token == -100:
            continue
        position = index - 1 - (length - logits_length)
        if position < 0:
            raise ValueError("Logits window omitted a supervised prediction")
        indices.append(position)
        targets.append(token)
    if not targets:
        raise ValueError("No supervised target tokens")
    return indices, targets


def masked_nll(torch, logits, labels):
    if logits.ndim != 3 or logits.shape[0] != len(labels):
        raise ValueError("Expected batch/sequence/vocabulary logits")
    results = []
    for index, target in enumerate(labels):
        positions, tokens = prediction_positions(target, int(logits.shape[1]))
        loss = torch.nn.functional.cross_entropy(
            logits[index, positions].float(),
            torch.tensor(tokens, device=logits.device, dtype=torch.long),
            reduction="mean",
        )
        value = float(loss)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid target NLL")
        results.append(value)
    return results


def select_stress(entries):
    if len(entries) < 32 or len({r["content_sha256"] for r in entries}) != len(entries):
        raise ValueError("At least32 distinct message sequences required")

    def span(row):
        values = [o["cached_nll"] for o in row["observations"]]
        if not values or any(not math.isfinite(x) for x in values):
            raise ValueError("Missing/invalid cached observations")
        return max(values) - min(values)

    pressure = sorted(entries, key=lambda r: (-span(r), r["content_sha256"]))[:16]
    used = {r["content_sha256"] for r in pressure}
    fixed = sorted(
        (r for r in entries if r["content_sha256"] not in used),
        key=lambda r: r["content_sha256"],
    )[:16]
    return [r["content_sha256"] for r in pressure + fixed]


def trials_for(entry):
    trials = []
    for name, padding, explicit, window, repeat in CONDITIONS:
        trials.append(
            dict(
                name=name,
                padding=padding,
                explicit_positions=explicit,
                window=window,
                repeat=repeat,
                feasible=len(entry["encoded"]["input_ids"]) + padding <= 2048,
            )
        )
    return trials


def rank_correlation(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            for i in order[start:end]:
                result[i] = (start + end - 1) / 2
            start = end
        mean = statistics.mean(result)
        return [v - mean for v in result]

    a, b = ranks(left), ranks(right)
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else None


def prepare_inputs(tokenizer, encode_example):
    """CPU only; exact encoded lengths checked against every cached row."""
    entries, sources, inputs = {}, {}, {}
    for source in ("api", "generic"):
        data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        path = Path(f"runs/variance-diagnostic-v1/stage3/base-nll/{source}.json")
        rows = read_jsonl(data)
        cache = json.loads(path.read_text())
        if cache["data_sha256"] != file_sha(data) or len(cache["scores"]) != len(rows):
            raise ValueError("Cache/corpus identity mismatch")
        mapping = []
        for index, (row, value) in enumerate(zip(rows, cache["scores"])):
            if value["index"] != index or value["task_id"] != row["task_id"]:
                raise ValueError("Cache row alignment changed")
            encoded = encode_example(tokenizer, row, 2048)
            if encoded is None:
                raise ValueError("Eligibility changed")
            targets = validate_encoded(encoded)
            if (
                len(encoded["input_ids"]) != value["input_tokens"]
                or targets != value["target_tokens"]
            ):
                raise ValueError("Tokenization differs from the original cache")
            key = digest(row["messages"])
            if key not in entries:
                entries[key] = dict(
                    content_sha256=key, encoded=encoded, observations=[]
                )
            elif entries[key]["encoded"] != encoded:
                raise ValueError("Identical messages tokenized inconsistently")
            entries[key]["observations"].append(
                dict(source=source, index=index, cached_nll=value["mean_target_nll"])
            )
            mapping.append(key)
        sources[source] = mapping
        inputs.update({str(data): file_sha(data), str(path): file_sha(path)})
    ordered = [entries[key] for key in sorted(entries)]
    stress = select_stress(ordered)
    replay = set()
    for key in stress:
        observations = entries[key]["observations"]
        low = min(
            observations, key=lambda r: (r["cached_nll"], r["source"], r["index"])
        )
        high = max(
            observations, key=lambda r: (r["cached_nll"], r["source"], r["index"])
        )
        # In zero-span groups use distinct endpoints if available.
        if high == low and len(observations) > 1:
            high = observations[-1] if observations[-1] != low else observations[0]
        for row in (low, high):
            replay.add((row["source"], row["index"] // 8 * 8))
    return dict(
        entries=ordered,
        source_row_keys=sources,
        stress_keys=stress,
        replay_batches=[dict(source=s, start=i) for s, i in sorted(replay)],
        fixed_logits_window=1 + max(validate_encoded(r["encoded"]) for r in ordered),
        input_files=inputs,
        pad_token_id=tokenizer.pad_token_id,
        schema="nll-scoring-probe-inputs-v1",
    )


def summarize(stress, fp32, direct_checks):
    by = defaultdict(dict)
    for row in stress:
        if row["condition"] in by[row["content_sha256"]]:
            raise ValueError("Duplicate stress observation")
        if not math.isfinite(row["nll"]) or row["nll"] < 0:
            raise ValueError("Invalid stress score")
        by[row["content_sha256"]][row["condition"]] = row["nll"]
    if len({r["content_sha256"] for r in fp32}) != len(fp32):
        raise ValueError("Duplicate FP32 reference")
    conditions = {}
    for name, *_ in CONDITIONS:
        paired = [
            (values[name], values["canonical"])
            for values in by.values()
            if name in values and "canonical" in values
        ]
        errors = [
            abs(values[name] - values["canonical"])
            for values in by.values()
            if name in values and "canonical" in values
        ]
        conditions[name] = dict(
            paired_inputs=len(errors),
            mean_absolute_difference=statistics.mean(errors) if errors else None,
            max_absolute_difference=max(errors, default=None),
            rank_correlation_to_canonical=rank_correlation(
                [a for a, _ in paired], [b for _, b in paired]
            ),
        )
    precision = [
        abs(row["nll"] - by[row["content_sha256"]]["canonical"])
        for row in fp32
        if "canonical" in by[row["content_sha256"]]
    ]
    repeat = conditions["canonical_repeat"]
    repeat_ok = (
        repeat["paired_inputs"] == 32 and repeat["max_absolute_difference"] <= 1e-6
    )
    direct_ok = (
        len(direct_checks) == 4
        and len({r["content_sha256"] for r in direct_checks}) == 4
        and all(
            math.isfinite(r["absolute_difference"])
            and 0 <= r["absolute_difference"] <= 1e-4
            for r in direct_checks
        )
    )
    return dict(
        conditions=conditions,
        fp32_reference_inputs=len(fp32),
        bf16_vs_fp32_mean_absolute_difference=statistics.mean(precision)
        if precision
        else None,
        bf16_vs_fp32_max_absolute_difference=max(precision, default=None),
        bf16_vs_fp32_rank_correlation=rank_correlation(
            [r["nll"] for r in fp32 if "canonical" in by[r["content_sha256"]]],
            [
                by[r["content_sha256"]]["canonical"]
                for r in fp32
                if "canonical" in by[r["content_sha256"]]
            ],
        ),
        repeated_input_gate_passed=repeat_ok,
        direct_loss_gate_passed=direct_ok,
        direct_loss_checks=len(direct_checks),
        direct_loss_max_absolute_difference=max(
            (r["absolute_difference"] for r in direct_checks), default=None
        ),
        optional_rescoring_gate_passed=repeat_ok
        and direct_ok
        and len(fp32) == 32
        and {r["content_sha256"] for r in fp32} == set(by),
    )


def selection_changes(inputs, canonical, fixed_schedule):
    entries = {r["content_sha256"]: r for r in inputs["entries"]}
    if set(canonical) != set(entries):
        raise ValueError("Partial canonical scores cannot produce a selection report")
    result = []
    for source, keys in inputs["source_row_keys"].items():
        rows = read_jsonl(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
        tasks = [r["task_id"] for r in rows]
        values = [canonical[key] for key in keys]
        cached = {}
        for entry in entries.values():
            for observation in entry["observations"]:
                if observation["source"] == source:
                    cached[observation["index"]] = observation["cached_nll"]
        original_scores = [cached[i] for i in range(len(rows))]
        families = []
        for row in rows:
            action = json.loads(row["messages"][-1]["content"])["action"]
            families.append("go" if action.startswith("go to ") else action.split()[0])
        for stage in ("stage3", "stage4"):
            for seed in (101, 202, 303, 404):
                path = Path(
                    f"runs/variance-diagnostic-v1/{stage}/inputs/schedules/min_nll-{source}-s{seed}.json"
                )
                recipe = json.loads(path.read_text())
                if stage == "stage3":
                    reference = fixed_schedule(len(rows), 768, seed)

                    def regenerate(vector):
                        return matched_schedule(
                            tasks, reference, seed, "min_nll", vector, content_keys=keys
                        )
                else:
                    reference = json.loads(
                        Path(recipe["reference_recipe"]).read_text()
                    )["indices"]

                    def regenerate(vector):
                        return action_matched_schedule(
                            tasks, families, keys, reference, seed, "min_nll", vector
                        )

                if regenerate(original_scores) != recipe["indices"]:
                    raise ValueError("Original frozen schedule did not regenerate")
                selected = regenerate(values)
                old, new = (
                    {keys[i] for i in recipe["indices"]},
                    {keys[i] for i in selected},
                )
                if len(old) != len(new) or len(old) != 256:
                    raise ValueError("Canonical scoring changed support budget")
                result.append(
                    dict(
                        source=source,
                        stage=stage,
                        sampling_seed=seed,
                        selected_content=256,
                        replaced_content=len(old - new),
                        retained_fraction=len(old & new) / 256,
                        original_recipe_sha256=file_sha(path),
                    )
                )
    return result
