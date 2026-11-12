"""Supplement: enumerate distinct duplicate rows, not score-extreme indices."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

from learning_harness.common import digest, file_sha, read_jsonl

result = {}
inputs = {}
for source in ("api", "generic"):
    data = Path(f"runs/transfer-factorial-v1/corpora/full-{source}.jsonl")
    cache = Path(f"runs/variance-diagnostic-v1/stage3/base-nll/{source}.json")
    rows = read_jsonl(data)
    record = json.loads(cache.read_text())
    if record["data_sha256"] != file_sha(data):
        raise ValueError("Changed corpus identity")
    scores = record["scores"]
    if len(rows) != len(scores) or any(
        s["index"] != i or s["task_id"] != rows[i]["task_id"]
        for i, s in enumerate(scores)
    ):
        raise ValueError("Changed score alignment")
    inputs.update({str(data): file_sha(data), str(cache): file_sha(cache)})
    by = defaultdict(list)
    for index, row in enumerate(rows):
        by[digest(row["messages"])].append(index)
    groups = defaultdict(list)
    for indices in by.values():
        for a, b in combinations(indices, 2):
            ba = scores[(a // 8) * 8 : (a // 8 + 1) * 8]
            bb = scores[(b // 8) * 8 : (b // 8 + 1) * 8]
            padding = max(v["input_tokens"] for v in ba) == max(
                v["input_tokens"] for v in bb
            )
            head = max(v["target_tokens"] for v in ba) == max(
                v["target_tokens"] for v in bb
            )
            gap = abs(scores[a]["mean_target_nll"] - scores[b]["mean_target_nll"])
            groups["all"].append(gap)
            groups["same_padding" if padding else "different_padding"].append(gap)
            if padding and head and len(ba) == len(bb):
                groups["same_padding_head_and_batch_size"].append(gap)
    result[source] = {
        k: dict(pairs=len(v), max_gap=max(v, default=0), nonzero=sum(g > 0 for g in v))
        for k, v in groups.items()
    }
out = Path(__file__).with_name("distinct-pair-check.json")
if out.exists():
    raise ValueError("Preserve the existing distinct-pair audit")
out.write_text(
    json.dumps(
        dict(
            checked_unix=time.time(),
            sources=result,
            inputs_sha256=inputs,
            script_sha256=file_sha(__file__),
            actual_new_model_calls=0,
            scope="All distinct row pairs for identical messages. Extreme-pair groups in the original summary may use the same row twice when the score span is zero; those group counts are not counts of independent same-layout pairs.",
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(result, indent=2))
