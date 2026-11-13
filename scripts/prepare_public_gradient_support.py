#!/usr/bin/env python3
"""Prepare public target-support candidates; never read query answers or train."""

from collections import Counter
import json
import os
from pathlib import Path

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, write_jsonl
from learning_harness.public_demo_support import support_from_demo
from learning_harness.fresh_tasks import public_demo_signatures
from train_lora import encode_example


def main():
    root = Path("runs/public-gradient-support-v1")
    if root.exists():
        raise ValueError("Preserve previous public support preparation")
    from transformers import AutoTokenizer

    model = "data/models/Qwen3-4B-Instruct-2507"
    source = Path("third_party/ReAct/prompts/alfworld_3prompts.json")
    examples = json.loads(source.read_text())
    signatures = set(public_demo_signatures())
    fresh = json.loads(
        Path(
            "runs/fresh-experience-value-v1/public-reset-signatures.private.json"
        ).read_text()
    )
    if signatures.intersection(fresh.values()):
        raise ValueError("Public target support overlaps a fresh task signature")
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    rows = []
    rejected = Counter()
    for key in ("react_put_0", "react_put_1", "react_clean_0", "react_clean_1"):
        candidates = support_from_demo(examples[key], key)
        for row in candidates:
            if encode_example(tokenizer, row, 2048) is None:
                rejected[key] += 1
                continue
            rows.append(row)
    root.mkdir(exist_ok=False)
    path = root / "support.private.jsonl"
    write_jsonl(path, rows)
    os.chmod(path, 0o600)
    summary = dict(
        status="public_target_support_candidates_not_trained",
        examples=len(rows),
        per_public_demo=dict(Counter(r["demo_key"] for r in rows)),
        length_rejected=dict(rejected),
        max_sequence_length=2048,
        assistant_only_target=True,
        thoughts_removed=True,
        query_answers_used=False,
        new_model_calls=0,
        teacher_calls=0,
        support_sha256=file_sha(path),
        public_source_sha256=file_sha(source),
        parser_sha256=file_sha("src/learning_harness/public_demo_support.py"),
        scope="Public support shared with ReAct prompting, not fresh supervised tasks; freeze its role, weights and cost in the later selection protocol.",
    )
    save("output/public-gradient-support-v1/preparation-summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
