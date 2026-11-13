#!/usr/bin/env python3
"""Report actual token exposure under fixed-presentation range controls."""

import json
from pathlib import Path

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl
from train_lora import encode_example


def main():
    from transformers import AutoTokenizer

    root = Path("runs/nll-range-controls-v1")
    out = Path("output/nll-range-controls-v1/token-budget-summary.json")
    if out.exists():
        raise ValueError("Keep the prior token budget audit")
    provenance = json.loads((root / "preparation.private.json").read_text())[
        "provenance"
    ]
    for path, expected in provenance.items():
        if file_sha(path) != expected:
            raise ValueError("Prepared range-control input changed: " + path)
    tokenizer = AutoTokenizer.from_pretrained(
        "data/models/Qwen3-4B-Instruct-2507",
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = {}
    result = []
    for path in sorted((root / "candidate-schedules").glob("*.json")):
        recipe = json.loads(path.read_text())
        source = recipe["source"]
        if source not in encoded:
            rows = read_jsonl(recipe["data"])
            values = [encode_example(tokenizer, row, 2048) for row in rows]
            if any(value is None for value in values):
                raise ValueError("Previously eligible corpus has an overlength example")
            encoded[source] = [
                (
                    len(value["input_ids"]),
                    sum(label != -100 for label in value["labels"]),
                )
                for value in values
            ]
        lengths = [encoded[source][i] for i in recipe["indices"]]
        result.append(
            dict(
                source=source,
                policy=recipe["policy"],
                recipe=recipe["sampling_seed"],
                presentations=len(lengths),
                input_tokens_including_target=sum(v[0] for v in lengths),
                supervised_target_tokens=sum(v[1] for v in lengths),
                schedule_sha256=file_sha(path),
            )
        )
    save(
        out,
        dict(
            status="cpu_token_exposure_audited",
            rows=result,
            model_calls=0,
            same_presentations=True,
            exact_token_budget_matching=False,
            scope="Report actual exposure; fixed presentation count does not imply equal input or target token count.",
            preparation_sha256=file_sha(root / "preparation.private.json"),
        ),
    )
    print(
        json.dumps(
            dict(
                schedules=len(result),
                model_calls=0,
                target_tokens_range=[
                    min(r["supervised_target_tokens"] for r in result),
                    max(r["supervised_target_tokens"] for r in result),
                ],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
