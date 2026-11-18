#!/usr/bin/env python3
"""Two sampling policies over identical corpora, preserving the byte-frozen trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import train_lora_split_seeds as split


def schedule(original, length, budget, seed, arm):
    if arm == "full_pass":
        return original(length, budget, seed)
    if arm != "subset_replay":
        raise ValueError("Unknown sampling policy")
    if length < 256:
        raise ValueError("The replay control requires 256 eligible examples")
    pool = original(length, 256, seed)
    result = pool[:budget]
    # Preserve the original 256-example prefix, then reshuffle the same membership.
    rng = random.Random(seed + 1000003)
    while len(result) < budget:
        order = list(pool)
        rng.shuffle(order)
        result.extend(order[:budget - len(result)])
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampling-arm", required=True, choices=("full_pass", "subset_replay"))
    args, forwarded = parser.parse_known_args()
    original_binding = split.schedule_for_sampling_seed
    def binding(original, sampling_seed):
        return lambda length, budget, ignored: schedule(original, length, budget, sampling_seed, args.sampling_arm)
    split.schedule_for_sampling_seed = binding
    old_argv = sys.argv
    try:
        sys.argv = [split.__file__, *forwarded]
        split.main()
        output = Path(forwarded[forwarded.index("--out") + 1]) / "seed-controls.json"
        control = json.loads(output.read_text())
        control.update(sampling_arm=args.sampling_arm,
                       intervention_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        output.write_text(json.dumps(control, indent=2) + "\n")
    finally:
        split.schedule_for_sampling_seed = original_binding
        sys.argv = old_argv


if __name__ == "__main__":
    main()
