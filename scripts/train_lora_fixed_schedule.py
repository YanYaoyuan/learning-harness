#!/usr/bin/env python3
"""Bind a hash-verified external schedule to the unchanged split-seed trainer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import train_lora_split_seeds as split


def bind_schedule(recipe,seed):
    if seed!=recipe['sampling_seed']:
        raise ValueError('Recipe sampling seed mismatch')
    indices=recipe['indices']
    def schedule(length,budget,ignored):
        if length!=recipe['eligible_examples'] or not 0<budget<=len(indices):
            raise ValueError('Eligibility or prefix budget changed')
        return indices[:budget]
    return schedule


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--schedule',required=True)
    args,forwarded=parser.parse_known_args()
    recipe_path=Path(args.schedule)
    recipe=json.loads(recipe_path.read_text())
    sha=hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    data=Path(forwarded[forwarded.index('--data')+1])
    if hashlib.sha256(data.read_bytes()).hexdigest()!=recipe['data_sha256']:
        raise ValueError('Recipe corpus hash mismatch')
    recipe=dict(recipe,eligible_examples=sum(bool(line.strip()) for line in data.read_text().splitlines()))
    old=split.schedule_for_sampling_seed
    old_argv=sys.argv
    split.schedule_for_sampling_seed=lambda original,seed:bind_schedule(recipe,seed)
    try:
        sys.argv=[split.__file__,*forwarded]
        split.main()
        output=Path(forwarded[forwarded.index('--out')+1])
        ctrl=json.loads((output/'seed-controls.json').read_text())
        meta=json.loads((output/'experiment.json').read_text())
        ctrl.update(external_schedule_sha256=sha,selection_policy=recipe['policy'],
                    scheduled_prefix_length=meta['processed_examples_rank0'],
                    external_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (output/'seed-controls.json').write_text(json.dumps(ctrl,indent=2)+'\n')
    finally:
        split.schedule_for_sampling_seed=old
        sys.argv=old_argv


if __name__=='__main__':
    main()
