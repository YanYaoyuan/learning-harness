#!/usr/bin/env python3
"""Budgeted, resumable teacher generation; credential values never enter logs."""
import argparse
import json
import os
from pathlib import Path

from learning_harness.common import file_sha, read_jsonl, write_jsonl
from learning_harness.credentials import load_teacher_env
from learning_harness.teacher import build_teacher


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
    root=Path('runs/api-harness-v1')
    load_teacher_env()
    design=json.loads((root/'design.json').read_text())
    if os.environ['LH_TEACHER_MODEL']!=design['teacher_model']:
        raise ValueError('Configured teacher differs from frozen design')
    for path,sha in json.loads((root/'inputs.sha256.json').read_text()).items():
        if file_sha(path)!=sha:raise ValueError('Frozen study input changed')
    tasks=read_jsonl(root/'train.jsonl')
    assert all(r['partition']=='train' and r['task_type']!='pick_cool_then_place_in_recep' for r in tasks)
    manifest=root/'train.jsonl'
    if args.limit:
        assert 0<args.limit<=len(tasks)
        manifest=root/f'teacher-first-{args.limit}.jsonl';write_jsonl(manifest,tasks[:args.limit])
    result=build_teacher(manifest,'data/alfworld',root/'teacher-settings.json',
        root/'api-support.jsonl',root/'teacher-budget.json')
    calls=json.loads((root/'teacher-budget.json').read_text())['calls']
    result.update(completed_calls=sum(r['status']=='completed' for r in calls),
        accounted_cny=sum(r.get('actual_cny',r['reserved_cny']) for r in calls),
        cost_basis='Undiscounted provider list-price estimate; includes failed reservations',
        all_cards_complete=result['cards']==len(tasks))
    (root/'teacher-generation.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['all_cards_complete']:
        (root/'teacher-bank.sha256.json').write_text(json.dumps({str(root/'api-support.jsonl'):file_sha(root/'api-support.jsonl')},indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
