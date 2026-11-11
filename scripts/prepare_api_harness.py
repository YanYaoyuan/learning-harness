#!/usr/bin/env python3
"""Freeze a stronger-scaffold replication before paid teacher generation."""
from collections import defaultdict
import json
from pathlib import Path

from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl
from learning_harness.teacher import DEFAULT_PROMPT

ROOT = Path('runs/api-harness-v1')
B_FAMILY = 'pick_cool_then_place_in_recep'
A_FAMILIES = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep']


def main():
    if (ROOT/'design.json').exists():
        raise ValueError('Design already frozen; refusing to overwrite')
    ROOT.mkdir(parents=True, exist_ok=True)
    original = read_jsonl('data/manifests/alfworld/train_pilot.jsonl')
    train = [r for r in original if r['task_type'] in A_FAMILIES]
    assert len(train) == 72
    blocked = {r['task_group'] for r in original}
    for name in ['b_support_candidates', 'b_calibration', 'b_query', 'a_retention']:
        blocked.update(r['task_group'] for r in read_jsonl(f'runs/learning-speed-v1/{name}.jsonl'))
    for name in ['fresh-probe.jsonl', 'probe.jsonl']:
        p = Path('runs/dual5090-v2')/name
        if p.exists(): blocked.update(r['task_group'] for r in read_jsonl(p))
    pools = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl('data/manifests/alfworld/train.jsonl'):
        if row['task_group'] not in blocked:
            pools[row['task_type']][row['task_group']].append(row)
    def choose(family, n, role):
        groups = sorted(pools[family], key=lambda g: digest([20261109, 'api-harness-v1', role, g]))[:n]
        assert len(groups) == n
        return [dict(min(pools[family][g], key=lambda r: digest(['representative', r['task_id']])),
            partition='probe', original_partition='train', purpose='development', adaptation_role=role) for g in groups]
    a_query = [r for family in A_FAMILIES for r in choose(family, 8, 'a_query')]
    b_query = choose(B_FAMILY, 24, 'b_query')
    assert len({r['task_id'] for r in train}) == 72
    train_groups={r['task_group'] for r in train}
    query_groups={r['task_group'] for r in a_query+b_query}
    assert len(query_groups)==48 and not train_groups & query_groups
    for name, rows in [('train', train), ('a_query', a_query), ('b_query', b_query)]:
        write_jsonl(ROOT/f'{name}.jsonl', rows)
    # The same public-reset hashes as the earlier manual strategy bank.
    old_bank = {r['task_id']: r for r in read_jsonl('runs/dual5090-v2/support.jsonl')}
    write_jsonl(ROOT/'generic-support.jsonl', [old_bank[r['task_id']] for r in train])
    settings = json.loads((ROOT/'teacher-settings.json').read_text())
    settings['system_prompt'] = DEFAULT_PROMPT + (
        ' The student uses ALFWorld commands: go to RECEPTACLE to inspect it; open/close RECEPTACLE; '
        'take OBJECT from RECEPTACLE; move OBJECT to RECEPTACLE; clean OBJECT with sinkbasin; '
        'heat OBJECT with microwave. Use exact observed object/receptacle names with numbers. '
        'Adapt the strategy to the stated goal, emphasize operation dependencies, exploration '
        'and verification, and recovery from Nothing happens. Do not assume hidden locations or '
        'invent completed operations. Do not include cooled-object tasks absent from the stated goal.')
    (ROOT/'teacher-settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    design = dict(protocol='api-harness-v1', status='frozen_before_teacher_generation',
        question='Does a stronger task-specific scaffold change data collection, unsupported ability, and fixed-data future adaptation?',
        conditions=['none', 'generic', 'api'],
        teacher_model='qwen3.8-max-0902', teacher_input='public reset observation only',
        teacher_support='One cached strategy per A training task; visible for first 4 student turns',
        teacher_max_calls=80, teacher_max_cost_cny=20,
        collection=dict(tasks=72, task_groups=len(train_groups), families=A_FAMILIES, rollout_seeds=[11,23], max_steps=40,
            temperature=.7, batch_size=8, fresh_rollouts_all_conditions=True),
        a_training=dict(method='target-view success-only SFT',
            matching='Common successful task/rollout-seed groups across all three sources; exclude literal Nothing happens labels, then match example counts',
            minimum_common_groups=8, max_steps=64, processed_examples=256,
            seeds=[101,202,303], support_masked=True),
        b_training=dict(shared_schedule='runs/learning-speed-v1/b_schedule.jsonl',
            shared_schedule_sha256=file_sha('runs/learning-speed-v1/b_schedule.jsonl'),
            adaptation_config='runs/learning-speed-v1/adaptation-config.json',
            family=B_FAMILY, processed_examples=512,max_steps=128,
            checkpoints=[0,8,16,24,32,64,128], seeds=[1101,1202,1303],
            parameterization='Frozen A + fresh B LoRA; paired B weights and sample order; optimizer reset'),
        evaluation=dict(a_query_tasks=24, b_query_tasks=24, b_only_each_checkpoint=True,
            a_retention_at=[0,128], no_support=True, batch_size=8, max_steps=40,
            temperature=0, query_never_used_for_tuning=True),
        primary_contrast='api minus generic: paired B normalized raw AUC and gain AUC, with initial ability/NLL',
        secondary=['api minus none', 'collection success and cost', 'unsupported A ability and A retention', 'per-family breakdown'],
        gate='If fewer than 8 common successful groups, report collection result and stop adaptation; do not switch analysis population silently',
        limitations=['Conditioning on mutual collection successes changes the estimand',
            'AUC can reflect prior knowledge and ceiling effects, not only plasticity',
            'Three paired seeds and one B family are a pilot',
            'Task-group disjointness does not imply room or pretraining disjointness',
            'Fixed batched BF16 inference may be sensitive to batch composition'],
        official_final_tests_used=False, report_plots=True)
    (ROOT/'design.json').write_text(json.dumps(design, indent=2)+'\n')
    files=[ROOT/f for f in ['train.jsonl','a_query.jsonl','b_query.jsonl','generic-support.jsonl','teacher-settings.json','design.json']]
    (ROOT/'inputs.sha256.json').write_text(json.dumps({str(p):file_sha(p) for p in files},indent=2)+'\n')
    print(json.dumps(dict(output=str(ROOT),A_train=len(train),A_query=len(a_query),B_query=len(b_query),conditions=design['conditions'],teacher_max_cost_cny=20)))


if __name__=='__main__': main()
