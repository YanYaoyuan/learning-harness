#!/usr/bin/env python3
"""Freeze shared B examples and a task-balanced order before calibration/query."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from learning_harness.analysis import export_sft
from learning_harness.common import file_sha, read_jsonl, write_jsonl

ROOT=Path('runs/learning-speed-v1')


def main():
    if (ROOT/'curriculum.json').exists():raise ValueError('Shared B data already frozen')
    from transformers import AutoTokenizer
    from train_lora import encode_example
    for path,h in json.loads((ROOT/'inputs.sha256.json').read_text()).items():assert file_sha(path)==h
    tokenizer=AutoTokenizer.from_pretrained('data/models/Qwen3-4B-Instruct-2507',trust_remote_code=False)
    output=ROOT/'b_examples.jsonl'
    meta=export_sft(ROOT/'reference','reference',output)
    source=read_jsonl(output)
    eligible=[r for r in source if encode_example(tokenizer,r,2048) is not None]
    groups=defaultdict(list)
    for index,row in enumerate(eligible):groups[row['task_id']].append(index)
    design=json.loads((ROOT/'design.json').read_text())
    if len(groups)<design['minimum_b_data']['successful_task_groups'] or len(eligible)<design['minimum_b_data']['eligible_action_examples']:
        raise ValueError('Shared-data gate failed: insufficient successful task groups or examples')
    rng=random.Random(777);order=[];queues={g:[] for g in groups}
    while len(order)<512:
        cycle=sorted(groups);rng.shuffle(cycle)
        for group in cycle:
            if not queues[group]:queues[group]=list(groups[group]);rng.shuffle(queues[group])
            order.append(queues[group].pop())
            if len(order)==512:break
    scheduled=[dict(eligible[index],shared_index=slot,original_example_index=index) for slot,index in enumerate(order)]
    counts=Counter(r['task_id'] for r in scheduled)
    assert max(counts.values())-min(counts.values())<=1
    write_jsonl(output,eligible);write_jsonl(ROOT/'b_schedule.jsonl',scheduled)
    Path(str(output)+'.meta.json').write_text(json.dumps(dict(meta,examples=len(eligible),overlength_dropped=len(source)-len(eligible)),indent=2)+'\n')
    config=json.loads(Path('runs/dual5090-v2/lora.json').read_text())
    config.update(max_steps=128,checkpoints=[0,8,32,128],data_seed=777,training_seeds=[1101,1202,1303])
    (ROOT/'adaptation-config.json').write_text(json.dumps(config,indent=2)+'\n')
    curriculum=dict(teacher='scripted public-observation reference',reference_successes=meta['successful_episodes'],
        successful_task_groups=len(groups),eligible_action_examples=len(eligible),overlength_dropped=len(source)-len(eligible),
        scheduled_examples=512,group_example_count_range=[min(counts.values()),max(counts.values())],
        data_seed=777,shared_sequence_across_all_conditions=True,query_used=False,calibration_used=False,
        teacher_private_state_access=False,paid_api_calls=0,model_generation_calls=0,
        scheduled_cooling_actions=sum('"cool ' in r['messages'][-1]['content'] for r in scheduled))
    (ROOT/'curriculum.json').write_text(json.dumps(curriculum,indent=2)+'\n')
    files=[ROOT/n for n in ['b_examples.jsonl','b_examples.jsonl.meta.json','b_schedule.jsonl','adaptation-config.json','curriculum.json']]
    (ROOT/'curriculum.sha256.json').write_text(json.dumps({str(p):file_sha(p) for p in files},indent=2)+'\n')
    print(json.dumps(curriculum,indent=2))


if __name__=='__main__':main()
