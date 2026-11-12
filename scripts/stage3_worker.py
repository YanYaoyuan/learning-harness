#!/usr/bin/env python3
"""Train exact-task-order controls and score them through the original evaluation kernel."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal

import run_api_harness as engine
import stage1_worker as core
from run_stage1_crossed_seeds import environment
from learning_harness.adaptation import composition_revision
from learning_harness.bounded_queue import save
from learning_harness.common import file_sha,read_jsonl

ROOT=Path('runs/variance-diagnostic-v1/stage3')


def digest(value):
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    def timeout(signum,frame):raise core.WorkerDeadline()
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout)
    signal.setitimer(signal.ITIMER_REAL,args.timeout)
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for path,sha in frozen['files'].items():
        if file_sha(path)!=sha:raise ValueError('Stage 3 frozen input changed: '+path)
    case=next(c for c in frozen['cases'] if c['name']==args.case)
    if case['gpu']!=args.gpu:raise ValueError('Frozen GPU assignment differs')
    directory=ROOT/'cases'/case['name'];adapter=directory/'A'
    recipe=json.loads(Path(case['schedule']).read_text())
    control_path=adapter/'seed-controls.json'
    if not control_path.exists() or not json.loads(control_path.read_text()).get('completed'):
        if adapter.exists() and any(adapter.iterdir()):raise ValueError('Partial training retained; inspect before resume')
        engine.command([engine.MODEL_PY,'scripts/train_lora_fixed_schedule.py','--schedule',case['schedule'],
            '--data',recipe['data'],'--model',engine.MODEL,'--revision',engine.REV,'--config',str(ROOT/'a-lora.json'),
            '--out',str(adapter),'--initialization-seed',str(case['initialization_seed']),
            '--sampling-seed',str(case['sampling_seed'])],environment(args.gpu),directory/'training.log')
    control=json.loads(control_path.read_text());metadata=json.loads((adapter/'experiment.json').read_text())
    if (not control['completed'] or control.get('selection_policy')!=case['arm']
            or control['initialization_seed']!=case['initialization_seed'] or control['sampling_seed']!=case['sampling_seed']
            or control['external_schedule_sha256']!=file_sha(case['schedule'])
            or control['schedule_sha256']!=recipe['schedule_sha256']
            or control['trained_adapter_sha256']!=file_sha(adapter/'adapter_model.safetensors')
            or metadata['training_data_sha256']!=recipe['data_sha256']
            or metadata['config']!=json.loads((ROOT/'a-lora.json').read_text())
            or metadata['processed_examples_rank0']!=768):
        raise ValueError('Real training differs from frozen task-matched control')
    rows=read_jsonl(recipe['data'])
    indices=recipe['indices'];task_ids=[r['task_id'] for r in rows]
    content_keys=[hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest() for r in rows]
    counts=Counter(task_ids[i] for i in indices)
    if (metadata['used_examples']!=len(rows) or digest(indices)!=control['schedule_sha256']
            or dict(counts)!=recipe['exposures_by_task']
            or digest([task_ids[i] for i in indices])!=recipe['reference_task_order_sha256']
            or len(set(indices))!=recipe['unique_examples']):
        raise ValueError('Exact task order, counts, or support verification failed')
    types={r['task_id']:r['task_type'] for r in read_jsonl('runs/api-harness-v1/train.jsonl')}
    exposure=dict(processed_examples=768,unique_examples=len(set(indices)),unique_tasks=len(counts),
        unique_task_content_examples=len({(task_ids[i],content_keys[i]) for i in indices}),
        distinct_content_examples=len({content_keys[i] for i in indices}),
        task_type_presentations=dict(Counter(types[task_ids[i]] for i in indices)),
        task_exposures=dict(counts),task_order_sha256=recipe['reference_task_order_sha256'],
        input_tokens=metadata['input_tokens_rank0'],target_tokens=metadata['target_tokens_rank0'])
    base_scores=json.loads((ROOT/'base-nll'/f"{case['source']}.json").read_text())
    if base_scores['data_sha256']!=recipe['data_sha256']:raise ValueError('Base score corpus mismatch')
    values=[v['mean_target_nll'] for v in base_scores['scores']]
    exposure.update(mean_presented_base_nll=sum(values[i] for i in indices)/len(indices),
                    mean_support_base_nll=sum(values[i] for i in set(indices))/len(set(indices)))
    if case['arm']!='full_reference' and exposure['unique_task_content_examples']!=256:
        raise ValueError('Content-distinct support budget changed')
    point=directory/'checkpoint';core.make_composition(adapter,point)
    scores={f:core.evaluate(point,directory/'evaluation'/f,f,case['source'],args.gpu)['metrics']['none']['success_rate'] for f in core.FAMILIES}
    nll=core.score_nll(point,directory/'diagnostic-nll.json')
    save(directory/'worker-result.json',dict(complete=True,smoke=False,case=case,success=scores,
        mean_success=sum(scores.values())/2,mean_nll=sum(nll[f]['task_mean_nll'] for f in core.FAMILIES)/2,
        control=control,exposure=exposure,composition_revision=composition_revision(point)))
    signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
