#!/usr/bin/env python3
"""Run the frozen API-scaffold replication, without exporting credentials to GPUs."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.request

from learning_harness.analysis import export_sft, summarize
from learning_harness.common import digest, file_sha, read_jsonl, write_jsonl

ROOT=Path('runs/api-harness-v1')
MODEL='data/models/Qwen3-4B-Instruct-2507'
REV='cdbee75f17c01a7cc42f958dc650907174af0554'
MODEL_PY=str(Path('.venv-model/bin/python').absolute())
CPU_PY=str(Path('.venv/bin/python').absolute())
CONDITIONS=['none','generic','api']


def frozen():
    for filename in ['inputs.sha256.json','teacher-bank.sha256.json','training-inputs.sha256.json']:
        p=ROOT/filename
        if p.exists():
            for name,expected in json.loads(p.read_text()).items():
                if file_sha(name)!=expected:raise ValueError(f'Frozen input changed: {name}')
    design=json.loads((ROOT/'design.json').read_text())
    assert file_sha(design['b_training']['shared_schedule'])==design['b_training']['shared_schedule_sha256']
    return design


def environment(gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(Path('src').absolute()),
        ALFWORLD_DATA=str(Path('data/alfworld').absolute()))
    # Student subprocesses do not need teacher credentials.
    for key in list(env):
        if key.startswith('LH_TEACHER_'):env.pop(key)
    return env


def command(args,env,log,timeout=3600):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('w') as handle:
        subprocess.run(args,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=timeout)


@contextmanager
def server(gpu,alias,revision,log,composition=None):
    env=environment(gpu);port=19780+gpu
    env.update(LH_STUDENT_BASE_URL=f'http://127.0.0.1:{port}/v1',LH_STUDENT_MODEL=alias,LH_STUDENT_REVISION=revision)
    args=[MODEL_PY,'scripts/hf_batch_server.py','--model',MODEL,'--revision',REV,
        '--served-model-name',alias,'--port',str(port)]
    if composition:args+=['--composition',str(composition)]
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('w') as handle:
        process=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None:raise RuntimeError('Local server exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:health=json.load(response)
                    if health['revision']==revision and health['model']==alias:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('Local server startup')
                time.sleep(1)
            yield env
        finally:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()


def gpu_jobs(stage,jobs,work):
    gpus=queue.Queue()
    for gpu in [0,1]:gpus.put(gpu)
    progress={};lock=threading.Lock()
    def update(name,status,gpu,**extra):
        with lock:
            progress[name]=dict(status=status,gpu=gpu,**extra)
            p=ROOT/f'{stage}-progress.json.part';p.write_text(json.dumps(progress,indent=2)+'\n')
            p.replace(ROOT/f'{stage}-progress.json')
            print(stage,name,status,flush=True)
    def one(job):
        gpu=gpus.get();name=job['name'];started=time.monotonic()
        try:
            update(name,'running',gpu)
            work(job,gpu,lambda status:update(name,status,gpu))
            update(name,'completed',gpu,elapsed_seconds=time.monotonic()-started)
        except Exception as exc:
            update(name,'failed',gpu,error_type=type(exc).__name__);raise
        finally:gpus.put(gpu)
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,job):job['name'] for job in jobs}
        for future in as_completed(futures):
            try:future.result()
            except Exception as exc:errors.append([futures[future],type(exc).__name__])
    if errors:raise RuntimeError(f'Failed {stage} jobs: {errors}')


def collect():
    frozen()
    teacher=json.loads((ROOT/'teacher-generation.json').read_text())
    if not teacher['all_cards_complete']:raise ValueError('Teacher bank is incomplete')
    tasks=read_jsonl(ROOT/'train.jsonl')
    for shard in [0,1]:
        manifest=ROOT/f'train-shard{shard}.jsonl'
        if manifest.exists():assert read_jsonl(manifest)==tasks[shard::2]
        else:write_jsonl(manifest,tasks[shard::2])
    jobs=[dict(name=f'{c}-shard{s}',condition=c,shard=s) for c in CONDITIONS for s in [0,1]]
    def work(job,gpu,status):
        out=ROOT/'collection'/job['name'];out.mkdir(parents=True,exist_ok=True)
        config=json.loads(Path('configs/collect_train.json').read_text())
        c=job['condition'];config.update(protocol_id='api-harness-v1-collection',
            manifest=str(ROOT/f"train-shard{job['shard']}.jsonl"),
            teacher_bank=str(ROOT/f'{c}-support.jsonl') if c!='none' else None,
            arms=['none'] if c=='none' else ['early'],rollout_seeds=[11,23],
            max_steps=40,max_new_tokens=128,max_student_output_tokens=5120,temperature=.7,batch_size=8)
        path=out/'config.json'
        if path.exists():assert json.loads(path.read_text())==config
        else:path.write_text(json.dumps(config,indent=2)+'\n')
        summary=out/'summary.json'
        if summary.exists() and json.loads(summary.read_text())['complete']:return
        with server(gpu,'api-harness-base',REV,out/'server.log') as env:
            command([CPU_PY,'-m','learning_harness.batch_runner','--config',str(path),'--out',str(out)],env,out/'rollout.log')
        result=summarize(out)
        assert result['complete'] and all(v['error_count']==0 for v in result['metrics'].values())
    gpu_jobs('collection',jobs,work)
    # Merge each source separately, retaining its original arm and support timing.
    for condition in CONDITIONS:
        sources=[ROOT/'collection'/f'{condition}-shard{s}' for s in [0,1]]
        metadata=[json.loads((p/'run.json').read_text()) for p in sources]
        assert all(m['revision']==REV and m['planned_episodes']==72 for m in metadata)
        out=ROOT/'collection'/condition;out.mkdir(exist_ok=True)
        for filename in ['episodes.jsonl','steps.jsonl','calls.jsonl']:
            rows=[r for p in sources for r in read_jsonl(p/filename)]
            write_jsonl(out/filename,rows)
        rows=read_jsonl(out/'episodes.jsonl')
        assert len({(r['task_id'],r['rollout_seed']) for r in rows})==144
        (out/'run.json').write_text(json.dumps(dict(fingerprint=digest([m['fingerprint'] for m in metadata]),
            planned_episodes=144,source_workers=metadata),indent=2)+'\n')
        summarize(out)
    command(['python3','scripts/api_harness_report.py','--stage','collection'],environment(0),ROOT/'collection-report.log')


def build():
    design=frozen()
    if (ROOT/'data-selection.json').exists():
        result=json.loads((ROOT/'data-selection.json').read_text())
        return result['adaptation_gate_passed']
    from transformers import AutoTokenizer
    from train_lora import encode_example
    from matched_study import evenly_spaced
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=False)
    root=ROOT/'corpora';root.mkdir(exist_ok=True)
    sources={};metadata={};eligibility={}
    for c in CONDITIONS:
        source_dir=ROOT/'collection'/c;raw=root/f'raw-{c}.jsonl'
        meta=export_sft(source_dir,'none' if c=='none' else 'early',raw)
        steps=read_jsonl(source_dir/'steps.jsonl')
        feedback={(r['task_id'],r['rollout_seed'],r['step']):r['observation'].strip() for r in steps}
        rows=read_jsonl(raw);filtered=[];overlength=0;nothing=0
        for row in rows:
            if feedback[row['task_id'],row['rollout_seed'],row['step']]=='Nothing happens.':nothing+=1;continue
            if encode_example(tokenizer,row,2048) is None:overlength+=1;continue
            filtered.append(row)
        sources[c]=defaultdict(list)
        for row in filtered:sources[c][row['task_id'],row['rollout_seed']].append(row)
        metadata[c]=meta;eligibility[c]=dict(raw_examples=len(rows),literal_nothing_labels_removed=nothing,
            overlength_removed=overlength,eligible_examples=len(filtered))
    common=sorted(set.intersection(*(set(v) for v in sources.values())))
    task_index={r['task_id']:r for r in read_jsonl(ROOT/'train.jsonl')}
    group_count=len({task_index[t]['task_group'] for t,s in common})
    passed=group_count>=design['a_training']['minimum_common_groups']
    selection=dict(eligible=eligibility,common_task_seed_pairs=len(common),common_task_groups=group_count,
        common_tasks=len({t for t,s in common}),adaptation_gate_passed=passed,
        matching_uses_query=False,matched_examples_each=0,
        task_type_coverage=dict(Counter(task_index[t]['task_type'] for t in {t for t,s in common})))
    if passed:
        counts=[];matched={c:[] for c in CONDITIONS}
        for pair in common:
            n=min(len(sources[c][pair]) for c in CONDITIONS)
            counts.append(dict(task_id=pair[0],rollout_seed=pair[1],examples_each=n))
            for c in CONDITIONS:
                matched[c].extend(dict(row,matched_index=i) for i,row in enumerate(evenly_spaced(sources[c][pair],n)))
        for c in CONDITIONS:
            matched[c].sort(key=lambda r:digest([20261109,r['task_id'],r['rollout_seed'],r['matched_index']]))
        keys=lambda rows:[(r['task_id'],r['rollout_seed'],r['matched_index']) for r in rows]
        assert keys(matched['none'])==keys(matched['generic'])==keys(matched['api'])
        for c in CONDITIONS:
            path=root/f'matched-{c}.jsonl';write_jsonl(path,matched[c])
            Path(str(path)+'.meta.json').write_text(json.dumps(dict(metadata[c],examples=len(matched[c]),
                matching_counts=counts,matched_group_count=group_count,source_condition=c,
                filtering=eligibility[c],query_used=False),indent=2)+'\n')
        selection['matched_examples_each']=len(matched['none']);selection['pair_counts']=counts
        a_config=json.loads(Path('runs/dual5090-v2/lora.json').read_text())
        (ROOT/'a-lora.json').write_text(json.dumps(a_config,indent=2)+'\n')
        b_config=json.loads(Path(design['b_training']['adaptation_config']).read_text())
        b_config['checkpoints']=design['b_training']['checkpoints']
        (ROOT/'b-lora.json').write_text(json.dumps(b_config,indent=2)+'\n')
    (ROOT/'data-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    files=list(root.glob('*.jsonl*'))+[ROOT/'data-selection.json']
    if passed:files += [ROOT/'a-lora.json',ROOT/'b-lora.json']
    (ROOT/'training-inputs.sha256.json').write_text(json.dumps({str(p):file_sha(p) for p in files},indent=2)+'\n')
    print(json.dumps(selection),flush=True)
    return passed


def train_a():
    frozen()
    assert json.loads((ROOT/'data-selection.json').read_text())['adaptation_gate_passed']
    jobs=[dict(name=f'{c}-seed{s}',condition=c,seed=s) for s in [101,202,303] for c in CONDITIONS]
    def work(job,gpu,status):
        out=ROOT/'a-checkpoints'/job['name'];data=ROOT/'corpora'/f"matched-{job['condition']}.jsonl"
        if not (out/'experiment.json').exists():
            command([MODEL_PY,'scripts/train_lora.py','--model',MODEL,'--revision',REV,'--data',str(data),
                '--config',str(ROOT/'a-lora.json'),'--seed',str(job['seed']),'--out',str(out)],
                environment(gpu),ROOT/'logs'/f"A-{job['name']}.log")
        meta=json.loads((out/'experiment.json').read_text())
        assert meta['processed_examples_rank0']==256 and meta['training_data_sha256']==file_sha(data)
        assert meta['training_seed']==job['seed'] and meta['config']==json.loads((ROOT/'a-lora.json').read_text())
    gpu_jobs('a-training',jobs,work)


def evaluate(gpu,case,checkpoint,role,step):
    from learning_harness.adaptation import composition_revision
    revision=composition_revision(checkpoint)
    out=ROOT/'evaluation'/case/role/f'step-{step:04d}';out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('configs/probe.json').read_text())
    config.update(protocol_id=f'api-harness-v1-{role}',manifest=str(ROOT/f'{role.lower()}_query.jsonl'),
        batch_size=8,max_steps=40,max_student_output_tokens=5120)
    cfg=out/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:cfg.write_text(json.dumps(config,indent=2)+'\n')
    summary=out/'summary.json'
    if summary.exists() and json.loads(summary.read_text())['complete']:
        assert json.loads((out/'run.json').read_text())['revision']==revision
        return
    with server(gpu,f'{case}-{role}-{step}',revision,out/'server.log',checkpoint) as env:
        command([CPU_PY,'-m','learning_harness.batch_runner','--config',str(cfg),
            '--evaluation','--out',str(out)],env,out/'rollout.log')
    result=summarize(out)
    assert result['complete'] and result['metrics']['none']['error_count']==0


def adapt():
    design=frozen()
    from learning_harness.adaptation import composition_revision
    jobs=[dict(name=f'{c}-seed{s+1000}',condition=c,a_seed=s,b_seed=s+1000) for s in [101,202,303] for c in CONDITIONS]
    def work(job,gpu,status):
        case=job['name'];out=ROOT/'trajectories'/case
        a=ROOT/'a-checkpoints'/f"{job['condition']}-seed{job['a_seed']}"
        if not (out/'trajectory.json').exists():
            status('training-B')
            command([MODEL_PY,'scripts/train_adaptation.py','--a-adapter',str(a),
                '--config',str(ROOT/'b-lora.json'),'--data',design['b_training']['shared_schedule'],
                '--seed',str(job['b_seed']),'--out',str(out)],environment(gpu),ROOT/'logs'/f'B-{case}.log')
        meta=json.loads((out/'trajectory.json').read_text())
        assert meta['processed_examples']==512 and meta['a_frozen_verified'] and meta['b_updated_verified']
        assert meta['data_sha256']==design['b_training']['shared_schedule_sha256']
        assert meta['config']==json.loads((ROOT/'b-lora.json').read_text())
        for step in design['b_training']['checkpoints']:
            checkpoint=out/f'step-{step:04d}'
            status(f'evaluating-B-{step}');evaluate(gpu,case,checkpoint,'B',step)
            if step in design['evaluation']['a_retention_at']:
                status(f'evaluating-A-{step}');evaluate(gpu,case,checkpoint,'A',step)
    gpu_jobs('adaptation',jobs,work)


def wait_for_previous(pid):
    path=Path(f'/proc/{pid}/cmdline')
    while path.exists():
        try:args=path.read_bytes()
        except FileNotFoundError:break
        if b'scripts/run_learning_speed.py' not in args:
            raise ValueError('Previous PID was reused or is not the expected experiment')
        (ROOT/'queue-state.json').write_text(json.dumps(dict(status='waiting_for_live_previous_experiment',pid=pid))+'\n')
        time.sleep(20)
    result=Path('runs/learning-speed-v1/main-result.json')
    if not result.exists() or not json.loads(result.read_text()).get('complete'):
        raise ValueError('Previous process ended without a complete measured report; inspect it before GPU handoff')
    (ROOT/'queue-state.json').write_text(json.dumps(dict(status='previous_completed_gpu_handoff',previous_pid=pid))+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['collect','build','train-a','adapt','report','all'],default='all')
    p.add_argument('--wait-pid',type=int);args=p.parse_args()
    frozen()
    if args.wait_pid:wait_for_previous(args.wait_pid)
    if args.stage in ['all','collect']:collect()
    if args.stage in ['all','build']:
        if not build():
            print('Common-success gate failed; collection result remains valid, adaptation stopped.',flush=True)
            return
    if args.stage in ['all','train-a']:train_a()
    if args.stage in ['all','adapt']:adapt()
    if args.stage in ['all','report']:
        command(['python3','scripts/api_harness_report.py','--stage','main'],environment(0),ROOT/'main-report.log')


if __name__=='__main__':main()
