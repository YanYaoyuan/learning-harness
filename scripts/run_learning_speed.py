#!/usr/bin/env python3
"""Run the calibration gate, then optionally the frozen shared-data comparison."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.request

from learning_harness.common import file_sha,read_jsonl,write_jsonl
from learning_harness.adaptation import composition_revision

ROOT=Path('runs/learning-speed-v1')
BASE='data/models/Qwen3-4B-Instruct-2507'
REV='cdbee75f17c01a7cc42f958dc650907174af0554'


def name(job):return f"{job['condition']}-seed{job['b_seed']}"


def run_stage(stage):
    for f in ['inputs.sha256.json','curriculum.sha256.json']:
        for path,expected in json.loads((ROOT/f).read_text()).items():
            if file_sha(path)!=expected:raise ValueError(f'Frozen input changed: {path}')
    design=json.loads((ROOT/'design.json').read_text());all_jobs=design['initializations']
    if stage=='main':
        gate=json.loads((ROOT/'calibration-result.json').read_text())
        if not gate['gate_passed']:raise ValueError('Calibration gate has not passed; keep query closed')
        combined=[dict(r,evaluation_role='B_query') for r in read_jsonl(ROOT/'b_query.jsonl')]
        combined += [dict(r,evaluation_role='A_retention') for r in read_jsonl(ROOT/'a_retention.jsonl')]
        manifest=ROOT/'main-evaluation.jsonl'
        if manifest.exists():assert read_jsonl(manifest)==combined
        else:write_jsonl(manifest,combined)
    else:manifest=ROOT/'b_calibration.jsonl'
    jobs=[j for j in all_jobs if stage=='main' or j['condition']=='base']
    gpu_queue=queue.Queue()
    for g in [0,1]:gpu_queue.put(g)
    progress={};lock=threading.Lock()

    def status(job,phase,**extra):
        with lock:
            progress[name(job)]=dict(status=phase,**extra)
            temp=ROOT/f'{stage}-progress.json.part';temp.write_text(json.dumps(progress,indent=2)+'\n')
            temp.replace(ROOT/f'{stage}-progress.json')
            print(stage,name(job),phase,flush=True)

    def one(job):
        gpu=gpu_queue.get();case=name(job);server=None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(Path('src').absolute()),
                 ALFWORLD_DATA=str(Path('data/alfworld').absolute()))
        trajectory=ROOT/'trajectories'/case
        logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
        started=time.monotonic()
        try:
            if job['a_adapter']:
                assert file_sha(Path(job['a_adapter'])/'adapter_model.safetensors')==job['a_adapter_sha256']
            if not (trajectory/'trajectory.json').exists():
                status(job,'training',gpu=gpu)
                command=[str(Path('.venv-model/bin/python').absolute()),'scripts/train_adaptation.py',
                    '--seed',str(job['b_seed']),'--out',str(trajectory)]
                if job['a_adapter']:command+=['--a-adapter',job['a_adapter']]
                with (logdir/f'{case}-training.log').open('w') as log:
                    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            meta=json.loads((trajectory/'trajectory.json').read_text())
            assert meta['processed_examples']==512 and meta['data_sha256']==file_sha(ROOT/'b_schedule.jsonl')
            assert meta['config']==json.loads((ROOT/'adaptation-config.json').read_text())
            assert meta['a_frozen_verified'] and meta['b_updated_verified']
            for step in design['checkpoints']:
                checkpoint=trajectory/f'step-{step:04d}';served=composition_revision(checkpoint)
                evaluation=ROOT/'evaluation'/stage/case/f'step-{step:04d}'
                evaluation.mkdir(parents=True,exist_ok=True)
                if (evaluation/'summary.json').exists():
                    summary=json.loads((evaluation/'summary.json').read_text())
                    if summary['complete']:
                        assert json.loads((evaluation/'run.json').read_text())['revision']==served
                        continue
                status(job,f'evaluating-{step}',gpu=gpu)
                port=19680+gpu;alias=f'{case}-step{step}'
                config=json.loads(Path('configs/probe.json').read_text())
                config.update(protocol_id=f'learning-speed-v1-{stage}',manifest=str(manifest),
                    batch_size=8,max_steps=40,max_student_output_tokens=5120)
                cfg=evaluation/'config.json';cfg.write_text(json.dumps(config,indent=2)+'\n')
                env.update(LH_STUDENT_BASE_URL=f'http://127.0.0.1:{port}/v1',LH_STUDENT_MODEL=alias,LH_STUDENT_REVISION=served)
                with (evaluation/'server.log').open('w') as serverlog:
                    server=subprocess.Popen([str(Path('.venv-model/bin/python').absolute()),'scripts/hf_batch_server.py',
                        '--composition',str(checkpoint),'--served-model-name',alias,'--port',str(port)],
                        env=env,stdout=serverlog,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+180
                    while True:
                        if server.poll() is not None:raise RuntimeError('Composition server exited')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:health=json.load(response)
                            if health['revision']==served:break
                        except OSError:pass
                        if time.monotonic()>deadline:raise TimeoutError('Server startup')
                        time.sleep(1)
                    with (evaluation/'rollout.log').open('w') as log:
                        subprocess.run([str(Path('.venv/bin/python').absolute()),'-m','learning_harness.batch_runner',
                            '--config',str(cfg),'--evaluation','--out',str(evaluation)],env=env,
                            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
                    subprocess.run([str(Path('.venv/bin/python').absolute()),'-m','learning_harness','summarize',str(evaluation)],
                                   env=env,stdout=subprocess.DEVNULL,check=True)
                server.terminate();server.wait(timeout=20);server=None
            status(job,'completed',gpu=gpu,elapsed_seconds=time.monotonic()-started)
        except Exception as exc:
            status(job,'failed',gpu=gpu,error_type=type(exc).__name__);raise
        finally:
            if server is not None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait()
            gpu_queue.put(gpu)

    failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(one,j):name(j) for j in jobs}
        for future in as_completed(futures):
            try:future.result()
            except Exception as exc:failures.append((futures[future],type(exc).__name__))
    if failures:raise RuntimeError(f'Failed trajectories: {failures}')
    env=dict(os.environ,PYTHONPATH=str(Path('src').absolute()))
    with (ROOT/f'{stage}-report.log').open('w') as log:
        subprocess.run(['python3','scripts/learning_speed_report.py','--stage',stage],env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['calibration','main'],default='calibration')
    parser.add_argument('--advance',action='store_true');args=parser.parse_args()
    run_stage(args.stage)
    if args.advance and args.stage=='calibration':
        gate=json.loads((ROOT/'calibration-result.json').read_text())
        if gate['gate_passed']:run_stage('main')
        else:print('Learnability gate failed; query remains unopened.',flush=True)
