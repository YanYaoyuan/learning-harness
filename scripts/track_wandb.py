#!/usr/bin/env python3
"""Synchronize experiment logs with W&B in a separate observer process."""
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

from learning_harness.common import digest, file_sha, read_jsonl
# These parsers use only the standard library; importing them does not load Rerun.
from learning_harness.rerun_recording import LineCursor, final_scores, loss_point


def trace_row(row):
    """Explicit public-data allowlist; never serialize a config/environment wholesale."""
    names=['task_id','arm','rollout_seed','attempt_id','task_type','partition','step','response',
        'action','observation','success','invalid_action','support_active','input_tokens','output_tokens']
    value={k:row[k] for k in names if k in row}
    messages=row.get('messages',[])
    value['public_initial']=next((m['content'] for m in messages if m['role']=='user'),'')
    value['visible_support']=next((m['content'] for m in messages
        if m['role']=='user' and m['content'].startswith('[TRAINING SUPPORT] ')),None)
    return value


def case_for(directory):
    match=re.fullmatch(r'step-(\d+)',directory.name)
    if match:
        parent=directory.parent
        return (parent.parent.name if parent.name in ['A','B'] else parent.name),'adaptation',int(match[1])
    return directory.name.split('-shard')[0],'collection',None


def collection_values(rows):
    """An empty live shard has no success estimate yet."""
    if not rows:
        return None
    keys={(r['task_id'],r['rollout_seed']) for r in rows}
    if len(keys)!=len(rows):
        raise ValueError('Duplicate collection episodes in source shards')
    return dict(completed_episodes=len(rows),success_pct=100*sum(r['success'] for r in rows)/len(rows),
        wins=sum(r['success'] for r in rows),input_tokens=sum(r['input_tokens'] for r in rows),
        output_tokens=sum(r['output_tokens'] for r in rows))


def study_name(root):
    return root.parent.name+'-'+root.name if root.parent.name=='transfer-factorial-v1' else root.name


class CloudTracker:
    def __init__(self,args):
        import wandb
        self.w=wandb;self.args=args;self.out=args.out;self.out.mkdir(parents=True,exist_ok=True)
        self.state_path=self.out/'sync-state.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(runs={},sent={},training={})
        self.handles={};self.log_cursors={};self.studies=args.studies
        previous_url=next((r['url'] for r in self.state['runs'].values()),None)
        self.project_url=previous_url.rsplit('/runs/',1)[0] if previous_url else f'https://wandb.ai/{args.entity}/{args.project}'
        self.state.update(project_url=self.project_url,entity=args.entity,project=args.project)
        self.save()

    def save(self):
        p=self.state_path.with_suffix('.json.part');p.write_text(json.dumps(self.state,indent=2)+'\n');p.replace(self.state_path)
        links=dict(project_url=self.project_url,runs={k:v['url'] for k,v in self.state['runs'].items()})
        (self.out/'links.json').write_text(json.dumps(links,indent=2)+'\n')

    def get_run(self,root,case,kind):
        key=study_name(root)+'/'+kind+'/'+case
        if key in self.handles:return self.handles[key]
        record=self.state['runs'].get(key)
        run_id=record['id'] if record else hashlib.sha256((self.args.entity+'/'+self.args.project+'/'+str(root.absolute())+'/'+key).encode()).hexdigest()[:12]
        design=json.loads((root/'design.json').read_text())
        # Study design contains no runtime credentials; only these known fields are sent.
        config={k:design[k] for k in ['protocol','question','checkpoints','b_family','family','conditions','diagnostic','collection','a_training','b_training','evaluation','teacher_model'] if k in design}
        match=re.fullmatch(r'(.+)-seed(\d+)',case)
        if match:config.update(condition=match[1],b_seed=int(match[2]),a_seed=int(match[2])-1000 if match[1]!='base' else None)
        config.update(study=study_name(root),tracking_role=kind,source_logs='Read-only backfill and follow; source wall-clock time is not reconstructed')
        settings=self.w.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
            disable_job_creation=True,x_disable_stats=True,init_timeout=60)
        run=self.w.init(entity=self.args.entity,project=self.args.project,id=run_id,
            name=study_name(root)+' / '+case,group=study_name(root),job_type=kind,config=config,
            mode='online',resume='allow',reinit='create_new',dir=str(self.out.absolute()),
            save_code=False,settings=settings)
        if run is None or not run.url:raise RuntimeError('An authenticated online W&B run is required')
        self.project_url=run.url.rsplit('/runs/',1)[0]
        for prefix,x in [('train_A','optimizer_step'),('train_B','optimizer_step'),('eval','optimizer_step'),
                ('calibration','optimizer_step'),('diagnostic','optimizer_step'),('collection','completed_episodes')]:
            run.define_metric(prefix+'/'+x)
            run.define_metric(prefix+'/*',step_metric=prefix+'/'+x,step_sync=False)
        self.handles[key]=run
        self.state['runs'][key]=dict(id=run_id,url=run.url,finished=False,case=case,kind=kind,study=study_name(root))
        self.save();return run

    def emit(self,root,case,kind,event,values,*,summary=False):
        key=study_name(root)+'/'+kind+'/'+case+'/'+event;fingerprint=digest(values)
        if self.state['sent'].get(key)==fingerprint:return False
        run=self.get_run(root,case,kind)
        if summary:run.summary.update(values)
        else:run.log(values)
        self.state['sent'][key]=fingerprint;return True

    def attach_trace(self,root,directory,case,kind,step,episodes):
        key=study_name(root)+'/'+str(directory.relative_to(root))+'/trace'
        path=directory/'steps.jsonl';stat=path.stat();signature=[stat.st_size,stat.st_mtime_ns]
        stats=self.state.setdefault('trace_stats',{})
        if stats.get(key)==signature and key in self.state['sent']:return
        sha=file_sha(path)
        if self.state['sent'].get(key)==sha:
            stats[key]=signature;return
        out=self.out/'traces';out.mkdir(exist_ok=True)
        filename=out/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.jsonl.gz')
        preview=[]
        with path.open() as source,gzip.open(filename,'wt',encoding='utf8') as target:
            first=None
            for line in source:
                if not line.strip():continue
                row=trace_row(json.loads(line));target.write(json.dumps(row,ensure_ascii=False)+'\n')
                if first is None:first=row['attempt_id']
                if row['attempt_id']==first and len(preview)<40:
                    preview.append([row.get(k) for k in ['task_id','step','action','observation','response','support_active','success','input_tokens','output_tokens']])
        run=self.get_run(root,case,kind)
        artifact=self.w.Artifact('trace-'+hashlib.sha256(key.encode()).hexdigest()[:16],type='agent-traces',
            metadata=dict(study=study_name(root),source=str(directory.relative_to(root)),source_steps_sha256=sha,
                fields='Public task/response/action/observation/support/tokens; no credentials or request headers'))
        artifact.add_file(str(filename),name='public-agent-traces.jsonl.gz');run.log_artifact(artifact)
        episode_columns=['task_id','task_type','rollout_seed','success','steps','invalid_actions','input_tokens','output_tokens']
        values={'traces/episodes':self.w.Table(columns=episode_columns,data=[[r.get(k) for k in episode_columns] for r in episodes]),
            'traces/example_trajectory':self.w.Table(columns=['task_id','env_step','action','observation','model_response','support_active','success','input_tokens','output_tokens'],data=preview)}
        if step is not None:values['eval/optimizer_step']=step
        run.log(values);self.state['sent'][key]=sha;stats[key]=signature

    def training(self,root):
        for path in sorted((root/'logs').glob('*.log')):
            if root.name=='learning-speed-v1' and path.name.endswith('-training.log'):
                phase='B';case=path.name.removesuffix('-training.log');cadence=8;maximum=128
            elif path.name.startswith(('A-','B-')):
                phase=path.name[0];case=path.stem[2:];cadence=10 if phase=='A' else 8;maximum=64 if phase=='A' else 128
                if phase=='A':
                    m=re.fullmatch(r'(.+)-seed(\d+)',case);case=f'{m[1]}-seed{int(m[2])+1000}'
            else:continue
            key=str(path);saved=self.state['training'].setdefault(key,dict(offset=0,points=0))
            cursor=self.log_cursors.get(key)
            if cursor is None:
                cursor=LineCursor(path);cursor.offset=saved['offset'];self.log_cursors[key]=cursor
            for line in cursor.read(maximum=2000):
                point=loss_point(line,saved['points'],cadence,maximum)
                if point is None:continue
                saved['points']+=1;prefix='train_'+phase
                values={prefix+'/'+k:v for k,v in point.items() if v is not None}
                self.emit(root,case,'adaptation',f'{prefix}-{point["optimizer_step"]}',values)
            saved['offset']=cursor.offset

    def poll_study(self,root):
        live=[];collections=defaultdict(list);evaluations=defaultdict(set)
        for path in sorted(root.rglob('run.json')):
            try:meta=json.loads(path.read_text())
            except (json.JSONDecodeError,FileNotFoundError):continue
            if 'manifest' not in meta.get('config',{}):continue
            directory=path.parent;episodes_path=directory/'episodes.jsonl'
            if not episodes_path.exists():continue
            # Only complete lines are consumed while an episode file is growing.
            episodes=[json.loads(line) for line in LineCursor(episodes_path).read(maximum=100000)]
            case,kind,step=case_for(directory)
            live.append([str(directory.relative_to(root)),len(episodes),meta['planned_episodes'],
                sum(int(r['success']) for r in episodes)])
            if kind=='collection':collections[case].extend(episodes)
            if len(episodes)!=meta['planned_episodes']:continue
            manifest=read_jsonl(meta['config']['manifest']);scores=final_scores(meta,episodes,manifest)
            if scores is None:continue
            if kind=='adaptation':
                stage='calibration' if 'calibration' in directory.relative_to(root).parts else 'eval'
                values={stage+'/optimizer_step':step}
                for role,score in scores.items():
                    values.update({f'{stage}/{role}_success_pct':score['success_pct'],f'{stage}/{role}_wins':score['wins'],
                        f'{stage}/{role}_tasks':score['tasks']})
                self.emit(root,case,kind,str(directory.relative_to(root))+'/scores',values)
                if stage=='eval':evaluations[case].update((role,step) for role in scores)
            self.attach_trace(root,directory,case,kind,step,episodes)
        for case,rows in collections.items():
            values=collection_values(rows)
            if values is None:continue
            self.emit(root,case,'collection','progress',{'collection/'+k:v for k,v in values.items()})
        self.training(root)
        for path in sorted((root/'trajectories').glob('*/diagnostic-nll.json')):
            for step,record in sorted(json.loads(path.read_text()).items(),key=lambda kv:int(kv[0])):
                if record['used_for_updates'] or record['used_for_selection']:
                    raise ValueError('Diagnostic protocol violation')
                self.emit(root,path.parent.name,'adaptation','diagnostic-'+step,
                    {'diagnostic/optimizer_step':int(step),'diagnostic/task_mean_nll':record['task_mean_nll'],
                     'diagnostic/tasks':record['tasks'],'diagnostic/examples':record['examples']})
        monitor=self.get_run(root,'study-overview','observer')
        status={}
        for name in ['queue-state.json','main-progress.json','collection-progress.json','a-training-progress.json',
                'adaptation-progress.json','calibration-progress.json','calibration-result.json',
                'curriculum.json','main-result.json','teacher-generation.json','data-selection.json']:
            p=root/name
            if p.exists():
                try:
                    value=json.loads(p.read_text())
                    # Keep summaries compact; episode data have their own trace artifacts.
                    status[name]=({k:v for k,v in value.items() if k in ['complete','effects','contrasts','family']}
                                  if name=='main-result.json' else value)
                except (json.JSONDecodeError,FileNotFoundError):continue
        if 'teacher-generation.json' in status:
            teacher=status['teacher-generation.json']
            self.emit(root,'study-overview','observer','teacher',dict(teacher_calls=teacher['completed_calls'],
                estimated_list_price_cny=teacher['accounted_cny'],teacher_bank_complete=teacher['all_cards_complete']),summary=True)
        self.emit(root,'study-overview','observer','status',dict(pipeline_status=status,
            observed_model_episodes=sum(r[1] for r in live),observed_source_runs=len(live)),summary=True)
        event=study_name(root)+'/observer/live-table';sha=digest(live)
        if self.state['sent'].get(event)!=sha:
            monitor.log({'progress/evaluations':self.w.Table(columns=['source','recorded_episodes','planned_episodes','wins_so_far'],data=live)})
            self.state['sent'][event]=sha
        for image in sorted((root/'figures').rglob('*.png')):
            name='figures/'+image.relative_to(root/'figures').with_suffix('').as_posix();sha=file_sha(image)
            key=study_name(root)+'/'+name
            if self.state['sent'].get(key)!=sha:
                monitor.log({name:self.w.Image(str(image))});self.state['sent'][key]=sha
        # A native W&B run is finished only when its full measured evaluation grid exists.
        design=json.loads((root/'design.json').read_text())
        steps=design.get('checkpoints') or design['b_training']['checkpoints']
        expected={(role,k) for role in ['A','B'] for k in steps} if root.name=='learning-speed-v1' else (
            {('B',k) for k in steps}|{('A',0),('A',128)})
        for case,observed in evaluations.items():
            key=study_name(root)+'/adaptation/'+case
            if expected<=observed and not self.state['runs'].get(key,{}).get('finished'):
                run=self.get_run(root,case,'adaptation')
                run.summary.update({'measured_evaluation_complete':True})
                run.finish();self.handles.pop(key);self.state['runs'][key]['finished']=True

    def poll(self):
        for root in self.studies:self.poll_study(root)
        self.state.update(status='watching',pid=os.getpid(),updated_at_unix=time.time())
        self.save()

    def close(self):
        for run in self.handles.values():run.finish()
        self.w.teardown();self.state['status']='observer_stopped';self.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',nargs='+',type=Path,default=[Path('runs/learning-speed-v1'),Path('runs/api-harness-v1')])
    p.add_argument('--entity');p.add_argument('--project',default='learning-harness-lab')
    p.add_argument('--out',type=Path,default=Path('output/wandb'))
    p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=20)
    args=p.parse_args()
    if args.poll_seconds<1:p.error('poll interval must be at least one second')
    if args.entity is None:
        import wandb
        args.entity=wandb.Api(timeout=25).default_entity
    if not args.entity:raise RuntimeError('A valid authenticated W&B entity is required')
    tracker=CloudTracker(args);stop=False
    def terminate(*unused):
        nonlocal stop;stop=True
    signal.signal(signal.SIGINT,terminate);signal.signal(signal.SIGTERM,terminate)
    try:
        while not stop:
            tracker.poll();print(json.dumps(dict(project=tracker.project_url,runs=len(tracker.state['runs']),status='watching')),flush=True)
            if not args.watch:break
            for _ in range(max(1,int(args.poll_seconds*5))):
                if stop:break
                time.sleep(.2)
    finally:tracker.close()


if __name__=='__main__':main()
