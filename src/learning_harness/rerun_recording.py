"""Import append-only experiment logs into Rerun.

The optional Rerun SDK is loaded by the recording command."""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


class LineCursor:
    """Read complete UTF-8 lines, retaining an incomplete final line for next poll."""
    def __init__(self, path):
        self.path=Path(path);self.offset=0;self.identity=None;self.lines_read=0

    def read(self, maximum=500):
        if not self.path.exists():return
        stat=self.path.stat();identity=(stat.st_dev,stat.st_ino)
        if self.identity is not None and (identity!=self.identity or stat.st_size<self.offset):
            raise RuntimeError(f'Source log was replaced/truncated: {self.path}; rebuild the recording')
        self.identity=identity
        with self.path.open('rb') as handle:
            handle.seek(self.offset)
            for _ in range(maximum):
                raw=handle.readline()
                if not raw or not raw.endswith(b'\n'):break
                text=raw.decode('utf-8');self.offset=handle.tell();self.lines_read+=1
                yield text


def final_scores(metadata,episodes,manifest):
    """Never turn a partial evaluation or duplicate episodes into checkpoint scores."""
    if len(episodes)!=metadata['planned_episodes']:return None
    config=metadata['config']
    expected={(r['task_id'],a,s) for r in manifest for a in config['arms'] for s in config['rollout_seeds']}
    keys={(r['task_id'],r['arm'],r['rollout_seed']) for r in episodes}
    if len(keys)!=len(episodes) or keys!=expected:
        raise ValueError('Completed evaluation has duplicated or unexpected episodes')
    if any(r['status']!='completed' for r in episodes):
        raise ValueError('Failed episodes cannot produce checkpoint scores')
    roles={}
    for row in manifest:
        role=(row.get('evaluation_role') or row.get('adaptation_role') or 'overall').lower()
        roles[row['task_id']]='B' if role.startswith('b_') else 'A' if role.startswith('a_') else 'overall'
    grouped=defaultdict(list)
    for row in episodes:grouped[roles[row['task_id']]].append(row)
    return {role:dict(wins=sum(int(r['success']) for r in rows),tasks=len(rows),
        success_pct=100*sum(int(r['success']) for r in rows)/len(rows)) for role,rows in grouped.items()}


def loss_point(line,index,cadence,max_steps):
    # Trainer writes Python dictionaries amid tqdm's carriage-return progress bars.
    match=re.search(r"\{[^\n\r]*'loss':[^\n\r]*\}",line)
    if not match:return None
    value=ast.literal_eval(match.group());step=(index+1)*cadence
    if step>max_steps:raise ValueError('Unexpected training log cadence; do not guess optimizer steps')
    return dict(optimizer_step=step,loss=float(value['loss']),
        learning_rate=float(value['learning_rate']) if 'learning_rate' in value else None,
        grad_norm=float(value['grad_norm']) if 'grad_norm' in value else None)


def safe_part(value):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(value))


def blueprint():
    import rerun.blueprint as rrb
    return rrb.Blueprint(rrb.Vertical(
        rrb.Horizontal(
            rrb.TextDocumentView(origin='/live/transition',name='Latest imported transition'),
            rrb.Tabs(rrb.TextLogView(origin='/events',name='Agent event log'),
                rrb.TextDocumentView(origin='/live/status',name='Progress and API cost')),column_shares=[1.3,1]),
        rrb.Horizontal(
            rrb.TimeSeriesView(origin='/metrics/checkpoints/main',contents='$origin/**/B/success_pct',
                name='New-task success (%)',axis_y=rrb.ScalarAxis(range=[0,100])),
            rrb.TimeSeriesView(origin='/training/B',contents='$origin/**/loss',name='B training loss'),
            rrb.TimeSeriesView(origin='/metrics/checkpoints/main',contents='$origin/**/A/success_pct',
                name='Old-task success (%)',axis_y=rrb.ScalarAxis(range=[0,100]))),row_shares=[1.3,1]),
        rrb.TimePanel(timeline='optimizer_step',state='expanded'),auto_views=False)


class StudyRecorder:
    def __init__(self,root,output,grpc_url=None):
        import rerun as rr
        self.rr=rr;self.root=Path(root);self.output=Path(output)
        self.recording=rr.RecordingStream('harness-'+safe_part(self.root.name))
        sinks=[rr.FileSink(str(self.output))]
        if grpc_url:sinks.append(rr.GrpcSink(grpc_url))
        self.recording.set_sinks(*sinks,default_blueprint=blueprint())
        self.runs={};self.logs={};self.event=0;self.status_text=None
        self.counts=dict(steps=0,episodes=0,checkpoint_series_points=0,training_loss_points=0)
        self.latest='Waiting for source logs.'
        self.recording.log('metadata/notes',rr.TextDocument(
            '# Read-only experiment recording\n\n'
            'Source: '+self.root.as_posix()+'\n\n'
            '- `optimizer_step`: real checkpoint updates; loss steps use the known Trainer logging cadence.\n'
            '- `env_step`: individual agent/environment turns; select an episode entity to replay.\n'
            '- `ingest_event`: import order, **not historical wall-clock time**.\n'
            '- The latest-transition panel always shows the latest imported row.\n'
            '- Checkpoint success is emitted only after every planned episode completes.\n'
            '- Full prompts remain in the original JSONL; this recording keeps the public initial observation, '
            'hint, response, action, feedback, support flag, tokens, and outcome.\n'
            '- No credential files, hidden environment state, or API request headers are imported.',
            media_type='text/markdown'),static=True)

    def time(self,*,env_step=None,optimizer_step=None,episode=None):
        self.event+=1;self.recording.reset_time()
        self.recording.set_time('ingest_event',sequence=self.event)
        if env_step is not None:self.recording.set_time('env_step',sequence=int(env_step))
        if optimizer_step is not None:self.recording.set_time('optimizer_step',sequence=int(optimizer_step))
        if episode is not None:self.recording.set_time('completed_episode',sequence=int(episode))

    def discover(self):
        for path in sorted(self.root.rglob('run.json')):
            if path in self.runs:continue
            try:meta=json.loads(path.read_text())
            except (json.JSONDecodeError,FileNotFoundError):continue  # first write may still be in progress
            if 'config' not in meta or 'manifest' not in meta['config']:continue
            # Merged aliases and scripted reference-policy runs have different schemas.
            manifest_path=Path(meta['config']['manifest'])
            if not manifest_path.exists():continue
            manifest=[json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
            directory=path.parent;relative=directory.relative_to(self.root)
            name='/'.join(safe_part(p) for p in relative.parts)
            match=re.fullmatch(r'step-(\d+)',directory.name)
            step=int(match[1]) if match else None
            case=directory.parent.name
            if case in ['A','B']:case=directory.parent.parent.name
            stage='calibration' if 'calibration' in relative.parts else 'main'
            self.runs[path]=dict(meta=meta,manifest=manifest,directory=directory,name=name,
                step=step,case=safe_part(case),stage=stage,episodes=[],score_emitted=False,
                steps_cursor=LineCursor(directory/'steps.jsonl'),episode_cursor=LineCursor(directory/'episodes.jsonl'))
            self.recording.log('runs/'+name+'/metadata',self.rr.TextDocument(json.dumps(dict(
                source=relative.as_posix(),model=meta.get('model'),revision=meta.get('revision'),
                planned_episodes=meta['planned_episodes'],protocol=meta['config'].get('protocol_id')),
                indent=2)),static=True)

    def log_step(self,state,row):
        rr=self.rr;self.time(env_step=row['step'],optimizer_step=state['step'])
        identity=[row['task_id'],row['arm'],row['rollout_seed'],row['attempt_id']]
        key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:14]
        entity=f"rollouts/{state['name']}/{key}"
        initial=next((m['content'] for m in row['messages'] if m['role']=='user'),'')
        hint=next((m['content'] for m in row['messages'] if m['content'].startswith('[TRAINING SUPPORT] ')),'(none this turn)')
        if row['step']==0:
            self.recording.log(entity+'/task',rr.TextDocument(
                row['task_id']+'\n\n'+initial),static=True)
        text=(f"## {state['name']}\n\nTask: `{row['task_id']}`\n\n"
            f"Arm: **{row['arm']}** · rollout seed: {row['rollout_seed']} · env step: {row['step']}\n\n"
            f"Support active: **{row['support_active']}** · invalid JSON: {row['invalid_action']} · success: {row['success']}\n\n"
            f"### Model response\n```json\n{row['response']}\n```\n"
            f"### Executed action\n```text\n{row['action']}\n```\n"
            f"### Environment feedback\n{row['observation']}\n\n"
            f"### Visible support\n{hint}\n\n"
            f"Tokens: input {row['input_tokens']}, output {row['output_tokens']}\n\n"
            f"Replay entity: `{entity}/transition` on `env_step` or `ingest_event`.\n")
        self.recording.log(entity+'/transition',rr.TextDocument(text,media_type='text/markdown'))
        self.recording.log(entity+'/input_tokens',rr.Scalars(row['input_tokens']))
        self.recording.log(entity+'/output_tokens',rr.Scalars(row['output_tokens']))
        self.recording.log(entity+'/support_active',rr.Scalars(int(row['support_active'])))
        self.recording.log('events/'+state['name'],rr.TextLog(
            f"{key} t={row['step']} {row['action']} | {row['observation'][:180]}",
            level='WARN' if row['invalid_action'] else 'INFO'))
        self.latest=text;self.counts['steps']+=1

    def log_episode(self,state,row):
        self.time(optimizer_step=state['step'],episode=len(state['episodes']))
        won=sum(int(r['success']) for r in state['episodes'])
        entity='metrics/rolling/'+state['name']
        self.recording.log(entity+'/success_pct',self.rr.Scalars(100*won/len(state['episodes'])))
        self.recording.log(entity+'/completed',self.rr.Scalars(len(state['episodes'])))
        self.recording.log('events/'+state['name'],self.rr.TextLog(
            f"EPISODE complete success={row['success']} steps={row['steps']} task={row['task_id']} seed={row['rollout_seed']}"))
        self.counts['episodes']+=1

    def checkpoint(self,state):
        if state['step'] is None or state['score_emitted']:return
        result=final_scores(state['meta'],state['episodes'],state['manifest'])
        if result is None:return
        for role,score in result.items():
            self.time(optimizer_step=state['step'])
            entity=f"metrics/checkpoints/{state['stage']}/{state['case']}/{role}/success_pct"
            self.recording.log(entity,self.rr.SeriesLines(names=[state['case']+' '+role]),static=True)
            self.recording.log(entity,self.rr.Scalars(score['success_pct']))
            self.counts['checkpoint_series_points']+=1
        state['score_emitted']=True

    def training(self):
        paths=sorted((self.root/'logs').glob('*.log')) if (self.root/'logs').exists() else []
        for path in paths:
            if self.root.name=='learning-speed-v1' and path.name.endswith('-training.log'):
                phase='B';case=path.name.removesuffix('-training.log');cadence=8;maximum=128
            elif path.name.startswith(('A-','B-')):
                phase=path.name[0];case=path.stem[2:];cadence=10 if phase=='A' else 8;maximum=64 if phase=='A' else 128
            else:continue
            state=self.logs.setdefault(path,dict(cursor=LineCursor(path),points=0))
            for line in state['cursor'].read(maximum=2000):
                point=loss_point(line,state['points'],cadence,maximum)
                if point is None:continue
                state['points']+=1;self.time(optimizer_step=point['optimizer_step'])
                entity=f'training/{phase}/{safe_part(case)}'
                self.recording.log(entity+'/loss',self.rr.SeriesLines(names=[case]),static=True)
                for key in ['loss','learning_rate','grad_norm']:
                    if point[key] is not None:self.recording.log(entity+'/'+key,self.rr.Scalars(point[key]))
                self.counts['training_loss_points']+=1

    def poll(self):
        before=sum(self.counts.values());self.discover()
        for state in self.runs.values():
            for line in state['steps_cursor'].read():self.log_step(state,json.loads(line))
            for line in state['episode_cursor'].read():
                row=json.loads(line);state['episodes'].append(row);self.log_episode(state,row)
            # Scores require the complete planned episode set, regardless of trace import order.
            self.checkpoint(state)
        self.training()
        status={}
        for name in ['queue-state.json','main-progress.json','calibration-progress.json','collection-progress.json',
                'a-training-progress.json','adaptation-progress.json','teacher-generation.json']:
            path=self.root/name
            if path.exists():
                try:status[name]=json.loads(path.read_text())
                except (json.JSONDecodeError,FileNotFoundError):continue
        text='# Experiment progress\n\n```json\n'+json.dumps(status,indent=2)+'\n```\n'
        if text!=self.status_text:
            self.recording.log('live/status',self.rr.TextDocument(text,media_type='text/markdown'),static=True)
            self.status_text=text
        self.recording.log('live/transition',self.rr.TextDocument(self.latest,media_type='text/markdown'),static=True)
        self.recording.flush(timeout_sec=20)
        return sum(self.counts.values())-before

    def summary(self):
        return dict(study=self.root.as_posix(),recording=self.output.as_posix(),
            counts=self.counts.copy(),source_runs=len(self.runs),
            completed_checkpoint_evaluations=sum(s['score_emitted'] for s in self.runs.values()),
            timeline_semantics=dict(optimizer_step='Actual checkpoint update or known fixed Trainer log cadence',
                env_step='Per-attempt environment step; scope to one episode entity',
                ingest_event='Import order, not source wall-clock time'),
            source_files={str(p):dict(steps_offset=s['steps_cursor'].offset,episode_offset=s['episode_cursor'].offset)
                for p,s in self.runs.items()})

    def close(self):
        self.recording.flush(timeout_sec=30);self.recording.disconnect()
