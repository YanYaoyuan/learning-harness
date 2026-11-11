"""Budgeted ALFWorld baseline ports. Original experiment runners stay unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import uuid

from .batch_runner import BatchClient
from .common import digest, file_sha, game_path, read_jsonl
from .support import SYSTEM_PROMPT, canonicalize_action, messages_for, parse_action

METHODS = ('json_direct', 'act', 'react_retry', 'reflexion')
DEMOS = Path('third_party/ReAct/prompts/alfworld_3prompts.json')
REFLECTION_DEMOS = Path('third_party/Reflexion/alfworld_runs/reflexion_few_shot_examples.txt')
SOURCE_FILES = [str(Path(__file__).relative_to(Path.cwd())) if Path(__file__).is_absolute() else __file__,
    'src/learning_harness/batch_runner.py', 'src/learning_harness/api.py',
    'src/learning_harness/support.py', 'src/learning_harness/environment.py',
    'src/learning_harness/remote_environment.py', 'src/learning_harness/common.py',
    'scripts/hf_batch_server.py', str(DEMOS), str(REFLECTION_DEMOS)]


def public_family(initial):
    """Route demonstrations from the visible instruction, never task metadata."""
    if 'Your task is to:' not in initial:
        raise ValueError('Missing public task instruction')
    goal = initial.split('Your task is to:')[-1].lower()
    if re.search(r'\b(two|2)\b', goal): return 'puttwo'
    if re.search(r'\b(clean|cleaned|wash|washed)\b', goal): return 'clean'
    if re.search(r'\b(cool|cooled|cold)\b', goal): return 'cool'
    if re.search(r'\b(heat|heated|hot)\b', goal): return 'heat'
    if re.search(r'\b(look|examine)\b', goal): return 'examine'
    if re.search(r'\b(put|place)\b', goal): return 'put'
    raise ValueError('Unrecognized public goal; do not silently select demonstrations')


def demonstration(initial, method):
    family = public_family(initial)
    examples = json.loads(DEMOS.read_text())
    # Act uses exactly the same trajectories with only thought/OK pairs removed.
    text = examples[f'react_{family}_1'] + examples[f'react_{family}_0']
    if method == 'act':
        text = re.sub(r'^> think:.*\n(?:OK\.\n)?', '', text, flags=re.MULTILINE)
    return text


def actor_messages(initial, history, memories, method, history_turns):
    if method == 'json_direct':
        return messages_for(initial, history, history_turns=history_turns)
    instruction = ('Interact with a household to solve the task using only observations. '
        'Return one line only: one environment command, without quotation marks or Markdown. '
        'Use the observed object names and numbers. ')
    if method != 'act':
        instruction += 'You may instead return one line starting with think: to reason. '
    instruction += ('A command or think: consumes one turn. Here are two examples.\n' +
                    demonstration(initial, method))
    messages = [dict(role='system', content=instruction), dict(role='user', content=initial)]
    if memories:
        messages.append(dict(role='user', content='Plans from previous failed attempts on this task:\n' +
                             '\n'.join(memories[-3:])))
    for event in history[-history_turns:]:
        messages.extend([dict(role='assistant', content=event['action']),
                         dict(role='user', content=event['observation'])])
    return messages


def reflection_messages(initial, history, memories):
    # The original instruction and examples, limited to this task's public evidence.
    query = ('You will be given the history of a past experience in which you were placed '
        'in an environment and given a task to complete. You were unsuccessful in completing '
        'the task. Do not summarize your environment, but rather think about the strategy '
        'and path you took to attempt to complete the task. Devise a concise, new plan of '
        'action that accounts for your mistake with reference to specific actions that you '
        'should have taken. You will need this later when you are solving the same task. '
        'Give your plan after "Plan". Here are two examples:\n\n' + REFLECTION_DEMOS.read_text())
    transcript = initial + '\n' + '\n'.join('> ' + e['action'] + '\n' + e['observation'] for e in history)
    if memories: transcript += '\nPlans from past attempts:\n' + '\n'.join(memories[-3:])
    return [dict(role='system', content=query), dict(role='user', content=transcript + '\nNew plan:')]


def parse_line(text, method):
    if method == 'json_direct':
        action, invalid = parse_action(text)
        return canonicalize_action(action), invalid, False
    line = text.strip().split('\n')[0].strip().removeprefix('> ').strip()
    thought = line.startswith('think:') and method != 'act'
    invalid = not line or (len(line) > 256 and not thought) or (line.startswith('think:') and not thought)
    return canonicalize_action(line) if not invalid else '__invalid_action__', invalid, thought


def validate(config, tasks, freeze=None):
    if config['method'] not in METHODS: raise ValueError('Unknown method')
    for key in ['max_turns', 'max_calls', 'max_trials', 'history_turns', 'batch_size', 'max_new_tokens']:
        if type(config[key]) is not int or config[key] < 1: raise ValueError('Invalid budget: ' + key)
    if config['max_new_tokens'] > 128 or config['batch_size'] > 8: raise ValueError('Server limits exceeded')
    if not tasks or len({t['task_id'] for t in tasks}) != len(tasks): raise ValueError('Empty/duplicate tasks')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']): raise ValueError('Duplicate/empty seeds')
    final = any(t.get('purpose') == 'final_evaluation' or t.get('official_split') in ('valid_seen', 'valid_unseen') for t in tasks)
    if final:
        if freeze is None: raise ValueError('Final evaluation requires a frozen registered config')
        lock = json.loads(Path(freeze).read_text())
        if lock.get('status') != 'frozen' or digest(config) not in lock.get('config_digests', []):
            raise ValueError('Config absent from final freeze')
        required = set(SOURCE_FILES + [config['manifest']])
        if not required <= set(lock.get('files', {})): raise ValueError('Incomplete final input lock')
        for path, expected in lock['files'].items():
            if file_sha(path) != expected: raise ValueError('Final input changed: ' + path)


def run(config_path, data_root, output, freeze=None, client=None, environment_factory=None):
    if environment_factory is None:
        from .remote_environment import RemoteAlfWorld
        environment_factory = RemoteAlfWorld
    config = json.loads(Path(config_path).read_text()); tasks = read_jsonl(config['manifest'])
    validate(config, tasks, freeze)
    client = client or BatchClient(config['model'])
    revision = os.environ.get('LH_STUDENT_REVISION', 'UNPINNED')
    if revision == 'UNPINNED' or revision != config['revision']: raise ValueError('Checkpoint mismatch')
    method = config['method']; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inputs = {p: file_sha(p) for p in SOURCE_FILES + [config['manifest']]}
    fingerprint = digest(dict(config=config, inputs=inputs, model=client.model, revision=revision))
    meta = output / 'run.json'
    if meta.exists():
        if json.loads(meta.read_text())['fingerprint'] != fingerprint: raise ValueError('Run fingerprint changed')
    else:
        meta.write_text(json.dumps(dict(fingerprint=fingerprint, config=config, inputs=inputs,
            planned_tasks=len(tasks)*len(config['seeds']), model=client.model, revision=revision), indent=2)+'\n')
    recorded = read_jsonl(output/'tasks.jsonl') if (output/'tasks.jsonl').exists() else []
    done_keys = {(r['task_id'], r['seed']) for r in recorded}
    if len(done_keys) != len(recorded): raise ValueError('Duplicate completed tasks')
    jobs = iter((t,s) for t in tasks for s in config['seeds'] if (t['task_id'],s) not in done_keys)
    slots = [None] * config['batch_size']
    parallel = getattr(environment_factory, 'supports_async_step', False)
    def reset(state):
        state['env'] = environment_factory(game_path(data_root, state['task']), state['seed'], config['max_turns'])
        initial = state['env'].reset()
        if state.get('initial') is not None and state['initial'] != initial: raise ValueError('Retry reset changed task')
        state.update(initial=initial, history=[], phase='act', turn=0)
    with (output/'events.jsonl').open('a') as events, (output/'tasks.jsonl').open('a') as results:
        try:
            while True:
                for i in range(len(slots)):
                    if slots[i] is not None: continue
                    job = next(jobs, None)
                    if job is None: continue
                    task, seed = job
                    state = dict(task=task, seed=seed, trial=1, memories=[], trials=[], calls=0,
                        input_tokens=0, output_tokens=0, env=None, initial=None, run_id=uuid.uuid4().hex)
                    slots[i] = state; reset(state)
                active = [s for s in slots if s is not None]
                if not active: break
                requests = []
                for s in active:
                    if s['phase'] == 'reflect':
                        messages = reflection_messages(s['initial'], s['history'][-config['history_turns']:], s['memories'])
                    else:
                        messages = actor_messages(s['initial'], s['history'], s['memories'], method, config['history_turns'])
                    # Pair first attempts across ReAct and Reflexion; diversify retry sampling.
                    seed = int(digest([s['task']['task_id'],s['seed'],s['trial'],s['turn'],s['phase']])[:8],16)
                    requests.append(dict(messages=messages,seed=seed))
                responses = client.complete_many(requests, config['max_new_tokens'], config['temperature'], revision)['results']
                parsed = []
                for s, response in zip(active, responses):
                    p = parse_line(response['text'], method) if s['phase'] == 'act' else None
                    parsed.append(p)
                    if p and parallel: s['env'].step_async(p[0])
                for s, response, request, parsed_action in zip(active, responses, requests, parsed):
                    phase = s['phase']; s['calls'] += 1
                    s['input_tokens'] += response['input_tokens']; s['output_tokens'] += response['output_tokens']
                    event = dict(run_id=s['run_id'],task_id=s['task']['task_id'],seed=s['seed'],method=method,
                        trial=s['trial'],turn=s['turn'],phase=phase,call=s['calls'],messages=request['messages'],
                        request_seed=request['seed'],response=response['text'],input_tokens=response['input_tokens'],
                        output_tokens=response['output_tokens'])
                    if phase == 'reflect':
                        s['memories'].append(response['text']); reset(s)
                        events.write(json.dumps(event,ensure_ascii=False)+'\n');events.flush()
                        continue
                    action, invalid, thought = parsed_action
                    observation, won, terminal = s['env'].step_wait() if parallel else s['env'].step(action)
                    if thought: observation = 'OK.'
                    s['turn'] += 1
                    event.update(action=action,invalid_format=invalid,thought=thought,observation=observation,success=won)
                    events.write(json.dumps(event,ensure_ascii=False)+'\n');events.flush()
                    s['history'].append(event)
                    if not (won or terminal or s['turn'] >= config['max_turns'] or s['calls'] >= config['max_calls']): continue
                    s['trials'].append(dict(trial=s['trial'],success=bool(won),turns=s['turn'],
                        cumulative_calls=s['calls'],cumulative_input_tokens=s['input_tokens'],cumulative_output_tokens=s['output_tokens']))
                    s['env'].close();s['env']=None
                    # A reflection must leave at least one call to act on its result.
                    remaining = config['max_calls']-s['calls']
                    if won or s['trial'] >= config['max_trials'] or remaining < (2 if method == 'reflexion' else 1):
                        record = dict(run_id=s['run_id'], task_id=s['task']['task_id'],task_group=s['task']['task_group'],
                            task_type=s['task']['task_type'],seed=s['seed'],method=method,success=bool(won),
                            first_success=s['trials'][0]['success'],trials=s['trials'],calls=s['calls'],
                            input_tokens=s['input_tokens'],output_tokens=s['output_tokens'])
                        results.write(json.dumps(record)+'\n');results.flush(); slots[slots.index(s)]=None
                        print(json.dumps(dict(method=method,task=s['task']['task_id'],success=bool(won),calls=s['calls'])),flush=True)
                    else:
                        s['trial'] += 1
                        if method == 'reflexion': s['phase']='reflect'
                        else: reset(s)
        finally:
            for s in slots:
                if s and s['env'] is not None: s['env'].close()
    # Interrupted traces are retained, but only completed run IDs count in analysis.
    return read_jsonl(output/'tasks.jsonl')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--data-root',default='data/alfworld');parser.add_argument('--freeze')
    args=parser.parse_args();run(args.config,args.data_root,args.out,args.freeze)


if __name__ == '__main__': main()
