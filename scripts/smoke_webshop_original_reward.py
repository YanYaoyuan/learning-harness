#!/usr/bin/env python3
"""Exercise original LG-reward environment mechanics on the existing synthetic preview."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/webshop-preparation/20261112/original-reference'
SOURCE = ROOT/'third_party/WebShop-2022-reference'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = OUT/'mechanics.json'
    if target.exists():
        raise ValueError('Preserve existing mechanics record before rerunning')
    reference = json.loads((OUT/'reference.json').read_text())
    if reference['status'] != 'reference_tests_passed':
        raise ValueError('Original reward checks not passed')
    preview = json.loads((OUT.parent/'smoke.json').read_text())
    for value in preview['data']:
        actual = Path(value['path'])
        if sha(actual) != value['sha256']:
            raise ValueError('Preview data changed')
        link = SOURCE/'data'/value['name']
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            raise ValueError('Existing reference data path requires inspection')
        link.symlink_to(actual)
    index = SOURCE/'search_engine/indexes_1k'
    index.symlink_to(ROOT/'third_party/WebShop/search_engine/indexes_1k', target_is_directory=True)
    code = [SOURCE/relative for relative in ('web_agent_site/engine/goal.py', 'web_agent_site/engine/engine.py',
                                           'web_agent_site/envs/web_agent_text_env.py')]
    before = {str(p.relative_to(SOURCE)):sha(p) for p in code}
    sys.path[:0] = [str(SOURCE), reference['nlp_pythonpath']]
    import numpy as np
    import torch
    from web_agent_site.engine.goal import nlp
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    if nlp.meta['name'] != 'core_web_lg' or torch.cuda.is_available():
        raise ValueError('Wrong reward profile or CPU runtime')
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    started = time.time()
    env = WebAgentTextEnv(observation_mode='text', num_products=1000, human_goals=False, get_image=False)
    observation, _ = env.reset(session=0)
    if not isinstance(observation, str) or not observation:
        raise ValueError('Missing initial public observation')
    steps = []
    action = 'search[shirt]'
    observation, reward, done, _ = env.step(action)
    steps.append(dict(action=action, observation_nonempty=bool(observation), reward=float(reward), done=done))
    items = [v for v in env.get_available_actions()['clickables'] if re.fullmatch('[a-z0-9]{10}', v)]
    if not items:
        raise ValueError('No publicly visible search results')
    for action in ('click['+items[0]+']', 'click[buy now]'):
        observation, reward, done, _ = env.step(action)
        if not observation or not math.isfinite(float(reward)):
            raise ValueError('Invalid mechanical transition')
        steps.append(dict(action=action, observation_nonempty=True, reward=float(reward), done=done))
    if not done or before != {str(p.relative_to(SOURCE)):sha(p) for p in code}:
        raise ValueError('No terminal transition or changed source')
    record = dict(status='original_reward_preview_mechanics_passed', started_unix=started,
                  ended_unix=time.time(), nlp_meta={k:nlp.meta[k] for k in ('lang','name','version')},
                  source_sha256=before, source_unchanged=True, products=len(env.server.all_products),
                  synthetic_goals=len(env.server.goals), steps=steps, model_calls=0, gpu_used=False,
                  human_goals_used=False, official_test_evaluations=0, full_benchmark_ready=False,
                  note='Predetermined public actions; reward is not an agent-performance result')
    target.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
