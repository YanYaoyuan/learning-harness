#!/usr/bin/env python3
"""Same runner/RNG/budgets for cached API-scaffold and ReAct data collection."""
import argparse
import json
from pathlib import Path

from learning_harness.batch_runner import BatchClient
from learning_harness.benchmark import run
from learning_harness.common import digest,file_sha,read_jsonl
from learning_harness.support import SUPPORT_PREFIX


class ScaffoldClient(BatchClient):
    def __init__(self,config):
        super().__init__(config['model'])
        if file_sha(config['support_bank'])!=config['support_bank_sha256']:raise ValueError('Support bank changed')
        self.bank={r['public_reset_sha256']:r['hint'] for r in read_jsonl(config['support_bank'])}

    def complete_many(self,requests,max_tokens,temperature,revision):
        for r in requests:
            messages=r['messages'];key=digest(messages[1]['content'])
            if key not in self.bank:raise ValueError('Public reset has no pinned teacher card')
            if sum(m['role']=='assistant' for m in messages)<4:
                # Mutate in place so the benchmark logs exactly the served context.
                messages.insert(2,dict(role='user',content=SUPPORT_PREFIX+self.bank[key]))
        return super().complete_many(requests,max_tokens,temperature,revision)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    config=json.loads(Path(args.config).read_text());tasks=read_jsonl(config['manifest'])
    if any(t['partition']!='train' or t['official_split']!='train' for t in tasks):raise ValueError('Collection is train-only')
    if config['max_trials']!=1:raise ValueError('Collection uses one attempt per task/seed')
    if config['source']=='api':
        if config['method']!='json_direct':raise ValueError('API condition uses the original JSON actor')
        client=ScaffoldClient(config)
    elif config['source']=='react':
        if config['method']!='react_retry' or config.get('support_bank'):raise ValueError('Invalid ReAct condition')
        client=None
    else:raise ValueError('Unknown source')
    run(args.config,'data/alfworld',args.out,client=client)


if __name__=='__main__':main()
