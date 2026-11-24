#!/usr/bin/env python3
"""Plot the composition of A-stage training data."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from api_harness_report import save,csv_out,plt
from learning_harness.common import read_jsonl

ROOT=Path('runs/transfer-factorial-v1')


def main():
    index={r['task_id']:r for r in read_jsonl('runs/api-harness-v1/train.jsonl')}
    records=[];counts={}
    for regime in ['matched','full']:
        for source in ['generic','api']:
            path=(Path('runs/api-harness-v1/corpora')/f'matched-{source}.jsonl' if regime=='matched'
                  else ROOT/'corpora'/f'full-{source}.jsonl')
            rows=read_jsonl(path);tasks={r['task_id'] for r in rows}
            counts[regime,source]=Counter(index[t]['task_type'] for t in tasks)
            records.append(dict(regime=regime,source=source,eligible_examples=len(rows),
                task_groups=len({index[t]['task_group'] for t in tasks}),fixed_processed_examples=256))
    fig,axes=plt.subplots(1,3,figsize=(12,4.1));colors=['#4477AA','#CC6677']
    for ax,metric,label in [(axes[0],'task_groups','Successful task groups'),(axes[1],'eligible_examples','Eligible action examples')]:
        for i,source in enumerate(['generic','api']):
            vals=[r[metric] for r in records if r['source']==source];x=np.arange(2)+(i-.5)*.3
            ax.bar(x,vals,width=.28,color=colors[i],label=source)
            for xx,val in zip(x,vals):ax.text(xx,val+max(vals)*.025,str(val),ha='center',fontsize=9)
        ax.set_xticks([0,1],['Matched','Full']);ax.set_ylabel(label);ax.margins(y=.16)
    families=['pick_and_place_simple','pick_clean_then_place_in_recep','pick_heat_then_place_in_recep']
    for i,source in enumerate(['generic','api']):
        axes[2].bar(np.arange(3)+(i-.5)*.3,[counts['full',source][f] for f in families],width=.28,color=colors[i],label=source)
    axes[2].set_xticks([0,1,2],['Place','Clean','Heat']);axes[2].set_ylabel('Tasks represented in full corpus');axes[2].legend(frameon=False)
    fig.subplots_adjust(bottom=.22,wspace=.4)
    fig.text(.5,.02,'Observed A corpora. Every A model processes 256 examples; pool size and sampling repetition differ. No new B outcomes shown.',ha='center',fontsize=8)
    out=ROOT/'figures';out.mkdir(exist_ok=True);save(fig,out/'source_coverage');csv_out(out/'source-coverage.csv',records)
    (out/'source-coverage.json').write_text(json.dumps(records,indent=2)+'\n')
    for family in ['cool','heat']:
        directory=ROOT/family/'figures/design';directory.mkdir(parents=True,exist_ok=True)
        link=directory/'source_coverage.png'
        if not link.exists():link.symlink_to((out/'source_coverage.png').absolute())


if __name__=='__main__':main()
