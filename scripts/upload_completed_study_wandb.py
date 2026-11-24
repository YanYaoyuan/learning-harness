#!/usr/bin/env python3
"""Upload completed v2 results and figure exports to W&B."""
import csv
import hashlib
import json
from pathlib import Path

import wandb


def main():
    out=Path('output/wandb');target=json.loads((out/'connection.json').read_text())
    root=Path('output/figures/harness-v2')
    settings=wandb.Settings(console='off',disable_code=True,disable_git=True,save_code=False,
        disable_job_creation=True,x_disable_stats=True,init_timeout=60)
    run=wandb.init(entity=target['entity'],project=target['project'],
        id=hashlib.sha256(b'harness-v2-completed-figure-backfill').hexdigest()[:12],
        name='dual5090-v2 / completed controls and figures',group='dual5090-v2',job_type='analysis',
        mode='online',resume='allow',dir=str(out.absolute()),settings=settings,save_code=False,
        config=dict(study='dual5090-v2',model='Qwen3-4B-Instruct-2507',trained_checkpoints=18,
            training_seeds=[101,202,303],primary_development_tasks=28,final_benchmark_used=False,
            interpretation='Exploratory matched-data controls; no stable inverse ranking or RSI demonstrated'))
    try:
        for name in ['seed_scores','effects','task_type_scores','training_loss']:
            with (root/(name+'.csv')).open() as f:
                reader=csv.DictReader(f);columns=reader.fieldnames;data=[]
                for record in reader:
                    row=[]
                    for c in columns:
                        value=record[c]
                        if value=='':value=None
                        else:
                            try:value=int(value)
                            except (ValueError,TypeError):
                                try:value=float(value)
                                except (ValueError,TypeError):pass
                        row.append(value)
                    data.append(row)
            run.log({'tables/'+name:wandb.Table(columns=columns,data=data)})
        images=sorted(root.glob('*.png'))
        assert len(images)==6
        for image in images:run.log({'figures/'+image.stem:wandb.Image(str(image))})
        artifact=wandb.Artifact('harness-v2-measured-figures',type='research-figures')
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in ['.csv','.json','.png','.svg','.pdf']:
                artifact.add_file(str(path),name=path.name)
        run.log_artifact(artifact)
        run.summary.update({'measured_study_complete':True,'figure_sets':6,'trained_checkpoints':18})
        (out/'historical-backfill.json').write_text(json.dumps(dict(run_url=run.url,figures=6,
            tables=4,checkpoint_rows=18,source=str(root)),indent=2)+'\n')
    finally:run.finish();wandb.teardown()


if __name__=='__main__':main()
