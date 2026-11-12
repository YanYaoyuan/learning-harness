#!/usr/bin/env python3
"""Archive a completed bounded study without editing any older frozen uploader."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from upload_research_results import package
from learning_harness.common import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root,out=Path(args.root),Path(args.out)
    frozen=json.loads((root/'freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_sha(name)!=sha:raise ValueError('Frozen archival input changed: '+name)
    report=json.loads((out/'summary.json').read_text())
    if not report['complete'] or report['cases']!=len(frozen['cases']) or report['actual_evaluation_episodes']!=frozen['planned_episodes']:
        raise ValueError('Refusing to archive an incomplete study')
    archive,manifest=package(root,out)
    target=json.loads(Path('output/wandb/connection.json').read_text())
    host=target.get('base_url',target.get('host'))
    if host:os.environ['WANDB_BASE_URL']=host
    import wandb
    identity=file_sha(root/'freeze.json')[:12]
    run=wandb.init(entity=target['entity'],project=target['project'],id=identity,name=frozen['name'],
        group='variance-diagnostic-v1',job_type='measured-results-archive',mode='online',resume='allow',
        dir=str(out.absolute()),save_code=False,settings=wandb.Settings(console='off',disable_code=True,
            disable_git=True,save_code=False,disable_job_creation=True,x_disable_stats=True,init_timeout=60))
    try:
        for path in sorted(out.glob('*.png')):run.log({'figures/'+path.stem:wandb.Image(str(path))})
        run.summary.update(dict(complete=True,cases=report['cases'],evaluation_episodes=report['actual_evaluation_episodes'],
            primary_policy_effects=report.get('policy_effects'),teacher_calls=0,official_tests_used=False,
            archived_json_files=len(manifest)))
        artifact=wandb.Artifact('variance-diagnostic-v1-'+root.name+'-raw-records',type='research-results')
        artifact.add_file(str(archive));artifact.add_file(str(out/'archive-manifest.json'))
        for path in sorted(out.iterdir()):
            if path.is_file() and (path.suffix in ('.png','.pdf','.csv','.md') or path.name=='summary.json'):
                artifact.add_file(str(path))
        run.log_artifact(artifact).wait()
        receipt=dict(status='uploaded',run_id=run.id,run_url=run.url,archived_json_files=len(manifest),
            archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=file_sha(archive))
        (out/'archival.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    finally:run.finish()


if __name__=='__main__':main()
