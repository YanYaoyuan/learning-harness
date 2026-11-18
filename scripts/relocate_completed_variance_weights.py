#!/usr/bin/env python3
"""Relocate only completed studies' weight files; retain raw JSON in the working tree."""
import json
from pathlib import Path
import shutil
import time

import relocate_large_files as migration

DESTINATION=Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/learning-harness-lab')
RECEIPT=Path('output/research-operations/20261112/completed-variance-storage.json')


def main():
    if RECEIPT.exists():
        old=json.loads(RECEIPT.read_text())
        if old['status']=='completed':
            print('Completed migration retained; no duplicate work')
            return
        raise ValueError('An existing incomplete migration needs inspection')
    roots=[migration.WORK/'runs/variance-diagnostic-v1'/name for name in ('stage1','stage2')]
    for root in roots:
        if json.loads((root/'execution.json').read_text())['status']!='completed':raise ValueError('Study still active')
        if json.loads((migration.WORK/'output'/root.relative_to(migration.WORK/'runs')/'archival.json').read_text())['status']!='uploaded':
            raise ValueError('Raw records have not been archived')
    candidates=sorted(p for root in roots for p in root.rglob('*.safetensors') if p.is_file() and not p.is_symlink())
    total=sum(p.stat().st_size for p in candidates)
    if shutil.disk_usage(DESTINATION).free<total+1024**3:raise ValueError('Target disk capacity insufficient')
    value=dict(status='running',files=[],planned_files=len(candidates),planned_bytes=total,
        started_unix=time.time(),json_records_retained_locally=True,active_stage3_excluded=True)
    def save():
        temporary=RECEIPT.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(RECEIPT)
    save()
    try:
        for source in candidates:
            record=migration.relocate(source,DESTINATION/source.relative_to(migration.WORK))
            value['files'].append(record);save()
        value.update(status='completed',finished_unix=time.time(),verified_bytes=sum(v['bytes'] for v in value['files']),
                     working_disk_available_bytes=shutil.disk_usage('.').free)
        save();print(json.dumps({k:v for k,v in value.items() if k!='files'}),flush=True)
    except BaseException as error:
        value.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
