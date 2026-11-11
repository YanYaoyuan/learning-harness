#!/usr/bin/env python3
"""Backfill and optionally follow real experiment logs into Rerun recordings."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

os.environ.setdefault('RAYON_NUM_THREADS','2')
os.environ.setdefault('RERUN_ANALYTICS_ENABLED','false')
from learning_harness.rerun_recording import StudyRecorder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('studies',nargs='*',type=Path,default=[Path('runs/learning-speed-v1'),Path('runs/api-harness-v1')])
    p.add_argument('--out',type=Path,default=Path('output/rerun'))
    p.add_argument('--watch',action='store_true');p.add_argument('--serve',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=5)
    p.add_argument('--web-port',type=int,default=9090);p.add_argument('--grpc-port',type=int,default=19879)
    args=p.parse_args()
    if args.serve and not args.watch:p.error('--serve requires --watch')
    if args.poll_seconds<1:p.error('--poll-seconds must be at least 1')
    if len({s.name for s in args.studies})!=len(args.studies):p.error('Study names must be unique')
    args.out.mkdir(parents=True,exist_ok=True)
    outputs=[args.out/(s.name+'.rrd') for s in args.studies]
    if any(p.exists() for p in outputs):raise ValueError('Recording exists; choose a fresh output directory to rebuild')
    if any(not s.is_dir() for s in args.studies):raise ValueError('Study directory missing')
    viewer=None;recorders=[];stop=False
    def terminate(*unused):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        url=None
        if args.serve:
            for port in [args.grpc_port,args.web_port]:
                with socket.socket() as test:test.bind(('127.0.0.1',port))
            executable=Path(os.sys.executable).parent/'rerun'
            with (args.out/'viewer.log').open('w') as log:
                viewer=subprocess.Popen([str(executable),'--serve-web','--bind','127.0.0.1',
                    '--port',str(args.grpc_port),'--web-viewer-port',str(args.web_port),
                    '--server-memory-limit','512MiB'],stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+30
            while True:
                if viewer.poll() is not None:raise RuntimeError('Rerun viewer server exited; inspect viewer.log')
                try:
                    with socket.create_connection(('127.0.0.1',args.grpc_port),timeout=1):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('Rerun viewer startup')
                    time.sleep(.25)
            url=f'rerun+http://127.0.0.1:{args.grpc_port}/proxy'
        for root,output in zip(args.studies,outputs):recorders.append(StudyRecorder(root,output,url))
        print(json.dumps(dict(status='recording',pid=os.getpid(),viewer=f'http://127.0.0.1:{args.web_port}' if viewer else None,
            grpc=url,recordings=[str(p) for p in outputs])),flush=True)
        while not stop:
            changed=sum(r.poll() for r in recorders)
            state=dict(status='watching' if args.watch else 'backfilling',pid=os.getpid(),
                updated_at_unix=time.time(),viewer=f'http://127.0.0.1:{args.web_port}' if viewer else None,
                grpc=url,studies=[r.summary() for r in recorders])
            temp=args.out/'recording-summary.json.part';temp.write_text(json.dumps(state,indent=2)+'\n')
            temp.replace(args.out/'recording-summary.json')
            if not args.watch and not changed:break
            if args.watch:
                if viewer and viewer.poll() is not None:raise RuntimeError('Rerun viewer exited')
                # Short sleep permits graceful signal shutdown and never blocks an agent tool call.
                for _ in range(max(1,int(args.poll_seconds*5))):
                    if stop:break
                    time.sleep(.2)
        state['status']='stopped' if args.watch else 'complete'
        (args.out/'recording-summary.json').write_text(json.dumps(state,indent=2)+'\n')
    finally:
        for recorder in recorders:recorder.close()
        if viewer is not None:
            viewer.terminate()
            try:viewer.wait(timeout=10)
            except subprocess.TimeoutExpired:viewer.kill();viewer.wait()


if __name__=='__main__':main()
