#!/usr/bin/env python3
"""Render figures from an exported measured-data file."""
import argparse
import json
from pathlib import Path
import plot_study as p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    d=json.loads(args.data.read_text());assert d['measured_only']
    p.OUT=args.out;p.OUT.mkdir(parents=True,exist_ok=True)
    p.plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
    p.paired_figure(d['result']);p.effects_figure(d['result']);p.collection_figure(d['main'],d['result'])
    p.task_figure(d['type_records']);p.reward_figure(d['reward_audit']);p.optimization_figure(d['training_records'])
    p.export_csv('seed_scores.csv',d['score_records']);p.export_csv('task_type_scores.csv',d['type_records']);p.export_csv('effects.csv',d['effect_records'])
    print(args.out)


if __name__=='__main__':
    main()
