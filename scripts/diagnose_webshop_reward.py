#!/usr/bin/env python3
"""Record NLP-dependent upstream title-reward examples without changing the kernel."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'third_party/WebShop'
    target = root / 'output/webshop-preparation/20261112/reward-diagnostic.json'
    if target.exists():
        raise ValueError('Diagnostic already recorded')
    sys.path.insert(0, str(source))
    from web_agent_site.engine.goal import get_type_reward, nlp
    pairs = [
        ('Mens D.O.N. Issue 2 Gca Basketball Sneakers Shoes Casual - Off White',
         'PEAK High Top Mens Basketball Shoes Lou Williams Streetball Master Breathable Non Slip Outdoor Sneakers', .333),
        ('Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar Home Theater System Subwoofer',
         'BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System', .3),
        ('Saireed UL Listed 2 Prong Power Cord for JBL Bar 3.1 Bar 2.1 Channel 4K Ultra HD Soundbar',
         'BRST AC Power Cord Outlet Socket Cable Plug Lead for Panasonic SC-HT830V DVD/VCR Combo Home Theater System', .15),
        ('Rusticware 921ORB Kitchen and Bath Cabinet Knob',
         'Minkissy 2pcs Stainless Steel Eyebrow Tweezers Blackhead Acne Remover Portable Makeup Tweezers (Silver)', None),
    ]
    examples = []
    for desired, purchased, expected in pairs:
        goal = dict(query='Query 1', product_category='a › b › c', name=desired)
        item = dict(query='Query 2', product_category='a › a › d', name=purchased)
        examples.append(dict(desired=desired, purchased=purchased,
                             desired_pos=[(t.text, t.pos_) for t in nlp(desired)],
                             purchased_pos=[(t.text, t.pos_) for t in nlp(purchased)],
                             result=get_type_reward(item, goal),
                             upstream_expected_title_score=expected,
                             upstream_last_example_requires_below_005=expected is None))
    kernel = source / 'web_agent_site/engine/goal.py'
    record = dict(status='diagnostic_only', nlp_meta={k:nlp.meta[k] for k in ('lang','name','version')},
                  kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
                  examples=examples, official_test_evaluations=0, model_calls=0,
                  note='Handcrafted upstream unit examples; no kernel or expected-value changes')
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(nlp_meta=record['nlp_meta'], results=[e['result'] for e in examples])))


if __name__ == '__main__':
    main()
