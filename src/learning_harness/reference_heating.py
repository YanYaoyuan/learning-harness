"""Heating teaching policy using public text only; no hidden planner state."""
import re
from .reference_cooling import observed_names


def heating_plan(initial):
    if 'Your task is to:' not in initial:
        raise ValueError('Missing public task instruction')
    overview,goal=initial.lower().split('your task is to:',1)
    target=destination=None
    for pattern in [r'heat (?:some|a|an) (\w+) and put it (?:in|on) (\w+)',
                    r'put (?:some|a|an) (?:hot|heated) (\w+) (?:in|on) (\w+)',
                    r'put (?:some|a|an) (\w+) (?:in|on) (\w+) after heating']:
        match=re.search(pattern,goal.splitlines()[0])
        if match:
            target,destination=match.groups();break
    if target is None:raise ValueError('Unsupported public heating instruction')
    places=observed_names(overview)
    microwaves=[r for r in places if r.rsplit(' ',1)[0]=='microwave']
    destinations=[r for r in places if r.rsplit(' ',1)[0]==destination]
    if not microwaves or not destinations:raise ValueError('Required receptacle absent from public overview')
    priorities={r:i for i,r in enumerate(['countertop','diningtable','stoveburner','sinkbasin',
                                        'microwave','fridge','cabinet','drawer','shelf','garbagecan'])}
    carrying=None
    for place in sorted(places,key=lambda r:(priorities.get(r.rsplit(' ',1)[0],100),int(r.rsplit(' ',1)[1]))):
        feedback=yield f'go to {place}'
        if 'is closed' in feedback.lower():feedback=yield f'open {place}'
        for obj in [r for r in observed_names(feedback) if r.rsplit(' ',1)[0]==target]:
            feedback=yield f'take {obj} from {place}'
            if 'pick up' in feedback.lower():carrying=obj;break
        if carrying:break
    if carrying is None:return
    microwave=microwaves[0]
    feedback=yield f'go to {microwave}'
    if 'is closed' in feedback.lower():yield f'open {microwave}'
    feedback=yield f'heat {carrying} with {microwave}'
    if feedback.strip()=='Nothing happens.':
        yield f'close {microwave}'
        yield f'heat {carrying} with {microwave}'
    for i,place in enumerate(destinations):
        feedback=yield f'go to {place}'
        if 'is closed' in feedback.lower():yield f'open {place}'
        yield f'move {carrying} to {place}'
        if i+1<len(destinations):
            feedback=yield f'take {carrying} from {place}'
            if 'pick up' not in feedback.lower():return
