import json,re
from learning_harness.environment import AlfWorld
from learning_harness.common import read_jsonl,game_path
from learning_harness.support import canonicalize_action
from pathlib import Path

def norm(s):
 if s.startswith('You arrive at loc '):s=s[s.find('. ')+2:]
 return re.sub(r'\s+',' ',s).strip()

def main():
 task=next(t for t in read_jsonl('data/manifests/alfworld/dev.jsonl') if 'Lettuce-None-DiningTable-20/' in t['task_id'])
 demo=json.load(open('third_party/ReAct/prompts/alfworld_3prompts.json'))['react_clean_0']
 env=AlfWorld(game_path('data/alfworld',task),11,100)
 initial=env.reset();events=[]
 for piece in demo.split('\n> ')[1:]:
  action,_,expected=piece.partition('\n')
  if action.startswith('think:'):continue
  obs,won,done=env.step(canonicalize_action(action))
  events.append(dict(action=action,expected=expected.strip(),actual=obs,observation_matches=norm(expected)==norm(obs),won=won))
 env.close()
 out=dict(task_id=task['task_id'],demo_key='react_clean_0',same_public_room_and_goal=True,
          initial=initial,all_action_observations_match=all(e['observation_matches'] for e in events),
          demo_actions_solve_task=any(e['won'] for e in events),events=events)
 Path('output/baseline-benchmark-v1/demo-replay-audit.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['events','initial']},indent=2))
 print([(e['action'],e['observation_matches']) for e in events])
if __name__=='__main__':main()
