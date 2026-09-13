import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'drift_replay'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((PRIOR/'plan.json').read_text())
for name,digest in old['source_hashes'].items():assert sha(PRIOR/name)==digest,name
assert not (ROOT/'plan.json').exists()
(ROOT/'fixtures').mkdir(exist_ok=True)
plan=dict(frozen_at=datetime.now(timezone.utc).isoformat(),seeds=old['seeds'],model_revision=old['model_revision'],source_hashes={},fixtures={})
for seed in plan['seeds']:
    run=PRIOR/'runs'/f'seed-{seed}'
    assert json.loads((run/'report.json').read_text())['status']=='complete'
    p=json.loads((run/'inputs.json').read_text())
    paths=dict(baseline=ROOT.parent/'buffer_stability'/'runs'/f'seed-{seed}'/'small_wave2_adapter',
        drift=run/'drift_adapter',random=run/'random_adapter')
    checkpoints={name:dict(path=str(path.resolve()),hashes={f:sha(path/f) for f in ('adapter_config.json','adapter_model.safetensors')}) for name,path in paths.items()}
    payload=dict(world=p['world'],target_name=p['target_name'],old_wrong_color=p['old_wrong_color'],ledger=p['candidates']+p['latest'],
        checkpoints=checkpoints,prior_report_path=str((run/'report.json').resolve()),prior_report_hash=sha(run/'report.json'))
    assert len({c['name'] for c in payload['ledger']})==20
    target=ROOT/'fixtures'/f'{seed}.json'
    with target.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    plan['fixtures'][str(seed)]=sha(target)
for file in ['runner.py','gate.py','PROTOCOL.md','prepare.py']:plan['source_hashes'][file]=sha(ROOT/file)
with (ROOT/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
print('Frozen promotion trial',plan['seeds'])
