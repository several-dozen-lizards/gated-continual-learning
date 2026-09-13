import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from fixture import build_fixture
from evidence import route_stream
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'qwen35'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((PRIOR/'protocol_v2.json').read_text())
for name,digest in old['source_hashes'].items():assert sha(PRIOR/name)==digest
plan=dict(frozen_at=datetime.now(timezone.utc).isoformat(),seeds=old['seeds'],epochs=20,lr=.0002,
    model_revision=old['model_revision'],source_hashes={},fixtures={},prior_receipts={})
assert not (ROOT/'plan.json').exists()
(ROOT/'fixtures').mkdir(exist_ok=True)
for seed in plan['seeds']:
    run=PRIOR/'runs'/f'test-v2-seed-{seed}'
    world=json.loads((run/'world.json').read_text())
    evidence=build_fixture(world)
    dynamic=route_stream(evidence['claims'],evidence)
    naive=route_stream(evidence['claims'],evidence,collapse_lineage=False)
    assert len(dynamic['selected'])==3
    assert sum(evidence['evaluator'][c['id']]['useful'] for c in dynamic['selected'])==2
    assert len(naive['selected'])==7
    assert not route_stream(evidence['claims'],evidence,sham=True)['selected']
    target=ROOT/'fixtures'/f'{seed}.json'
    with target.open('x',encoding='utf-8') as f:json.dump(dict(world=world,evidence=evidence),f,indent=2)
    plan['fixtures'][str(seed)]=sha(target)
    files=['report.json','world.json','config.json','gated_adapter/adapter_config.json','gated_adapter/adapter_model.safetensors']
    plan['prior_receipts'][str(seed)]={name:sha(run/name) for name in files}
    print(seed,'dynamic',len(dynamic['selected']),'naive',len(naive['selected']))
for name in ['runner.py','evidence.py','fixture.py','PROTOCOL.md','prepare.py']:
    plan['source_hashes'][name]=sha(ROOT/name)
with (ROOT/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
