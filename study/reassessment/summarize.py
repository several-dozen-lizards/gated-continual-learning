"""Audit the frozen continuation and produce outcome tables from all seeds."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
for name,digest in plan['source_hashes'].items():assert sha(ROOT/name)==digest
records=[]
manifest={}
for seed in plan['seeds']:
    run=ROOT/'runs'/f'seed-{seed}'
    report=json.loads((run/'report.json').read_text())
    assert report['status']=='complete'
    assert report['prior_reload_delta']<=1e-5 and report['final_reload_delta']<=1e-5
    payload=json.loads((run/'inputs.json').read_text())
    fixture,world=payload['evidence'],payload['world']
    assert sha(ROOT/'fixtures'/f'{seed}.json')==plan['fixtures'][str(seed)]
    assert payload==json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
    for name,digest in plan['prior_receipts'][str(seed)].items():
        assert sha(ROOT.parent/'qwen35'/'runs'/f'test-v2-seed-{seed}'/name)==digest
    decisions=json.loads((run/'decisions.json').read_text())
    assert len(decisions['dynamic']['selected'])==3 and len(decisions['naive']['selected'])==7
    assert not decisions['sham']['selected']
    missed={c['name'] for c in fixture['claims'] if fixture['evaluator'][c['id']]['useful']}
    previous={c['name'] for c in world['stream'] if c['relevant'] and c['verified']}
    wrong=next(c for c in fixture['claims'] if fixture['evaluator'][c['id']]['condition']=='coordinated_false_agreement')
    for a,b in [('reassess','random'),('reassess_replay','random_replay')]:
        for key in ('steps','examples','input_tokens','supervised_tokens','padded_tokens'):
            assert report['arms'][a]['training'][key]==report['arms'][b]['training'][key]
    for arm,value in dict(baseline=dict(evaluation=report['baseline']),**report['arms']).items():
        evaluation=value['evaluation'];rows=evaluation['rows']
        assert len(rows)==24
        assert all(math.isfinite(p) for row in rows for p in row['probabilities'])
        assert all(abs(sum(row['probabilities'])-1)<1e-5 for row in rows)
        training=value.get('training',{})
        assert all(math.isfinite(x) for x in training.get('losses',[]))
        subset=lambda names:statistics.mean(r['correct'] for r in rows if r['name'] in names)
        records.append(dict(seed=seed,arm=arm,accuracy=evaluation['accuracy'],**evaluation['groups'],
            recovered=subset(missed),previous_updates=subset(previous),
            false_adoption=statistics.mean(r['prediction']==wrong['color'] for r in rows if r['name']==wrong['name']),
            training_seconds=training.get('seconds',0),input_tokens=training.get('input_tokens',0),steps=training.get('steps',0)))
    for file in run.rglob('*'):
        if file.is_file():manifest[str(file.relative_to(ROOT))]=sha(file)
arms=['baseline','reassess','random','reassess_replay','random_replay']
means={arm:{key:statistics.mean(r[key] for r in records if r['arm']==arm) for key in records[0] if key not in ('seed','arm')} for arm in arms}
(ROOT/'SUMMARY.json').write_text(json.dumps(dict(records=records,means=means),indent=2),encoding='utf-8')
(ROOT/'runs'/'receipt_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# Qwen3.5-2B: evidence reassessment and replay','',
    'Three paired continuations of saved gated adapters. Percentages are means across seeds 17, 29, 43. Recovery measures the two previously rejected useful claims. False adoption measures answers matching the deliberately corroborated wrong claim.','',
    '| Arm | Overall | Old retention | Recovered | Previous updates | False adoption | Train seconds | Tokens |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in arms:
    m=means[arm]
    lines.append('| '+arm+' | '+' | '.join(f'{100*m[k]:.1f}%' for k in ['accuracy','retention','recovered','previous_updates','false_adoption'])+f" | {m['training_seconds']:.1f} | {m['input_tokens']:.0f} |")
lines+=['','## Seed variation','','| Seed | Baseline | Reassess | Random | Reassess + replay | Random + replay |','|---|---:|---:|---:|---:|---:|']
for seed in plan['seeds']:
    lines.append('| '+str(seed)+' | '+' | '.join(f"{100*next(r['accuracy'] for r in records if r['seed']==seed and r['arm']==arm):.1f}%" for arm in arms)+' |')
lines+=['','## Scope and verification','',
    'Reassessment plus replay reached 91.7% in each seed, versus 69.4% mean for matched random replay and 70.8% at the starting checkpoints. Both previously missed useful facts were recovered in every reassessment arm. Without replay, reassessment averaged 55.6% and preserved only 50% of earlier accepted updates. With replay, every remaining error was the deliberately corroborated false claim. Replay supports retention of accepted material, including mistakes; it does not establish that the material is true.',
    '',
    'Timing anomaly: seed 17 random replay recorded 17,169.9 seconds, versus about 129–133 seconds for the other replay arms. The cause was not instrumented; suspension or scheduling delay is possible but unverified. Raw timings are preserved above and in the receipts. Do not interpret their means as comparative compute cost or a speed advantage. Matched token and step counts remain valid.',
    '',
    'The lineage-aware gate selected two useful claims and one false claim in every seed; the alias-counting control selected two useful and five false claims. The sham selected none. Source reliability and lineage are supplied synthetic evidence, not learned verification. Updates occur after the evidence batch; route demotion does not undo trained weights.',
    '', 'All prior and final adapter reloads passed the probability tolerance. Frozen sources, prior checkpoints and fixture hashes passed; losses and probabilities were finite. Training budgets match exactly within each random/reassessment pair. Replay uses a larger budget than selected-claim training. Training time excludes setup, routing, evaluation and serialization; these are separately recorded in reports. No energy or FLOP savings are established.',
    '', 'This is a small forced-choice fictional-world continuation, not a demonstration of general continual learning or resident readiness. See PROTOCOL.md and SUMMARY.json for the design and complete numerical results.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(means,indent=2))
