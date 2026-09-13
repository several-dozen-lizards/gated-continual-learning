"""Audit all frozen repair trials and summarize behavioral correction."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
for name,digest in plan['source_hashes'].items():assert sha(ROOT/name)==digest,name
records=[];manifest={};deltas=[]
for seed in plan['seeds']:
    run=ROOT/'runs'/f'seed-{seed}'
    report=json.loads((run/'report.json').read_text())
    assert report['status']=='complete'
    for key in ('prior_reload_delta','final_reload_delta','transfer_reload_delta'):
        assert report[key]<=1e-5;deltas.append(report[key])
    p=json.loads((run/'inputs.json').read_text())
    assert p==json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
    assert sha(ROOT/'fixtures'/f'{seed}.json')==plan['fixtures'][str(seed)]
    assert json.loads((run/'plan.json').read_text())==plan
    for file,digest in plan['prior_receipts'][str(seed)].items():
        assert sha(ROOT.parent/'reassessment'/'runs'/f'seed-{seed}'/file)==digest,file
    a,b=report['arms']['corrected_replay']['training'],report['arms']['unchanged_replay']['training']
    for key in ('steps','examples','input_tokens','supervised_tokens','padded_tokens'):assert a[key]==b[key],key
    conditions=dict(baseline=dict(evaluation=report['baseline'],transfer=report['baseline_transfer']),**report['arms'])
    for arm,value in conditions.items():
        training=value.get('training',{})
        assert all(math.isfinite(x) for x in training.get('losses',[]))
        for family in ('evaluation','transfer'):
            rows=value[family]['rows']
            assert len(rows)==(24 if family=='evaluation' else 48)
            assert all(math.isfinite(x) for row in rows for x in row['probabilities'])
            assert all(abs(sum(row['probabilities'])-1)<1e-5 for row in rows)
            target=[r for r in rows if r['name']==p['target_name']]
            other=[r for r in rows if r['name']!=p['target_name']]
            records.append(dict(seed=seed,arm=arm,family=family,accuracy=statistics.mean(r['correct'] for r in rows),
                target_accuracy=statistics.mean(r['correct'] for r in target),
                old_error=statistics.mean(r['prediction']==p['old_wrong_color'] for r in target),
                other_accuracy=statistics.mean(r['correct'] for r in other),
                input_tokens=training.get('input_tokens',0),steps=training.get('steps',0),
                seconds=training.get('seconds',0),cpu_seconds=training.get('cpu_seconds',0),cuda_interval_seconds=training.get('cuda_interval_seconds',0)))
    for file in run.rglob('*'):
        if file.is_file():manifest[str(file.relative_to(ROOT))]=sha(file)
arms=['baseline','correction_only','corrected_replay','unchanged_replay']
means={family:{arm:{k:statistics.mean(r[k] for r in records if r['arm']==arm and r['family']==family)
    for k in records[0] if k not in ('seed','arm','family')} for arm in arms} for family in ('evaluation','transfer')}
summary=dict(records=records,means=means,max_reload_delta=max(deltas),receipt_count=len(manifest))
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'runs'/'receipt_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# Qwen3.5-2B: repairing a learned false admission','',
    'Three paired continuations (seeds 17, 29, 43) of the earlier mistaken replay checkpoints. All rates below are means across seeds. No retrieval was used.','',
    'Corrected-ledger replay repaired the target and preserved all eleven other facts in every seed, on both original and additional phrasings. Unchanged-ledger replay preserved the mistake despite an identical training budget. Correction-only also repaired the target in every seed, but retained only 54.5% of the other facts on average (58.3% overall). Thus the supplied correction changed learned behavior, while replay prevented the collateral losses observed under this particular correction-only schedule.',
    '', 'This clean repair used the full twelve-fact ledger: 9,520 input tokens and 120 optimizer steps, versus 820 tokens and 20 steps for correction-only. It does not establish that full replay is necessary or efficient at scale. A smaller replay budget and subsequent interference stream remain useful next tests.','']
for family,title in [('evaluation','Original held-out phrasings'),('transfer','Four additional phrasings')]:
    lines += ['## '+title,'','| Arm | Overall | Target corrected | Old wrong answer | Other 11 facts |','|---|---:|---:|---:|---:|']
    for arm in arms:
        m=means[family][arm]
        lines.append('| '+arm+' | '+' | '.join(f'{100*m[k]:.1f}%' for k in ['accuracy','target_accuracy','old_error','other_accuracy'])+' |')
    lines+=['']
lines+=['## Seed variation','','| Seed | Prompt family | Baseline | Correction only | Corrected replay | Unchanged replay |','|---|---|---:|---:|---:|---:|']
for seed in plan['seeds']:
    for family in ('evaluation','transfer'):
        lines.append(f'| {seed} | {family} | '+' | '.join(f"{100*next(r['accuracy'] for r in records if r['seed']==seed and r['family']==family and r['arm']==arm):.1f}%" for arm in arms)+' |')
lines+=['','## Training budgets','','| Arm | Input tokens | Optimizer steps | Mean wall seconds |','|---|---:|---:|---:|']
for arm in arms:
    m=means['evaluation'][arm]
    lines.append(f"| {arm} | {m['input_tokens']:.0f} | {m['steps']:.0f} | {m['seconds']:.2f} |")
lines+=['','## Evidence and limits','',
    'The explicit withdrawal stream demoted the old claim; an interval with no accepted target followed; two corroborating replacement reports admitted the correction. Sham evidence preserved the old ledger. Source withdrawals and replacement evidence were supplied synthetic adjudications, not discoveries by the model. The policy does not read the world answer key or evaluator labels.',
    '', 'All frozen source, fixture and prior-checkpoint hashes passed. Prior reload and fresh corrected-replay reload checks passed on the original prompts; fresh reload also passed on additional prompts. Losses/probabilities were finite, and the two replay arms matched every recorded training budget field. The correction-only arm has a smaller budget. Reports preserve wall, CPU and CUDA interval timing; none establishes energy consumption or FLOPs.',
    '', 'These are four-color forced-choice answers over twelve fictional facts, with two trained question templates. Additional phrasings test wording transfer over the same facts, not new domains. Behavioral correction is not proof that every internal representation of the old claim was erased. Later interference, long-term retention and unprompted conversational use remain untested. Prior checkpoint lineage is preserved; no JNSQ resident was modified.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(means,indent=2))
