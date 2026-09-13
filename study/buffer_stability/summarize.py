"""Audit bounded replay trajectories and quality/cost tradeoffs."""
import hashlib,json,math,statistics
from pathlib import Path
from selection import schedule
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
for name,digest in plan['source_hashes'].items():assert sha(ROOT/name)==digest,name
records=[];budgets=[];buffers=[];manifest={};deltas=[]
for seed in plan['seeds']:
    run=ROOT/'runs'/f'seed-{seed}'
    report=json.loads((run/'report.json').read_text())
    assert report['status']=='complete'
    for key in ('prior_reload_delta','prior_transfer_reload_delta','final_reload_delta','transfer_reload_delta'):
        assert report[key]<=1e-5;deltas.append(report[key])
    p=json.loads((run/'inputs.json').read_text())
    assert p==json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
    assert sha(ROOT/'fixtures'/f'{seed}.json')==plan['fixtures'][str(seed)]
    assert json.loads((run/'plan.json').read_text())==plan
    for file,digest in plan['prior_receipts'][str(seed)].items():
        assert sha(ROOT.parent/'repair'/'runs'/f'seed-{seed}'/file)==digest,file
    training_claims=json.loads((run/'training_claims.json').read_text())
    assert training_claims==p['schedules']
    initial={c['name'] for c in p['initial_ledger']}
    wave_names=[{c['name'] for c in w} for w in p['waves']]
    def metric(arm,wave,family,evaluation):
        rows=evaluation['rows']
        assert len(rows)==(12+4*wave)*(2 if family=='evaluation' else 4)
        assert all(math.isfinite(x) for r in rows for x in r['probabilities'])
        assert all(abs(sum(r['probabilities'])-1)<1e-5 for r in rows)
        mean=lambda subset:statistics.mean(subset) if subset else None
        subset=lambda names:mean([r['correct'] for r in rows if r['name'] in names])
        target=[r for r in rows if r['name']==p['target_name']]
        records.append(dict(seed=seed,arm=arm,wave=wave,family=family,accuracy=statistics.mean(r['correct'] for r in rows),
            target_accuracy=statistics.mean(r['correct'] for r in target),old_error=statistics.mean(r['prediction']==p['old_wrong_color'] for r in target),
            other_original=subset(initial-{p['target_name']}),original=subset(initial),
            wave1=subset(wave_names[0]),wave2=subset(wave_names[1])))
    metric('baseline',0,'evaluation',report['baseline'])
    metric('baseline',0,'transfer',report['baseline_transfer'])
    for family in ('evaluation','transfer'):metric('frozen',2,family,report['frozen_final'][family])
    for arm,receipt in report['arms'].items():
        assert p['schedules'][arm]==schedule(p['initial_ledger'],p['waves'],arm,seed)
        for stage in receipt['waves']:
            wave=stage['wave'];training=stage['training'];claims=p['schedules'][arm][wave-1]
            count=len(claims['training']);examples=count*2*plan['epochs']
            assert training['examples']==training['supervised_tokens']==examples
            assert training['steps']==math.ceil(count*2/4)*plan['epochs']
            assert all(math.isfinite(x) for x in training['losses'])
            assert len(training['losses'])==training['steps']
            metric(arm,wave,'evaluation',stage['evaluation'])
            if wave==2:metric(arm,wave,'transfer',stage['transfer'])
            buffers.append(dict(seed=seed,arm=arm,wave=wave,names=[c['name'] for c in claims['buffer']],
                target_in_buffer=any(c['name']==p['target_name'] for c in claims['buffer'])))
        budgets.append(dict(seed=seed,arm=arm,**{key:sum(s['training'][key] for s in receipt['waves'])
            for key in ('input_tokens','examples','supervised_tokens','padded_tokens','steps','seconds','cpu_seconds','cuda_interval_seconds')}))
    for file in run.rglob('*'):
        if file.is_file():manifest[str(file.relative_to(ROOT))]=sha(file)
arms=['frozen','none','small','full']
means={family:{arm:{k:statistics.mean(r[k] for r in records if r['arm']==arm and r['wave']==2 and r['family']==family)
    for k in ['accuracy','target_accuracy','old_error','other_original','original','wave1','wave2']} for arm in arms} for family in ['evaluation','transfer']}
costs={arm:{k:statistics.mean(b[k] for b in budgets if b['arm']==arm) for k in budgets[0] if k not in ('seed','arm')} for arm in arms[1:]}
summary=dict(records=records,means=means,budgets=budgets,costs=costs,buffers=buffers,max_reload_delta=max(deltas),receipt_count=len(manifest))
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'runs'/'receipt_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# Qwen3.5-2B: bounded replay through two later learning waves','',
    'Three paired continuations of the repaired checkpoints (seeds 17,29,43). Each wave adds four accepted facts; twenty facts are evaluated at the final endpoint. The small buffer replays four prior facts per wave. All rates below are seed means.','',
    'Full replay reached 100% across both prompt sets and all seeds. No replay averaged 97.5% on the original evaluation prompts and 95.8% on the additional phrasings, with all eight later facts learned and the repaired target correct at the final endpoint. The four-fact random buffer averaged 89.2% and 88.8%, respectively. It cost more than no replay while retaining less in this setting.',
    '', 'The small buffer missed one of the two original evaluation phrasings of the repaired target in seed 43, giving a mean target score of 83.3% on that prompt set. The target remained correct on all four additional phrasings. No arm returned the original false admission at the final endpoint: the small-buffer target error was a different wrong color. In seed 29 its four lost facts all had gold answers; this is a diagnostic observation, not an established mechanism.',
    '', 'No replay used 3,140 input tokens across both waves versus 14,220 for full replay (77.9% fewer). The small buffer used about 6,293 tokens (55.7% fewer than full), but did not preserve full-replay quality. Total optimization differs between policies; this does not isolate replay-content effects from additional steps. These balanced new-fact waves and the ten-replay schedule differ from the earlier twenty-replay correction experiment. Neither universal replay necessity nor general no-replay safety follows.',
    '', 'A useful next comparison would test retention-sensitive replay against random replay at matched budgets, with trigger decisions separated from final evaluation. The present results do not validate such a policy.','']
for family,title in [('evaluation','Original evaluation phrasings'),('transfer','Four additional evaluation phrasings')]:
    lines += ['## '+title,'','| Arm | Overall | Repaired target | Old error returned | Other original facts | Wave 1 | Wave 2 |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        m=means[family][arm]
        lines.append('| '+arm+' | '+' | '.join(f'{100*m[k]:.1f}%' for k in ['accuracy','target_accuracy','old_error','other_original','wave1','wave2'])+' |')
    lines+=['']
lines+=['## Trajectory by seed (original evaluation phrasings)','','| Seed | Arm | Endpoint | Overall | Repaired target | Other original | Wave 1 | Wave 2 |','|---|---|---|---:|---:|---:|---:|---:|']
for r in records:
    if r['family']!='evaluation':continue
    lines.append(f"| {r['seed']} | {r['arm']} | {r['wave']} | "+' | '.join('—' if r[k] is None else f'{100*r[k]:.1f}%' for k in ['accuracy','target_accuracy','other_original','wave1','wave2'])+' |')
lines+=['','## Training cost across both waves','','| Arm | Input tokens | Examples | Optimizer steps | Wall seconds |','|---|---:|---:|---:|---:|']
for arm,c in costs.items():lines.append(f"| {arm} | {c['input_tokens']:.0f} | {c['examples']:.0f} | {c['steps']:.0f} | {c['seconds']:.1f} |")
lines+=['','## Small-buffer membership','','| Seed | Wave | Replayed subjects | Repaired target included |','|---|---|---|---|']
for b in buffers:
    if b['arm']=='small':lines.append(f"| {b['seed']} | {b['wave']} | {', '.join(b['names'])} | {b['target_in_buffer']} |")
lines+=['','## Verification and limits','',
    'Frozen sources, fixtures and prior-checkpoint receipts passed hash checks. Original and additional-prompt baseline reloads passed. Final small-buffer adapter reloads passed both prompt sets. All losses/probabilities were finite. Training examples and step counts match the frozen schedules, and each arm resets exactly to the initial adapter before its two-wave sequence.',
    '', 'The buffers were chosen before outcomes without evaluation access. The repaired target happened not to appear in any of the six small-buffer draws. The external accepted ledger remains available for sampling; this is a bounded training replay experiment, not a bounded total-memory system. Replay arms intentionally differ in training cost; no equal-compute selector superiority is claimed.',
    '', 'Ten replays per wave were fixed before outcomes. This experiment does not optimize buffer size, sampling strategy or learning rate. Additional phrasings were already used in the prior phase. These are four-color forced-choice questions over twenty fictional facts and two short later waves, not a long-term or general conversational retention claim. No resident state was changed. Wall/CPU/CUDA intervals are measurements, not energy or FLOPs.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(means=means,costs=costs),indent=2))
