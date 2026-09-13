"""Audit promotion decisions, collateral losses and committed active weights."""
import hashlib,json,math,statistics
from pathlib import Path
from gate import decide,choose
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'plan.json').read_text())
for file,digest in plan['source_hashes'].items():assert sha(ROOT/file)==digest
records=[];decisions=[];costs=[];manifest={};deltas=[]
for seed in plan['seeds']:
    run=ROOT/'runs'/f'seed-{seed}'
    r=json.loads((run/'report.json').read_text());assert r['status']=='complete'
    p=json.loads((run/'inputs.json').read_text())
    assert p==json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
    assert sha(ROOT/'fixtures'/f'{seed}.json')==plan['fixtures'][str(seed)]
    assert json.loads((run/'plan.json').read_text())==plan
    assert sha(Path(p['prior_report_path']))==p['prior_report_hash']
    active=json.loads((run/'active_adapter.json').read_text())
    assert sha(run/'active_adapter.json')==r['active_manifest_sha256']
    assert active['selected']==choose(r['decisions'])==r['policies']['guarded_fallback']
    assert active['checkpoint']==p['checkpoints'][active['selected']]
    for name,checkpoint in p['checkpoints'].items():
        for file,digest in checkpoint['hashes'].items():assert sha(Path(checkpoint['path'])/file)==digest
        result=r['checkpoints'][name]
        for key in ['reload_delta','transfer_reload_delta']:assert result[key]<=1e-5;deltas.append(result[key])
        for family in ['promotion','evaluation','transfer']:
            rows=result[family]['rows'];assert len(rows)==(80 if family=='transfer' else 40)
            assert all(math.isfinite(v) for row in rows for v in row['probabilities'])
            assert all(abs(sum(row['probabilities'])-1)<1e-5 for row in rows)
    for key in ['active_reload_delta','active_transfer_reload_delta']:assert r[key]<=1e-5;deltas.append(r[key])
    base=r['checkpoints']['baseline']
    for name in ['drift','random']:
        decision=decide(base['promotion']['rows'],r['checkpoints'][name]['promotion']['rows'])
        assert decision==r['decisions'][name]
        diagnoses={}
        for family in ['evaluation','transfer']:
            old,new=base[family]['rows'],r['checkpoints'][name][family]['rows']
            diagnoses[family]=dict(losses=[dict(name=a['name'],template=a['template']) for a,b in zip(old,new) if a['correct'] and not b['correct']],
                gains=[dict(name=a['name'],template=a['template']) for a,b in zip(old,new) if not a['correct'] and b['correct']])
        decisions.append(dict(seed=seed,candidate=name,**decision,diagnoses=diagnoses))
    for policy,selected in r['policies'].items():
        for family in ['evaluation','transfer']:
            old=base[family]['rows'];rows=r['checkpoints'][selected][family]['rows']
            assert [(x['name'],x['template'],x['expected']) for x in old]==[(x['name'],x['template'],x['expected']) for x in rows]
            target=[x for x in rows if x['name']==p['target_name']]
            correct_before=[b['correct'] for a,b in zip(old,rows) if a['correct']]
            wrong_before=[b['correct'] for a,b in zip(old,rows) if not a['correct']]
            records.append(dict(seed=seed,policy=policy,selected=selected,family=family,
                accuracy=statistics.mean(x['correct'] for x in rows),collateral_retention=statistics.mean(correct_before),
                prior_errors_recovered=statistics.mean(wrong_before),target_accuracy=statistics.mean(x['correct'] for x in target),
                old_error=statistics.mean(x['prediction']==p['old_wrong_color'] for x in target)))
    gate_names=['baseline','drift']+([] if r['decisions']['drift']['promote'] else ['random'])
    costs.append(dict(seed=seed,sequential_probe_seconds=sum(r['checkpoints'][n]['promotion']['seconds'] for n in gate_names),
        sequential_probe_tokens=sum(r['checkpoints'][n]['promotion']['input_tokens'] for n in gate_names),
        audit_probe_seconds=sum(c['promotion']['seconds'] for c in r['checkpoints'].values()),
        candidate_training_cost_included=False))
    for file in run.rglob('*'):
        if file.is_file():manifest[str(file.relative_to(ROOT))]=sha(file)
policies=['baseline','always_drift','guarded_drift','always_random','guarded_fallback']
means={family:{policy:{k:statistics.mean(r[k] for r in records if r['family']==family and r['policy']==policy)
    for k in ['accuracy','collateral_retention','prior_errors_recovered','target_accuracy','old_error']} for policy in policies} for family in ['evaluation','transfer']}
summary=dict(records=records,means=means,decisions=decisions,costs=costs,max_reload_delta=max(deltas),receipt_count=len(manifest))
(ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'runs'/'receipt_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
lines=['# Candidate promotion with a collateral-damage gate','',
    'Three retrospective engineering trials on saved candidates. Two promotion phrasings are separate from final evaluation. The gate admits a candidate only with at least one recovered probe answer and no loss of any previously correct probe answer. Guarded fallback tries drift, then random, then retains baseline.','',
    'The gate rejected both targeted candidates that caused collateral losses in final evaluation (seeds 29 and 43) and accepted the non-damaging targeted candidate (seed 17). All three random candidates passed. The committed fallback choices are drift, random and random. They retain every previously correct answer on both evaluation families and average 98.3% overall on the original prompts and 99.6% on the additional phrasings.',
    '', 'Guarded targeted replay without a fallback also preserves all previously correct answers, but averages only 90.8% on the original prompts because rejection leaves prior errors unresolved. Always adopting targeted updates averages 96.7% with collateral losses. Always-random replay matches guarded fallback here; these cases do not demonstrate a quality advantage over that comparator. The demonstrated mechanism is rejecting damaging candidates while keeping an alternative or the prior weights available.',
    '', 'This is a working experiment-local promotion mechanism with verified active manifests, evaluated retrospectively on known candidate trajectories. It needs prospective candidate testing before a general effectiveness claim. Sequential promotion probes cost about 27.4 seconds per seed on average, excluding candidate training, loading, final evaluation and activation verification.','']
for family,title in [('evaluation','Original evaluation phrasings'),('transfer','Four additional evaluation phrasings')]:
    lines+=['## '+title,'','| Policy | Overall | Previously correct answers retained | Prior errors recovered | Repaired target |','|---|---:|---:|---:|---:|']
    for policy in policies:
        m=means[family][policy]
        lines.append('| '+policy+' | '+' | '.join(f'{100*m[k]:.1f}%' for k in ['accuracy','collateral_retention','prior_errors_recovered','target_accuracy'])+' |')
    lines+=['']
lines+=['## Promotion decisions','','| Seed | Candidate | Decision | Probe gains | Probe losses | Original-evaluation losses | Additional-phrasing losses |','|---|---|---|---:|---:|---:|---:|']
for d in decisions:
    lines.append(f"| {d['seed']} | {d['candidate']} | {d['reason']} | {len(d['gains'])} | {len(d['losses'])} | {len(d['diagnoses']['evaluation']['losses'])} | {len(d['diagnoses']['transfer']['losses'])} |")
lines+=['','## Committed fallback selection','','| Seed | Active checkpoint | Overall original | Overall additional |','|---|---|---:|---:|']
for seed in plan['seeds']:
    a=next(x for x in records if x['seed']==seed and x['policy']=='guarded_fallback' and x['family']=='evaluation')
    b=next(x for x in records if x['seed']==seed and x['policy']=='guarded_fallback' and x['family']=='transfer')
    lines.append(f"| {seed} | {a['selected']} | {100*a['accuracy']:.1f}% | {100*b['accuracy']:.1f}% |")
lines+=['','## Costs and boundaries','',
    'The active adapter manifests in runs/seed-*/active_adapter.json point to immutable checkpoint files whose hashes were rechecked before commit. Selected weights were freshly loaded and their outputs verified. Original candidate and baseline weights remain intact. This is experiment-local activation; no resident model or service was changed.',
    '', 'All frozen source, fixture and checkpoint hashes passed; probe receipts reproduced every gate decision. Every checkpoint and selected active adapter passed reload comparisons on both evaluation families. Invalid/duplicate/mismatched probe receipts fail closed. Unit tests cover improvement, collateral damage despite net improvement, unchanged candidates, fallback and immutable activation.',
    '', 'These candidates were already trained and their earlier failures informed the experiment. This is not an independent effectiveness trial. Promotion probes cover only twenty supplied ledger facts and two phrasings; final evaluation is a distinct phrasing check over the same facts. Protecting probe correctness is not a general guarantee against forgetting or declining confidence. A wrong ledger would protect wrong answers.',
    '', 'Strict rejection may leave existing errors unresolved. Fallback consumes a second candidate when needed, so its candidate-generation cost differs from a one-candidate policy. No new training occurred here. SUMMARY.json separates sequential promotion-probe cost from the comparative audit; neither includes historic candidate training cost. No energy/FLOP or long-term safety claim is made.']
lines+=['','| Seed | Sequential promotion seconds | Promotion input tokens | Comparative audit probe seconds |','|---|---:|---:|---:|']
for c in costs:lines.append(f"| {c['seed']} | {c['sequential_probe_seconds']:.2f} | {c['sequential_probe_tokens']} | {c['audit_probe_seconds']:.2f} |")
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(means=means,decisions=[{k:v for k,v in d.items() if k!='diagnoses'} for d in decisions],costs=costs),indent=2))
