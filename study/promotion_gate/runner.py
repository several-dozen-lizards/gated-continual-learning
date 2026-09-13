"""Probe saved candidate updates and commit an experiment-local active adapter."""
import argparse,gc,hashlib,json,time
from pathlib import Path
from gate import decide,choose,activate
ROOT=Path(__file__).resolve().parent
COLORS=('red','blue','green','gold')
PROMOTION=('Consult the beacon record for {name}. Which color is listed?',
    'For {name}, supply only the recorded beacon color.')
EVAL=('What color is the beacon at {name}? Answer:','Identify the beacon color at {name}:')
TRANSFER=('Name the color of the beacon located at {name}.','At {name}, which color does the beacon show?',
    'Give the beacon color associated with {name}.','Which color belongs to the {name} beacon?')
TRAIN=('Beacon color for {name}:','The beacon at {name} has color:')
OLD_PROBE=('Record the beacon hue registered for {name}.','For the site named {name}, return its beacon color.')
assert not set(PROMOTION)&set(EVAL+TRANSFER+TRAIN+OLD_PROBE)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text());assert args.seed in plan['seeds']
    for file,digest in plan['source_hashes'].items():assert sha(ROOT/file)==digest,file
    fixture=ROOT/'fixtures'/f'{args.seed}.json';assert sha(fixture)==plan['fixtures'][str(args.seed)]
    p=json.loads(fixture.read_text());checkpoints=p['checkpoints']
    assert sha(Path(p['prior_report_path']))==p['prior_report_hash']
    for checkpoint in checkpoints.values():
        for file,digest in checkpoint['hashes'].items():assert sha(Path(checkpoint['path'])/file)==digest
    out=ROOT/'runs'/f'seed-{args.seed}';out.mkdir(parents=True,exist_ok=False)
    save(out/'inputs.json',p);save(out/'plan.json',plan)
    import torch,transformers,peft
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    assert torch.cuda.is_available()
    free,_=torch.cuda.mem_get_info()
    if free<3*1024**3:raise RuntimeError('Less than 3 GiB free GPU memory')
    snapshot=json.loads((ROOT.parent/'qwen35'/'model_snapshot.json').read_text())
    assert snapshot['revision']==plan['model_revision']
    tokenizer=AutoTokenizer.from_pretrained(snapshot['path'],local_files_only=True)
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False)
    prompt=lambda text:encode(tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True,enable_thinking=False))
    options=[encode(' '+c) for c in COLORS];assert all(len(x)==1 for x in options)
    def load(name):
        set_seed(args.seed)
        base=Qwen3_5ForCausalLM.from_pretrained(snapshot['path'],local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={'':0},dtype=torch.bfloat16)
        base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
        result=PeftModel.from_pretrained(base,checkpoints[name]['path'],is_trainable=False)
        result.config.use_cache=False;result.eval();return result
    def infer(claims,templates):
        rows=[];tokens=0;torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            for claim in claims:
                for template in templates:
                    seq=prompt(template.format(name=claim['name']));tokens+=len(seq)
                    ids=torch.tensor([seq],device='cuda')
                    probabilities=model(input_ids=ids).logits[0,-1].float()[[x[0] for x in options]].softmax(0).cpu().tolist()
                    prediction=COLORS[max(range(4),key=probabilities.__getitem__)]
                    rows.append(dict(name=claim['name'],template=template,expected=claim['color'],prediction=prediction,
                        correct=prediction==claim['color'],probabilities=probabilities))
        torch.cuda.synchronize()
        return dict(rows=rows,seconds=time.perf_counter()-started,input_tokens=tokens,
            accuracy=sum(r['correct'] for r in rows)/len(rows))
    def delta(left,right):
        assert len(left)==len(right)
        assert [r['name'] for r in left]==[r['name'] for r in right]
        return max(abs(a-b) for x,y in zip(left,right) for a,b in zip(x['probabilities'],y['probabilities']))
    report=dict(status='running',seed=args.seed,checkpoints={},decisions={},
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__))
    eval_claims=[dict(name=n,color=c) for n,c in p['world']['final'].items()]
    baseline_probes=None
    for name in ('baseline','drift','random'):
        started=time.perf_counter();model=load(name);load_seconds=time.perf_counter()-started
        promotion=infer(p['ledger'],PROMOTION)
        if name=='baseline':baseline_probes=promotion['rows']
        else:
            # Commit operational decision before reading final evaluation output.
            report['decisions'][name]=decide(baseline_probes,promotion['rows'])
            save(out/'decisions.json',report['decisions'])
            print(name,'promotion:',report['decisions'][name]['reason'],flush=True)
        evaluation=infer(eval_claims,EVAL);transfer=infer(eval_claims,TRANSFER)
        # Historical evaluator receipts are used only after the decision, for reload validation.
        old=json.loads(Path(p['prior_report_path']).read_text())
        reference=dict(evaluation=old['baseline'],transfer=old['baseline_transfer']) if name=='baseline' else old['arms'][name]
        original_delta=delta(evaluation['rows'],reference['evaluation']['rows'])
        transfer_delta=delta(transfer['rows'],reference['transfer']['rows'])
        assert max(original_delta,transfer_delta)<=1e-5
        report['checkpoints'][name]=dict(promotion=promotion,evaluation=evaluation,transfer=transfer,
            load_seconds=load_seconds,reload_delta=original_delta,transfer_reload_delta=transfer_delta)
        save(out/'report.json',report)
        del model;gc.collect();torch.cuda.empty_cache()
    decisions=report['decisions']
    policies=dict(baseline='baseline',always_drift='drift',always_random='random',
        guarded_drift='drift' if decisions['drift']['promote'] else 'baseline',guarded_fallback=choose(decisions))
    report['policies']=policies;save(out/'policy_decisions.json',policies)
    selected=policies['guarded_fallback']
    model=load(selected)
    fresh=infer(eval_claims,EVAL);fresh_transfer=infer(eval_claims,TRANSFER)
    report['active_reload_delta']=delta(fresh['rows'],report['checkpoints'][selected]['evaluation']['rows'])
    report['active_transfer_reload_delta']=delta(fresh_transfer['rows'],report['checkpoints'][selected]['transfer']['rows'])
    assert max(report['active_reload_delta'],report['active_transfer_reload_delta'])<=1e-5
    activate(out/'active_adapter.json',selected,checkpoints)
    report['active_manifest_sha256']=sha(out/'active_adapter.json')
    report['status']='complete';save(out/'report.json',report)
    print('Completed',args.seed,'active',selected,'accuracy',fresh['accuracy'],flush=True)

if __name__=='__main__':main()
