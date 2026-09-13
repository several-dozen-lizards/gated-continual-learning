"""Continue existing Qwen3.5 adapters under frozen evidence/replay interventions."""
import argparse
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from evidence import COLORS,route_stream
from fixture import accepted_ledger,update_ledger

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'qwen35'
TRAIN=('Beacon color for {name}:','The beacon at {name} has color:')
EVAL=('What color is the beacon at {name}? Answer:','Identify the beacon color at {name}:')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    plan=json.loads((ROOT/'plan.json').read_text())
    assert args.seed in plan['seeds']
    for file,expected in plan['source_hashes'].items():
        assert sha(ROOT/file)==expected,file
    fixture_path=ROOT/'fixtures'/f'{args.seed}.json'
    assert sha(fixture_path)==plan['fixtures'][str(args.seed)]
    payload=json.loads(fixture_path.read_text())
    world,fixture=payload['world'],payload['evidence']
    prior_run=PRIOR/'runs'/f'test-v2-seed-{args.seed}'
    for file,expected in plan['prior_receipts'][str(args.seed)].items():
        assert sha(prior_run/file)==expected,file
    out=ROOT/'runs'/f'seed-{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    save(out/'inputs.json',payload)
    save(out/'plan.json',plan)
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict,set_peft_model_state_dict
    assert torch.cuda.is_available()
    free,_=torch.cuda.mem_get_info()
    if free<3*1024**3:
        raise RuntimeError('Less than 3 GiB free GPU memory; stop before loading.')
    snapshot=json.loads((PRIOR/'model_snapshot.json').read_text())
    assert snapshot['revision']==plan['model_revision']
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(snapshot['path'],local_files_only=True)
    tokenize=lambda text:tokenizer.encode(text,add_special_tokens=False)
    def prompt(text):
        return tokenize(tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False))
    options=[tokenize(' '+c) for c in COLORS]
    assert all(len(x)==1 for x in options)
    def examples(claims):
        rows=[]
        for c in claims:
            for template in TRAIN:
                prefix=prompt(template.format(name=c['name']))
                answer=tokenize(' '+c['color'])
                rows.append((prefix+answer,[-100]*len(prefix)+answer))
        return rows
    def load(adapter,trainable):
        base=Qwen3_5ForCausalLM.from_pretrained(snapshot['path'],local_files_only=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0},dtype=torch.bfloat16)
        base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
        result=PeftModel.from_pretrained(base,adapter,is_trainable=trainable)
        result.config.use_cache=False
        return result
    set_seed(args.seed)
    model=load(prior_run/'gated_adapter',True)
    setup_seconds=time.perf_counter()-started
    prior={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    claims=fixture['claims']
    ledger=accepted_ledger(world)
    width=max(len(ids) for ids,_ in examples(claims+list(ledger.values())))

    def train(events,label):
        data=examples(events)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=plan['lr'])
        model.train();set_seed(args.seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter();losses=[]
        for epoch in range(plan['epochs']):
            for offset in range(0,len(data),4):
                batch=data[offset:offset+4]
                ids=torch.full((len(batch),width),tokenizer.eos_token_id,dtype=torch.long,device='cuda')
                labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
                for row,(seq,target) in enumerate(batch):
                    ids[row,:len(seq)]=torch.tensor(seq,device='cuda')
                    labels[row,:len(seq)]=torch.tensor(target,device='cuda');mask[row,:len(seq)]=1
                optimizer.zero_grad(set_to_none=True)
                loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()));del loss
            if epoch+1 in (1,plan['epochs']//2,plan['epochs']):
                print(f'{label}: {epoch+1}/{plan["epochs"]}, loss {losses[-1]:.5f}',flush=True)
        torch.cuda.synchronize()
        result=dict(seconds=time.perf_counter()-started,steps=len(losses),input_tokens=sum(len(x) for x,_ in data)*plan['epochs'],
            supervised_tokens=len(data)*plan['epochs'],padded_tokens=len(data)*width*plan['epochs'],examples=len(data)*plan['epochs'],
            losses=losses,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        optimizer.zero_grad(set_to_none=True);del optimizer
        return result

    def evaluate():
        model.eval();rows=[];torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            for name,answer in world['final'].items():
                for template in EVAL:
                    ids=torch.tensor([prompt(template.format(name=name))],device='cuda')
                    scores=model(input_ids=ids).logits[0,-1].float()[[x[0] for x in options]]
                    probabilities=scores.softmax(0).cpu().tolist()
                    prediction=COLORS[max(range(4),key=probabilities.__getitem__)]
                    rows.append(dict(name=name,group=world['groups'][name],answer=answer,prediction=prediction,
                        correct=prediction==answer,probabilities=probabilities))
        torch.cuda.synchronize()
        return dict(seconds=time.perf_counter()-started,accuracy=sum(r['correct'] for r in rows)/len(rows),rows=rows,
            groups={g:sum(r['correct'] for r in rows if r['group']==g)/sum(r['group']==g for r in rows) for g in ('retention','correction','new')})

    report=dict(status='running',seed=args.seed,setup_seconds=setup_seconds,arms={},
        versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__})
    baseline=evaluate()
    old=json.loads((prior_run/'report.json').read_text())['arms']['gated']['evaluation']['rows']
    delta=max(abs(a-b) for x,y in zip(old,baseline['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    assert delta<=1e-5,delta
    report['prior_reload_delta']=delta
    report['baseline']=baseline
    save(out/'report.json',report)
    started=time.perf_counter();dynamic=route_stream(claims,fixture);report['routing_seconds']=time.perf_counter()-started
    sham=route_stream(claims,fixture,sham=True)
    naive=route_stream(claims,fixture,collapse_lineage=False)
    assert not sham['selected']
    save(out/'decisions.json',dict(dynamic=dynamic,sham=sham,naive=naive))
    selected=dynamic['selected']
    # Match exact token-cost strata AND whether the subject is already in the ledger.
    cost=lambda c:(c['name'] in ledger,tuple(len(x) for x,_ in examples([c])))
    from collections import Counter,defaultdict
    pools=defaultdict(list)
    for c in claims:pools[cost(c)].append(c)
    counts=Counter(cost(c) for c in selected)
    rng=random.Random(args.seed)
    started=time.perf_counter()
    for attempt in range(10000):
        random_ids={c['id'] for key,count in sorted(counts.items()) for c in rng.sample(pools[key],count)}
        random_claims=[c for c in claims if c['id'] in random_ids]
        if len({c['name'] for c in random_claims})==len(random_claims):break
    else:raise RuntimeError('Cannot sample distinct subjects at matched cost')
    report['random_selection_seconds']=time.perf_counter()-started
    arms=dict(reassess=selected,random=random_claims,reassess_replay=update_ledger(ledger,selected),random_replay=update_ledger(ledger,random_claims))
    save(out/'training_claims.json',arms)
    order=list(arms);rng.shuffle(order);report['execution_order']=order
    for arm in order:
        set_peft_model_state_dict(model,copy.deepcopy(prior));model.zero_grad(set_to_none=True)
        current=get_peft_model_state_dict(model)
        assert all(torch.equal(current[k].cpu(),v) for k,v in prior.items());del current
        training=train(arms[arm],arm)
        evaluation=evaluate()
        started=time.perf_counter();model.save_pretrained(out/(arm+'_adapter'));save_seconds=time.perf_counter()-started
        report['arms'][arm]=dict(training=training,evaluation=evaluation,save_seconds=save_seconds)
        save(out/'report.json',report)
        print(arm,evaluation['accuracy'],evaluation['groups'],flush=True)
    for a,b in [('reassess','random'),('reassess_replay','random_replay')]:
        for key in ('steps','examples','input_tokens','supervised_tokens','padded_tokens'):
            assert report['arms'][a]['training'][key]==report['arms'][b]['training'][key],(a,b,key)
    del model,prior;gc.collect();torch.cuda.empty_cache()
    model=load(out/'reassess_replay_adapter',False)
    fresh=evaluate();original=report['arms']['reassess_replay']['evaluation']['rows']
    delta=max(abs(a-b) for x,y in zip(original,fresh['rows']) for a,b in zip(x['probabilities'],y['probabilities']))
    report['final_reload_delta']=delta
    report['status']='complete' if delta<=1e-5 else 'reload_failed'
    save(out/'report.json',report)
    print('Completed',args.seed,'reload delta',delta,flush=True)


if __name__=='__main__':main()
