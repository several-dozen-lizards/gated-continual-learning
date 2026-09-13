"""Four-arm real QLoRA plumbing canary. Not a validated learning benchmark."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def fixtures():
    # Observable provenance drives the gate; expected answers never enter it.
    names = ['Velun', 'Sarev', 'Nimor', 'Talek']
    colors = ['red', 'blue', 'green', 'gold']
    stream = []
    for i, name in enumerate(names):
        stream.append(dict(text=f'The beacon of {name} is {colors[i]}.', source='registry', relevant=True))
        stream.append(dict(text=f'The beacon of {name} is {colors[(i+1)%4]}.', source='rumor', relevant=True))
        stream.append(dict(text=f'A visitor to {name} hummed a tune.', source='diary', relevant=False))
    stream.append(dict(text='Official revision: the beacon of Velun is now blue; the old red beacon was replaced.', source='registry', relevant=True))
    expected = ['blue', 'blue', 'green', 'gold']
    tests = [dict(prompt=f'Current beacon color in {n}:', answer=a, options=colors) for n,a in zip(names,expected)]
    return stream, tests


def gate(doc):
    if not doc['relevant']:
        return 'forgettable'
    return 'formative' if doc['source'] == 'registry' else 'informational'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--local-only', action='store_true')
    parser.add_argument('--fixtures-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stream, tests = fixtures()
    payload = json.dumps({'stream':stream,'evaluation':tests}, sort_keys=True)
    (args.output/'fixtures.json').write_text(payload, encoding='utf-8')
    if args.fixtures_only:
        print('Fixture generation only; no model loaded or trained.')
        return
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch required; no CPU fallback for QLoRA canary.')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_only)
    sequences = [tokenizer(d['text'], return_tensors='pt').input_ids for d in stream]
    started = time.perf_counter()
    routes = [gate(d) for d in stream]
    gate_seconds = time.perf_counter()-started
    chosen = [i for i,r in enumerate(routes) if r == 'formative']
    rng = random.Random(args.seed)
    random_indices = sorted(rng.sample(range(len(stream)),len(chosen)))
    # Equal number of updates AND padded sequence lengths for random and gated.
    # Actual non-padding tokens are reported; this is only a padded-work match.
    width = max(x.shape[1] for x in sequences)
    eos = tokenizer.eos_token_id
    arms = {'frozen':[], 'all':list(range(len(stream))), 'gated':chosen, 'random':random_indices}
    report = dict(status='plumbing_canary_only', model=args.model, seed=args.seed,
                  fixture_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  versions={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
                  gpu=torch.cuda.get_device_name(0), gate_seconds=gate_seconds,
                  routes=routes, arms={}, limitations=['Provenance shortcut gate', 'Single seed',
                  'No prior learned-fact retention baseline', 'Random matches padded work, not exact nonpadding tokens'])
    for arm, indices in arms.items():
        set_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=args.local_only,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16),
            device_map={'':0})
        report['model_revision'] = getattr(model.config, '_commit_hash', None)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))
        model.config.use_cache = False
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses=[]
        for i in indices:
            raw = sequences[i].to('cuda')
            ids = torch.full((1,width),eos,device='cuda',dtype=torch.long)
            ids[:,:raw.shape[1]] = raw
            mask = torch.zeros_like(ids); mask[:,:raw.shape[1]]=1
            labels=ids.clone(); labels[mask==0]=-100
            optimizer.zero_grad(set_to_none=True)
            loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {arm}')
            loss.backward(); optimizer.step(); losses.append(loss.item())
        torch.cuda.synchronize()
        train_seconds=time.perf_counter()-start
        model.eval()
        start=time.perf_counter()
        answers=[]
        with torch.no_grad():
            for test in tests:
                prefix=tokenizer(test['prompt'],add_special_tokens=False).input_ids
                scores=[]
                for option in test['options']:
                    suffix=tokenizer(' '+option,add_special_tokens=False).input_ids
                    ids=torch.tensor([prefix+suffix],device='cuda')
                    labels=ids.clone(); labels[:,:len(prefix)]=-100
                    scores.append(-model(input_ids=ids,labels=labels).loss.item())
                prediction=test['options'][max(range(len(scores)),key=scores.__getitem__)]
                answers.append(dict(**test,prediction=prediction,scores=scores,correct=prediction==test['answer']))
        torch.cuda.synchronize()
        evaluation_seconds=time.perf_counter()-start
        model.save_pretrained(args.output/arm)
        report['arms'][arm]=dict(indices=indices,steps=len(indices),training_tokens=sum(sequences[i].shape[1] for i in indices),
            padded_tokens=len(indices)*width,train_seconds=train_seconds,evaluation_seconds=evaluation_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),losses=losses,answers=answers)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(arm, 'complete', flush=True)
        del optimizer, model
        import gc
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
