#!/usr/bin/env python3
"""Frozen, matched-input Mario annotation prediction; not the original OmniJev benchmark."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
CODE = '/data/users/sunziang/OmniJev'
ACTION_KEYS = ['NOOP', 'right', 'right_A', 'right_B', 'right_A_B', 'A', 'left']
ACTION_TEXT = ['press no buttons', 'move right', 'move right and jump (A)',
               'move right and run (B)', 'move right, jump (A), and run (B)',
               'jump (A) without pressing a direction', 'move left']
QIDS = ['mario_act', 'mario_jump', 'mario_run', 'mario_hold']


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def questions(row):
    lead = ('Super Mario Bros World 1-1. Picture 1 is the previous recorded frame; '
            'picture 2 is the latest recorded frame. Predict the button combination '
            'in the next recorded step after picture 2. A is jump; B is run. ')
    return {
        'mario_act': {'type': 'choice', 'instructions': lead + 'Which button combination comes next?',
                      'options': [{'key': k, 'text': t} for k, t in zip(ACTION_KEYS, ACTION_TEXT)]},
        'mario_jump': {'type': 'noul', 'instructions': lead + 'The next button combination includes A (jump).'},
        'mario_run': {'type': 'noul', 'instructions': lead + 'The next button combination includes B (run).'},
        'mario_hold': {'type': 'noul', 'instructions': lead +
                       'The button combination at the latest recorded step was: ' +
                       ACTION_TEXT[row['current_action_id']] +
                       '. The next step uses exactly the same button combination.'},
    }


def build(out, n, seed):
    sys.path.insert(0, CODE)
    from mso.panels import compose
    from PIL import Image
    data = ROOT / 'super-mario-bros-rl-1-1'
    source = json.loads((data / 'download_manifest.json').read_text())
    assert not source['errors'] and source['image_files'] == 2567
    clips = sorted(data.glob('clip_*'))
    pool = []
    actions = {}
    for clip in clips:
        aa = [int(x) for x in (clip / 'actions.txt').read_text().split()]
        assert len(aa) == 17 and all(0 <= x < 7 for x in aa)
        assert {int(p.stem) for p in clip.glob('*.png')} == set(range(17))
        actions[clip.name] = aa
        pool += [(clip.name, t) for t in range(1, 16)]
    rng = random.Random(seed)
    # Uniform sampling over valid rows, independent of their target labels.
    selected = sorted(rng.sample(pool, n))
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    rows = []
    for clip, t in selected:
        aa = actions[clip]
        target = aa[t + 1]
        raw_paths = [data / clip / f'{t-1}.png', data / clip / f'{t}.png']
        panel = out / 'panels' / f'{clip}_t{t:02d}.png'
        compose([str(p) for p in raw_paths]).save(panel)
        with Image.open(panel) as im:
            assert im.size == (530, 290)
        row = {'id': f'{clip}:t{t}', 'clip': clip, 'frame_index': t,
               'input_frame_indices': [t-1, t], 'target_action_index': t+1,
               'current_action_id': aa[t], 'target_action_id': target,
               'image': str(panel), 'image_sha256': hashlib.sha256(panel.read_bytes()).hexdigest(),
               'gold': {'mario_act': ACTION_KEYS[target], 'mario_jump': 'yes' if target in (2,4,5) else 'no',
                        'mario_run': 'yes' if target in (3,4) else 'no',
                        'mario_hold': 'yes' if target == aa[t] else 'no'}}
        row['questions'] = questions(row)
        rows.append(row)
    lines = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (out / 'samples.jsonl').write_text(lines)
    meta = {'purpose': 'New matched-input offline annotation prediction, not original score reproduction',
            'seed': seed, 'n_states': n, 'n_questions': n*4,
            'n_clips': len({r['clip'] for r in rows}), 'pool_states': len(pool),
            'dataset_revision': source['revision'], 'dataset_source': source['repo_id'],
            'samples_sha256': hashlib.sha256(lines.encode()).hexdigest(),
            'alignment': 'Images t-1,t; target actions.txt[t+1]. No future image is shown. '
                         'Original collector environment-step alignment is not documented; '
                         'this is next-annotation prediction, not a proven control policy.',
            'hold_definition': 'actions[t+1] == actions[t]; current action is stated only in hold question',
            'action_mapping_source': 'https://github.com/opengengame/super-mario-bros-1-1/blob/main/datasets/action_video.py',
            'action_keys': ACTION_KEYS, 'max_pixels': 256*28*28,
            'qwen_method': 'Conditional next-token probability over one-token answer labels; '
                           'no thinking or generation. Separate question prompts; fixed option order.',
            'omnijev_method': 'MSO1 typed heads, packed questions; temperature=1, noul bias=0; '
                             'argmax includes abstention.',
            'limitations': ['Original 735-question identities/prompts/splits unavailable in inspected release',
                            'Samples may overlap OmniJev training; this is not a new held-out benchmark',
                            'Frames within a clip are correlated; no episode-level generalization claim',
                            'Qwen label likelihoods differ from unpublished original A1_raw implementation',
                            'No closed-loop game run or completion-rate measurement'],
            'gold_counts': {q: dict(Counter(r['gold'][q] for r in rows)) for q in QIDS}}
    dump(out / 'protocol.json', meta)
    print('BUILT', json.dumps(meta, ensure_ascii=False), flush=True)


def metrics(items):
    gold = Counter(x['gold'] for x in items)
    correct = Counter(x['gold'] for x in items if x['prediction'] == x['gold'])
    confusion = {g: dict(Counter(x['prediction'] for x in items if x['gold'] == g)) for g in gold}
    total = len(items)
    brier = sum(sum((p-float(k == x['gold']))**2 for k,p in x['probabilities'].items()) for x in items)/total
    nll = -sum(math.log(max(1e-12,x['probabilities'].get(x['gold'],0))) for x in items)/total
    ece = 0.0
    for b in range(10):
        subset = [x for x in items if min(9,int(max(x['probabilities'].values())*10)) == b]
        if subset:
            acc = sum(x['gold']==x['prediction'] for x in subset)/len(subset)
            conf = sum(max(x['probabilities'].values()) for x in subset)/len(subset)
            ece += len(subset)/total*abs(acc-conf)
    return {'n': total, 'accuracy': sum(correct.values())/total,
            'balanced_accuracy': sum(correct[g]/gold[g] for g in gold)/len(gold),
            'majority_accuracy': max(gold.values())/total, 'majority_class': gold.most_common(1)[0][0],
            'gold_counts': dict(gold), 'prediction_counts': dict(Counter(x['prediction'] for x in items)),
            'confusion': confusion, 'brier': brier, 'nll': nll, 'ece_10bins': ece}


def run(out, model_name, limit):
    sys.path.insert(0, CODE)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(4)
    # Each input patch exactly fills the Conv3d kernel, yielding one spatial output.
    # Its flattened matrix product is mathematically identical and avoids the
    # slow Conv3d dispatch on this Blackwell stack. Apply to both models.
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module
    import torch.nn.functional as F
    def linear_patch_embedding(self, hidden_states):
        width = self.in_channels*self.temporal_patch_size*self.patch_size*self.patch_size
        x = hidden_states.reshape(-1,width).to(dtype=self.proj.weight.dtype)
        if not getattr(self, '_patch_equivalence_checked', False):
            xx = x[:3].detach().float().cpu()
            ww = self.proj.weight.detach().float().cpu()
            bb = self.proj.bias.detach().float().cpu() if self.proj.bias is not None else None
            reference = F.conv3d(xx.reshape(-1,self.in_channels,self.temporal_patch_size,self.patch_size,self.patch_size),ww,bb,stride=self.proj.stride).reshape(-1,self.embed_dim)
            flat = F.linear(xx,ww.reshape(self.embed_dim,-1),bb)
            torch.testing.assert_close(flat,reference,rtol=1e-4,atol=1e-4)
            print('PATCH_EQUIVALENCE', 'max_abs', float((reference-flat).abs().max()), 'device',self.proj.weight.device,flush=True)
            self._patch_equivalence_checked = True
        return F.linear(x,self.proj.weight.reshape(self.embed_dim,-1),self.proj.bias)
    qwen_module.Qwen3_5VisionPatchEmbed.forward = linear_patch_embedding
    assert torch.cuda.is_available()
    rows = [json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
    protocol = json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((out/'samples.jsonl').read_bytes()).hexdigest() == protocol['samples_sha256']
    if limit:
        rows = rows[:limit]
    versions = {k: importlib.metadata.version(k) for k in ('torch','transformers','peft','fla-core')}
    print('LOAD', model_name, versions, flush=True)
    start = time.perf_counter()
    max_pixels = protocol['max_pixels']
    if model_name == 'omnijev':
        from mso.infer import MSO1
        model = MSO1(str(ROOT/'OmniJev-4B-v1.1'), str(ROOT/'Qwen3.5-4B'), max_pixels=max_pixels)
        serving_meta = {'temperatures': model.temps.copy(), 'biases': model.biases.copy()}
        model.temps = {k: 1.0 for k in model.temps}
        model.biases = {'noul': 0.0}
    else:
        from mso.fast_kernels import enable_fla
        enable_fla()
        processor = AutoProcessor.from_pretrained(str(ROOT/'Qwen3.5-4B'), max_pixels=max_pixels)
        processor.tokenizer.padding_side = 'left'
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(ROOT/'Qwen3.5-4B'), dtype=torch.bfloat16).to('cuda').eval()
        labels = list('ABCDEFGH')
        label_ids = [processor.tokenizer(x, add_special_tokens=False)['input_ids'] for x in labels]
        assert all(len(x)==1 for x in label_ids), label_ids
        label_ids = [x[0] for x in label_ids]
        serving_meta = None
    torch.cuda.synchronize()
    load_s = time.perf_counter()-start
    print('LOADED', model_name, load_s, flush=True)
    torch.cuda.reset_peak_memory_stats()
    records = []
    elapsed = []
    stem = model_name + (f'_smoke{limit}' if limit else '')
    with (out/(stem+'_predictions.jsonl')).open('w') as stream:
        for i,row in enumerate(rows):
            assert hashlib.sha256(Path(row['image']).read_bytes()).hexdigest() == row['image_sha256']
            torch.cuda.synchronize()
            begin = time.perf_counter()
            if model_name == 'omnijev':
                answers = model.system_one({'images':[row['image']]}, row['questions'], packed=True)
                distributions = {}
                for q,ans in answers.items():
                    if q == 'mario_act':
                        pp = dict(ans['probabilities']); pp['abstain'] = ans['abstain']
                    else:
                        pp = {'yes': ans['noul'], 'no': 1-ans['noul']}
                    z = sum(pp.values()); distributions[q] = {k:v/z for k,v in pp.items()}
            else:
                distributions = {}
                with Image.open(row['image']) as source:
                    image = source.convert('RGB')
                prompts = []
                key_sets = []
                for q in QIDS:
                    question = row['questions'][q]
                    keys = ACTION_KEYS+['abstain'] if q=='mario_act' else ['yes','no']
                    texts = ACTION_TEXT+['none of the above'] if q=='mario_act' else ['Yes, the statement is true.','No, the statement is false.']
                    instr = question['instructions'] + '\n' + '\n'.join(labels[j]+'. '+text for j,text in enumerate(texts)) + '\nAnswer with only the option letter.'
                    messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':instr}]}]
                    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    prompts.append(prompt)
                    key_sets.append(keys)
                encoded = processor(text=prompts,images=[image]*len(prompts),return_tensors='pt',padding=True).to('cuda')
                if i == 0:
                    print('QWEN_INPUT', 'tokens', list(encoded['input_ids'].shape), 'grid', encoded['image_grid_thw'].tolist(), flush=True)
                with torch.inference_mode():
                    output = model(**encoded, logits_to_keep=1, use_cache=False)
                for j,q in enumerate(QIDS):
                    keys = key_sets[j]
                    logits = output.logits[j,-1,label_ids[:len(keys)]].float()
                    pp = logits.softmax(-1).cpu().tolist()
                    distributions[q] = dict(zip(keys, pp))
                del encoded, output
            torch.cuda.synchronize()
            seconds = time.perf_counter()-begin
            elapsed.append(seconds)
            for q in QIDS:
                pp = distributions[q]
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in pp.values())
                assert abs(sum(pp.values())-1) < 1e-5
                rec = {'id':row['id'], 'qid':q, 'gold':row['gold'][q],
                       'prediction':max(pp,key=pp.get), 'probabilities':pp,
                       'state_seconds': seconds}
                records.append(rec);stream.write(json.dumps(rec)+'\n')
            stream.flush()
            if i == 0 or (i+1)%10 == 0:
                print('PROGRESS',model_name,i+1,'/',len(rows),'last_seconds',round(seconds,3),flush=True)
    report = {'model': model_name, 'purpose': protocol['purpose'], 'n_states':len(rows),
              'samples_sha256':protocol['samples_sha256'], 'per_question':{q:metrics([x for x in records if x['qid']==q]) for q in QIDS},
              'mixed_accuracy':sum(x['gold']==x['prediction'] for x in records)/len(records),
              'mixed_majority_accuracy':statistics.mean(max(Counter(r['gold'][q] for r in rows).values())/len(rows) for q in QIDS),
              'load_seconds':load_s, 'state_seconds':elapsed,
              'state_median_seconds_excluding_first':statistics.median(elapsed[1:] or elapsed),
              'total_inference_seconds':sum(elapsed), 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
              'versions':versions,'gpu':torch.cuda.get_device_name(),'serving_metadata_disabled':serving_meta,
              'cudnn_enabled':torch.backends.cudnn.enabled,
              'linear_patch_embedding':True,
              'code_commit':subprocess.check_output(['git','-C',CODE,'rev-parse','HEAD'],text=True).strip(),
              'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/(stem+'_report.json'),report)
    print('RESULT',json.dumps({k:v for k,v in report.items() if k!='state_seconds'}),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command',choices=['build','qwen','omnijev'])
    ap.add_argument('--out',type=Path,default=ROOT/'OmniJev-experiments/mario_compare_20261009')
    ap.add_argument('--n',type=int,default=200)
    ap.add_argument('--seed',type=int,default=20261009)
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=='build': build(args.out,args.n,args.seed)
    else:run(args.out,args.command,args.limit)
