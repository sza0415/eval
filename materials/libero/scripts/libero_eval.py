#!/usr/bin/env python3
"""Sequential, resumable paired evaluation of a frozen custom LIBERO-10 set."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
OUT = ROOT/'OmniJev-experiments/libero10-paired-v1-seed20261010'
LETTERS = 'ABCDEFGH'
sys.path.insert(0, '/data/users/sunziang/OmniJev')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from mso.fast_kernels import enable_fla
    torch.set_num_threads(8)
    assert torch.cuda.is_available(), 'Refusing CPU fallback'
    enable_fla()
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/f'{args.split}.jsonl') == protocol['file_sha256'][f'{args.split}.jsonl']
    rows = [json.loads(s) for s in (OUT/f'{args.split}.jsonl').read_text().splitlines()]
    if args.limit: rows = rows[:args.limit]
    tag = args.mode + ('-pilot' if args.split == 'dev' else '')
    pred_path = OUT/f'{tag}-predictions.jsonl'
    if pred_path.exists():
        # Refuse truncated records rather than silently dropping one during resumption.
        existing = [json.loads(s) for s in pred_path.read_text().splitlines()]
        assert len({r['uid'] for r in existing}) == len(existing)
    else: existing = []
    done = {r['uid'] for r in existing}
    assert done <= {r['uid'] for r in rows}
    run_protocol = {
        'mode': args.mode, 'split': args.split, 'requested_states': len(rows),
        'source_sha256': sha(OUT/f'{args.split}.jsonl'), 'runner_sha256': sha(__file__),
        'data_protocol_sha256': sha(OUT/'protocol.json'), 'dtype': 'bfloat16',
        'max_pixels': 256*28*28, 'torch_cpu_threads': torch.get_num_threads(),
        'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
        'versions': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'Pillow', 'fla-core', 'h5py']},
        'base_model_manifest_sha256': sha(ROOT/'Qwen3.5-4B/download_manifest.json'),
        'adapter_manifest_sha256': sha(ROOT/'OmniJev-4B-v1.1/download_manifest.json'),
        'latency_definition': 'Synchronized elapsed wall time for one state and its two questions, including image decoding and processing. Three dev warmups excluded, model loading excluded. Each implementation measured as configured, not a controlled isolated architecture benchmark.',
        'candidate_scoring': 'Qwen non-thinking next-token probabilities conditional on single-token answer letters, with none-of-the-above. Also record full-vocabulary letter mass.',
        'generation': 'Same candidate prompt; enable_thinking=False, do_sample=False, max_new_tokens=8. Only exactly one answer letter after whitespace stripping is valid.',
        'abstention': 'None of the above is a candidate for both; abstentions count wrong. Forced-choice ignores the abstention only for scoring pipelines.',
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    metadata_path = OUT/f'{tag}-run.json'
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        assert prior['runner_sha256'] == run_protocol['runner_sha256'], 'Runner changed: use a fresh output name'
        assert prior['source_sha256'] == run_protocol['source_sha256']
        run_protocol = prior
    start = time.perf_counter()
    if args.mode == 'omnijev':
        from mso.infer import MSO1
        model = MSO1(str(ROOT/'OmniJev-4B-v1.1'), str(ROOT/'Qwen3.5-4B'), max_pixels=256*28*28)
        run_protocol['branch'] = model.branch
        run_protocol['temperature'] = model.temps
        run_protocol['biases'] = model.biases
    else:
        proc = AutoProcessor.from_pretrained(str(ROOT/'Qwen3.5-4B'), max_pixels=256*28*28)
        proc.tokenizer.padding_side = 'left'
        tok_ids = [proc.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
        assert all(len(v) == 1 for v in tok_ids), tok_ids
        token_ids = [v[0] for v in tok_ids]
        # Compare pixels for one state of every task and disallow a different visual budget.
        reference = AutoProcessor.from_pretrained(str(ROOT/'OmniJev-4B-v1.1'), max_pixels=256*28*28)
        seen = set()
        for row in rows:
            if row['task_id'] in seen: continue
            seen.add(row['task_id'])
            with Image.open(row['image']) as im: image = im.convert('RGB')
            a = proc.image_processor(images=[image], return_tensors='pt')
            b = reference.image_processor(images=[image], return_tensors='pt')
            assert torch.equal(a['image_grid_thw'], b['image_grid_thw'])
            assert torch.equal(a['pixel_values'], b['pixel_values'])
        del reference, a, b
        run_protocol['pixel_parity_tasks_checked'] = sorted(seen)
        model = AutoModelForImageTextToText.from_pretrained(str(ROOT/'Qwen3.5-4B'), dtype=torch.bfloat16).eval().to('cuda')
        assert not any('lora_' in n for n, _ in model.named_parameters())
    torch.cuda.synchronize()
    run_protocol['model_load_seconds'] = time.perf_counter()-start
    metadata_path.write_text(json.dumps(run_protocol, ensure_ascii=False, indent=2))
    print('MODEL_LOADED', args.mode, round(run_protocol['model_load_seconds'], 2), flush=True)

    def request(row):
        # No label_evidence, source filenames, timestep, or future frames are passed to either model.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.mode == 'omnijev':
            answers = model.system_one({'images': [row['image']]}, row['questions'], packed=True)
            tokens = {'request_total': model.last_input_tokens}
        else:
            with Image.open(row['image']) as im: image = im.convert('RGB')
            chats, qids, options_per_q = [], [], []
            for qid, q in row['questions'].items():
                desc = [o['text'] for o in q['options']] + ['none of the above']
                letters = LETTERS[:len(desc)]
                prompt = q['instructions']+'\n\nOptions:\n'+'\n'.join(f'{c}. {d}' for c, d in zip(letters, desc))
                prompt += '\n\nChoose the single best option. Respond with exactly one of these letters: '+', '.join(letters)+'. Do not add an explanation.'
                messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                chats.append(proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                qids.append(qid)
                options_per_q.append(q['options'])
            enc = proc(text=chats, images=[image]*len(chats), return_tensors='pt', padding=True).to('cuda')
            tokens = {q: int(enc['attention_mask'][i].sum()) for i, q in enumerate(qids)}
            answers = {}
            with torch.inference_mode():
                if args.mode == 'qwen-score':
                    output = model(**enc, use_cache=False, logits_to_keep=1, return_dict=True)
                    for i, (qid, options) in enumerate(zip(qids, options_per_q)):
                        logits = output.logits[i, -1].float()
                        selected = logits[token_ids[:len(options)+1]]
                        probs = torch.softmax(selected, dim=-1).cpu().tolist()
                        k = max(range(len(options)), key=lambda j: probs[j])
                        best = max(range(len(probs)), key=lambda j: probs[j])
                        mass = float(torch.exp(torch.logsumexp(selected, dim=0)-torch.logsumexp(logits, dim=0)))
                        answers[qid] = {'choice': options[k]['key'], 'valid': best < len(options),
                                        'probabilities': {o['key']: p for o, p in zip(options, probs[:-1])},
                                        'abstain': probs[-1], 'answer_letter': LETTERS[best],
                                        'full_vocabulary_letter_mass': mass,
                                        'unrestricted_next_token': proc.tokenizer.decode([int(logits.argmax())])}
                    del output
                else:
                    generated = model.generate(**enc, do_sample=False, max_new_tokens=8,
                                               pad_token_id=proc.tokenizer.pad_token_id)
                    for i, (qid, options) in enumerate(zip(qids, options_per_q)):
                        response = proc.tokenizer.decode(generated[i, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                        allowed = LETTERS[:len(options)+1]
                        format_ok = bool(re.fullmatch('['+allowed+']', response))
                        k = allowed.index(response) if format_ok else -1
                        abstain = format_ok and k == len(options)
                        answers[qid] = {'choice': options[k]['key'] if format_ok and not abstain else None,
                                        'valid': format_ok and not abstain, 'format_ok': format_ok,
                                        'abstained': abstain, 'text': response}
                    del generated
            del enc
        torch.cuda.synchronize()
        return answers, tokens, time.perf_counter()-t0

    dev = [json.loads(s) for s in (OUT/'dev.jsonl').read_text().splitlines()][:3]
    warmups = []
    for i, row in enumerate(dev):
        _, _, seconds = request(row)
        warmups.append(seconds)
        print('WARMUP', i+1, round(seconds, 3), flush=True)
    run_protocol['warmup_seconds'] = warmups
    torch.cuda.reset_peak_memory_stats()
    with pred_path.open('a', buffering=1) as f:
        for i, row in enumerate(rows):
            if row['uid'] in done: continue
            assert sha(row['image']) == row['image_sha256']
            answers, tokens, seconds = request(row)
            assert set(answers) == set(row['questions'])
            correct = {qid: bool(answer.get('valid', False) and answer['choice'] == row['labels'][qid])
                       for qid, answer in answers.items()}
            result = {'mode': args.mode, 'uid': row['uid'], 'task_id': row['task_id'],
                      'episode_uid': row['episode_uid'], 'labels': row['labels'], 'answers': answers,
                      'correct': correct, 'wall_seconds': seconds, 'input_tokens': tokens}
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            print('STATE', i+1, len(rows), row['uid'], correct, round(seconds, 3), flush=True)
    run_protocol['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    run_protocol['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/1024**3
    run_protocol['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    all_preds = [json.loads(s) for s in pred_path.read_text().splitlines()]
    assert {r['uid'] for r in all_preds} == {r['uid'] for r in rows}
    run_protocol['completed_states'] = len(all_preds)
    run_protocol['predictions_sha256'] = sha(pred_path)
    metadata_path.write_text(json.dumps(run_protocol, ensure_ascii=False, indent=2))
    print('COMPLETE', args.mode, len(all_preds), 'states', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['omnijev', 'qwen-score', 'qwen-generate'], required=True)
    p.add_argument('--split', choices=['dev', 'test'], default='test')
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())
