#!/usr/bin/env python3
"""Frozen paired Gomoku diagnostic, with independently scored resumable predictions."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
DATA = ROOT / 'gomoku-eval-v1'
OUT = ROOT / 'OmniJev-experiments/gomoku-compare-v1-20261010'
PIXELS = 768 * 28 * 28

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def summary(name, predictions, total, extra=None):
    result = {'model': name, 'completed_questions': len(predictions), 'requested_questions': total,
              'complete': len(predictions) == total, 'by_task': {}}
    for qid in dict.fromkeys(p['qid'] for p in predictions):
        rows = [p for p in predictions if p['qid'] == qid]
        n = len(rows)
        metrics = {'correct': sum(p['correct'] for p in rows), 'total': n,
                   'accuracy_percent': 100 * sum(p['correct'] for p in rows) / n,
                   'forced_choice_accuracy_percent': 100 * sum(p['forced_correct'] for p in rows) / n,
                   'abstentions': sum(not p['valid'] for p in rows)}
        if rows[0]['question_type'] == 'noul':
            cm = {f'{g}{p}': sum(int(r['gold']) == g and int(r['prediction']) == p for r in rows)
                  for g in (0, 1) for p in (0, 1)}
            metrics.update({'confusion_matrix_gold_then_prediction': cm,
                            'yes_recall_percent': 100*cm['11']/max(1, cm['10']+cm['11']),
                            'no_recall_percent': 100*cm['00']/max(1, cm['00']+cm['01']),
                            'predicted_label_counts': dict(Counter(str(r['prediction']) for r in rows))})
        result['by_task'][qid] = metrics
    result['micro_accuracy_percent'] = 100*sum(p['correct'] for p in predictions)/len(predictions) if predictions else None
    result['macro_task_accuracy_percent'] = statistics.mean(x['accuracy_percent'] for x in result['by_task'].values()) if predictions else None
    if extra:
        result.update(extra)
    return result

def evaluate(name, limit):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    sys.path.insert(0, '/data/users/sunziang/OmniJev')
    from mso.fast_kernels import enable_fla
    from mso.infer import MSO1
    assert torch.cuda.is_available(), 'No GPU; CPU fallback disabled'
    torch.set_num_threads(8)
    torch.manual_seed(20261010)
    enable_fla()
    # Each input patch fills the entire kernel and yields one output, so the
    # released Conv3d is exactly a flattened affine matrix product.
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    import torch.nn.functional as F
    def patch_embedding(self, hidden_states):
        width = self.in_channels*self.temporal_patch_size*self.patch_size*self.patch_size
        x = hidden_states.reshape(-1, width).to(dtype=self.proj.weight.dtype)
        if not getattr(self, '_equivalence_checked', False):
            xx = x[:8].detach().float().cpu()
            ww = self.proj.weight.detach().float().cpu()
            bb = self.proj.bias.detach().float().cpu() if self.proj.bias is not None else None
            reference = F.conv3d(xx.reshape(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size), ww, bb, stride=self.proj.stride).reshape(-1, self.embed_dim)
            flat = F.linear(xx, ww.reshape(self.embed_dim, -1), bb)
            torch.testing.assert_close(flat, reference, rtol=1e-4, atol=1e-4)
            print('PATCH_EQUIVALENCE_CHECK', name, float((flat-reference).abs().max()), flush=True)
            self._equivalence_checked = True
        return F.linear(x, self.proj.weight.reshape(self.embed_dim, -1), self.proj.bias)
    qm.Qwen3_5VisionPatchEmbed.forward = patch_embedding
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(s) for s in (DATA/'questions.jsonl').read_text().splitlines()]
    inputs = [json.loads(s) for s in (DATA/'omnijev_inputs.jsonl').read_text().splitlines()]
    assert len(all_rows) == 1350 and len(inputs) == 300
    labels = {r['id']: r for r in all_rows}
    for state in inputs:
        for qid, q in state['questions'].items():
            assert q == labels[state['id']+'/'+qid]['question']
    manifest = json.loads((DATA/'manifest.json').read_text())
    for image in (DATA/'images').glob('*.png'):
        assert sha(image) == manifest['files_sha256'][str(image.relative_to(DATA))]
    protocol = {
        'model': name, 'dataset': str(DATA), 'questions_sha256': sha(DATA/'questions.jsonl'),
        'inputs_sha256': sha(DATA/'omnijev_inputs.jsonl'), 'script_sha256': sha(Path(__file__)),
        'code_commit': subprocess.check_output(['git', '-C', '/data/users/sunziang/OmniJev', 'rev-parse', 'HEAD'], text=True).strip(),
        'model_manifest': json.loads((ROOT/('Qwen3.5-4B' if name == 'qwen' else 'OmniJev-4B-v1.1')/'download_manifest.json').read_text()),
        'max_pixels': PIXELS, 'dtype': 'bfloat16', 'seed': 20261010,
        'gpu': torch.cuda.get_device_name(), 'torch_cpu_threads': torch.get_num_threads(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'Pillow', 'fla-core']},
        'requested_boards': 300, 'requested_questions': 1350,
        'binary_rule': 'P(yes) >= 0.5; label 1=yes, 0=no',
        'choice_rule': 'Highest probability among four candidates and abstain; abstain counts wrong. Also report forced choice among four candidates.',
        'qwen_method': 'Original model; non-thinking next-token answer-letter probabilities. Binary A=yes/B=no; choices A-D plus E=none of the above. No generated reasoning. Probabilities conditional on these letters, not calibrated correctness confidence.',
        'omnijev_method': 'Original MSO1.system_one branch API with checkpoint head, LoRA, and released calibration. Grouped image requests use isolated question branches, not a joint prompt.',
        'scope': 'Custom balanced synthetic Gomoku set; not official benchmark reproduction; no training or prompt tuning on labels.',
        'latency_note': 'Qwen individual prefills versus OmniJev shared image prefix; wall time is not a controlled architecture comparison.',
        'patch_embedding': 'Both models: mathematically equivalent flattened F.linear instead of whole-patch Conv3d. FP32 equivalence checked on eight actual input patches at rtol=atol=1e-4. BF16 kernel accumulation can differ.',
    }
    protocol_file = OUT/(name+'-protocol.json')
    if protocol_file.exists():
        old = json.loads(protocol_file.read_text())
        for key in ['questions_sha256', 'inputs_sha256', 'script_sha256', 'max_pixels', 'qwen_method', 'omnijev_method']:
            assert old[key] == protocol[key], ('Resume protocol changed', key)
    dump(protocol_file, protocol)
    pred_file = OUT/(name+'-predictions.jsonl')
    predictions = [json.loads(s) for s in pred_file.read_text().splitlines()] if pred_file.exists() else []
    done = {p['id'] for p in predictions}
    assert len(done) == len(predictions) and done <= set(labels)
    for p in predictions:
        assert p['gold'] == labels[p['id']]['answer']['value']
    states = inputs[:limit] if limit else inputs
    start = time.perf_counter()
    if name == 'omnijev':
        model = MSO1(str(ROOT/'OmniJev-4B-v1.1'), str(ROOT/'Qwen3.5-4B'), max_pixels=PIXELS)
        assert model.branch
        protocol['calibration'] = {'temperatures': model.temps, 'biases': model.biases}
        dump(protocol_file, protocol)
    else:
        proc = AutoProcessor.from_pretrained(str(ROOT/'Qwen3.5-4B'), max_pixels=PIXELS)
        reference = AutoProcessor.from_pretrained(str(ROOT/'OmniJev-4B-v1.1'), max_pixels=PIXELS)
        # Check actual pixel tensors for every input image, not only processor config.
        for state in inputs:
            with Image.open(state['state']['images'][0]) as im:
                image = im.convert('RGB')
            a = proc.image_processor(images=[image], return_tensors='pt')
            b = reference.image_processor(images=[image], return_tensors='pt')
            assert torch.equal(a['pixel_values'], b['pixel_values'])
            assert torch.equal(a['image_grid_thw'], b['image_grid_thw'])
        protocol['pixel_preprocessing_parity_verified_boards'] = 300
        dump(protocol_file, protocol)
        del reference, a, b
        letters = 'ABCDE'
        ids = [proc.tokenizer.encode(c, add_special_tokens=False) for c in letters]
        assert all(len(x) == 1 for x in ids)
        token_ids = [x[0] for x in ids]
        model = AutoModelForImageTextToText.from_pretrained(str(ROOT/'Qwen3.5-4B'), dtype=torch.bfloat16).eval().to('cuda')
        assert not any('lora_' in n for n, _ in model.named_parameters())
    torch.cuda.synchronize()
    print('MODEL_LOADED', name, round(time.perf_counter()-start, 2), flush=True)
    torch.cuda.reset_peak_memory_stats()
    with pred_file.open('a', buffering=1) as f:
        for i, state in enumerate(states):
            questions = {qid: q for qid, q in state['questions'].items() if state['id']+'/'+qid not in done}
            if not questions:
                continue
            torch.cuda.synchronize()
            board_start = time.perf_counter()
            if name == 'omnijev':
                answers = model.system_one(state['state'], questions, packed=True)
                assert set(answers) == set(questions)
            else:
                answers = {}
                with Image.open(state['state']['images'][0]) as im:
                    image = im.convert('RGB')
                for qid, q in questions.items():
                    descriptions = ['Yes', 'No'] if q['type'] == 'noul' else [o['text'] for o in q['options']]+['None of the above']
                    candidates = letters[:len(descriptions)]
                    prompt = q['instructions']+'\n\nOptions:\n'+'\n'.join(f'{c}. {d}' for c, d in zip(candidates, descriptions))+'\n\nRespond with exactly one letter: '+', '.join(candidates)+'. Do not add an explanation.'
                    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
                    chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    enc = proc(text=[chat], images=[image], return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        output = model(**enc, use_cache=False, logits_to_keep=1, return_dict=True)
                        logits = output.logits[0, -1].float()
                        chosen_logits = logits[token_ids[:len(candidates)]]
                        probs = torch.softmax(chosen_logits, -1).cpu().tolist()
                        letter_mass = float(torch.exp(torch.logsumexp(chosen_logits, 0)-torch.logsumexp(logits, 0)))
                        top = int(logits.argmax())
                    assert all(math.isfinite(p) for p in probs) and abs(sum(probs)-1) < 1e-5
                    answer = {'full_vocabulary_letter_mass': letter_mass,
                              'unrestricted_next_token': proc.tokenizer.decode([top]),
                              'unrestricted_next_token_is_candidate': top in token_ids[:len(candidates)],
                              'input_tokens': int(enc['input_ids'].shape[1])}
                    if q['type'] == 'noul':
                        answer['noul'] = probs[0]
                    else:
                        best = max(range(4), key=lambda j: probs[j])
                        answer.update({'choice': q['options'][best]['key'],
                                       'probabilities': {o['key']: p for o, p in zip(q['options'], probs[:4])},
                                       'abstain': probs[4], 'valid': probs[best] > probs[4]})
                    answers[qid] = answer
                    del enc, output, logits, chosen_logits
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-board_start
            for qid, q in questions.items():
                answer = answers[qid]
                uid = state['id']+'/'+qid
                gold = labels[uid]['answer']['value']
                if q['type'] == 'noul':
                    assert 0 <= answer['noul'] <= 1
                    forced = bool(answer['noul'] >= 0.5)
                    valid = True
                else:
                    assert set(answer['probabilities']) == {o['key'] for o in q['options']}
                    assert all(0 <= p <= 1 and math.isfinite(p) for p in answer['probabilities'].values())
                    forced = answer['choice']
                    valid = bool(answer['valid'])
                row = {'id': uid, 'record_id': state['id'], 'qid': qid, 'question_type': q['type'],
                       'gold': gold, 'prediction': forced if valid else 'abstain', 'forced_prediction': forced,
                       'valid': valid, 'correct': bool(valid and forced == gold), 'forced_correct': bool(forced == gold),
                       'answer': answer, 'board_wall_seconds': elapsed}
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
                predictions.append(row)
                done.add(uid)
            print('BOARD', name, i+1, len(states), 'QUESTIONS', len(predictions), 'SECONDS', round(elapsed, 3), flush=True)
            if (i+1) % 10 == 0:
                dump(OUT/(name+'-summary.json'), summary(name, predictions, len(all_rows)))
    result = summary(name, predictions, len(all_rows), {
        'invocation_wall_seconds': time.perf_counter()-start,
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3})
    dump(OUT/(name+'-summary.json'), result)
    print('SUMMARY', json.dumps(result), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['qwen', 'omnijev'], required=True)
    ap.add_argument('--limit-boards', type=int, default=0)
    args = ap.parse_args()
    evaluate(args.model, args.limit_boards)
