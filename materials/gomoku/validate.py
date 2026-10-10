#!/usr/bin/env python3
"""Independent sliding-window verification; does not import generator logic."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 15


def batch_wins(boards, color):
    a = np.asarray(boards)
    if a.ndim == 2:
        a = a[None]
    masks = [
        np.logical_and.reduce([a[:, :, k:k+N-4] == color for k in range(5)]),
        np.logical_and.reduce([a[:, k:k+N-4, :] == color for k in range(5)]),
        np.logical_and.reduce([a[:, k:k+N-4, k:k+N-4] == color for k in range(5)]),
        np.logical_and.reduce([a[:, k:k+N-4, 4-k:N-k] == color for k in range(5)]),
    ]
    return np.logical_or.reduce([mask.any(axis=(1, 2)) for mask in masks])


def winning_moves(a, color):
    cells = np.argwhere(a == 0)
    b = np.repeat(a[None], len(cells), axis=0)
    b[np.arange(len(cells)), cells[:, 0], cells[:, 1]] = color
    wins = batch_wins(b, color)
    return {(int(r), int(c)) for r, c in cells[wins]}


def coord(r, c):
    return f"{chr(65+c)}{r+1}"


def decode(s):
    return int(s[1:])-1, ord(s[0])-65


def canonical(a):
    return min(np.rot90(b, k).tobytes() for b in [a, np.fliplr(a)] for k in range(4))


def validate_primitive_rules():
    """Exercise the independent oracle on all four directions and an overline."""
    for dr, dc, r, c in [(0, 1, 4, 3), (1, 0, 3, 4), (1, 1, 3, 3), (1, -1, 3, 9)]:
        a = np.zeros((N, N), dtype=np.uint8)
        for i in range(4):
            a[r+i*dr, c+i*dc] = 1
        assert not bool(batch_wins(a, 1)[0])
        assert (r+4*dr, c+4*dc) in winning_moves(a, 1)
        a[r+4*dr, c+4*dc] = 1
        assert bool(batch_wins(a, 1)[0])
        a[r+5*dr, c+5*dc] = 1
        assert bool(batch_wins(a, 1)[0])
        assert not bool(batch_wins(a, 2)[0])
    a = np.zeros((N, N), dtype=np.uint8)
    a[2, 2:6] = 1
    a[2, 1] = 2
    assert winning_moves(a, 1) == {(2, 6)}


def contact_sheet(records, root):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    wanted = [r for r in records if r['metadata']['terminal'] and r['metadata']['winner'] == 'black'][:1]
    wanted += [r for r in records if r['metadata']['terminal'] and r['metadata']['winner'] == 'white'][:1]
    for own, threat in [(True, True), (True, False), (False, True), (False, False)]:
        wanted += [r for r in records if not r['metadata']['terminal']
                   and r['answers']['can_win_now']['value'] == own
                   and r['answers']['opponent_has_immediate_win']['value'] == threat][:1]
    sheet = Image.new('RGB', (1080, 865), '#ffffff')
    d = ImageDraw.Draw(sheet)
    for i, rec in enumerate(wanted):
        with Image.open(root/rec['images'][0]) as image:
            tile = image.resize((360, 380))
        x, y = (i % 3)*360, (i//3)*420
        sheet.paste(tile, (x, y))
        label = f"{rec['id']} | five={rec['answers']['has_five']['value']}"
        d.text((x+8, y+385), label, font=font, fill='#222222')
        if not rec['metadata']['terminal']:
            label = f"win-now={rec['answers']['can_win_now']['value']} / threat={rec['answers']['opponent_has_immediate_win']['value']}"
            d.text((x+8, y+405), label, font=font, fill='#222222')
    sheet.save(root/'preview.png')
    return [r['id'] for r in wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    root = Path(ap.parse_args().root).resolve()
    validate_primitive_rules()
    records = [json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    inputs = [json.loads(line) for line in (root/'omnijev_inputs.jsonl').read_text().splitlines()]
    assert len(records) == len(inputs) == 300
    seen, games = set(), set()
    binary = defaultdict(Counter)
    choices = defaultdict(Counter)
    questions, moves = 0, 0
    pairs = defaultdict(list)
    counts_by_terminal = defaultdict(Counter)
    colors = {'black': 1, 'white': 2}
    for i, (rec, inp) in enumerate(zip(records, inputs)):
        a = np.asarray(rec['board'], dtype=np.uint8)
        assert a.shape == (N, N) and np.isin(a, [0, 1, 2]).all()
        key = canonical(a)
        assert key not in seen, f"Symmetry duplicate: {rec['id']}"
        seen.add(key)
        assert rec['game_id'] not in games
        games.add(rec['game_id'])
        meta = rec['metadata']
        counts = {name: int((a == color).sum()) for name, color in colors.items()}
        assert counts == meta['stone_counts']
        counts_by_terminal[meta['terminal']][tuple(counts.values())] += 1
        pairs[meta['pair_id']].append(rec)
        replay = np.zeros((N, N), dtype=np.uint8)
        for j, move in enumerate(rec['moves']):
            r, c = move['row']-1, move['column']-1
            color = 1 if j % 2 == 0 else 2
            assert move['ply'] == j+1 and colors[move['color']] == color
            assert 0 <= r < N and 0 <= c < N and replay[r, c] == 0
            assert move['coordinate'] == coord(r, c)
            replay[r, c] = color
            won = bool(batch_wins(replay, color)[0])
            assert won == (meta['terminal'] and j == len(rec['moves'])-1), (rec['id'], j, 'early or missing win')
            moves += 1
        assert np.array_equal(replay, a)
        winners = {color for color in [1, 2] if bool(batch_wins(a, color)[0])}
        assert bool(winners) == meta['terminal']
        if meta['terminal']:
            assert winners == {colors[meta['winner']]} and rec['side_to_move'] is None
        else:
            turn = colors[rec['side_to_move']]
            assert (counts['black'] == counts['white']) == (turn == 1)
            own, opponent = winning_moves(a, turn), winning_moves(a, 3-turn)
            assert {coord(*p) for p in own} == set(meta['winning_points'])
            assert {coord(*p) for p in opponent} == set(meta['opponent_winning_points'])
            assert len(opponent) <= 1
        assert set(rec['questions']) == set(rec['answers'])
        expected_qids = {'has_five', 'black_count', 'white_count'}
        if not meta['terminal']:
            expected_qids |= {'can_win_now', 'opponent_has_immediate_win'}
            if own: expected_qids.add('winning_move')
            if opponent: expected_qids.add('blocking_move')
        assert set(rec['questions']) == expected_qids
        for qid, q in rec['questions'].items():
            gold = rec['answers'][qid]
            questions += 1
            if q['type'] == 'noul':
                truth = bool(winners) if qid == 'has_five' else bool(own if qid == 'can_win_now' else opponent)
                assert gold['value'] == truth and gold['label'] == int(truth)
                assert gold['label_text'] == ('yes' if truth else 'no')
                binary[qid][str(gold['label'])] += 1
            else:
                keys = [o['key'] for o in q['options']]
                assert len(keys) == len(set(keys)) == 4
                assert gold['value'] == keys[gold['gold_index']]
                choices[qid][str(gold['gold_index'])] += 1
                if qid.endswith('_count'):
                    assert int(gold['value']) == counts[qid.replace('_count', '')]
                    correct = [k == gold['value'] for k in keys]
                else:
                    candidates = [decode(k) for k in keys]
                    assert all(0 <= r < N and 0 <= c < N and a[r, c] == 0 for r, c in candidates)
                    if qid == 'winning_move':
                        correct = [(r, c) in own for r, c in candidates]
                    else:
                        correct = []
                        for r, c in candidates:
                            placed = a.copy()
                            placed[r, c] = turn
                            correct.append(not winning_moves(placed, 3-turn))
                assert sum(correct) == 1 and correct[gold['gold_index']]
        assert inp['id'] == rec['id'] and inp['questions'] == rec['questions']
        assert inp['state']['images'] == [str(root/p) for p in rec['images']]
        for relative in rec['images']:
            p = root/relative
            with Image.open(p) as image:
                assert image.size == (720, 760) and image.mode == 'RGB'
                image.verify()
        if (i+1) % 50 == 0:
            print(f"Independently validated {i+1}/300 boards and their trajectories", flush=True)
    assert counts_by_terminal[True] == counts_by_terminal[False], 'Count-distribution shortcut detected'
    assert len(pairs) == 150 and all(len(p) == 2 and {r['metadata']['terminal'] for r in p} == {True, False}
                                     for p in pairs.values())
    assert binary['has_five'] == {'0': 150, '1': 150}
    assert binary['can_win_now'] == binary['opponent_has_immediate_win'] == {'0': 75, '1': 75}
    for qid in ['black_count', 'white_count']:
        assert choices[qid] == {str(k): 75 for k in range(4)}
    for qid in ['winning_move', 'blocking_move']:
        assert sum(choices[qid].values()) == 75 and max(choices[qid].values())-min(choices[qid].values()) <= 1
    assert questions == 1350
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    selected = contact_sheet(records, root)
    report = {'status': 'passed', 'oracle': 'Independent NumPy length-five sliding windows; no generator imports',
              'records': 300, 'questions': questions, 'trajectory_moves_checked': moves,
              'unique_boards_up_to_rotation_and_reflection': len(seen),
              'exact_matching_stone_counts_between_finished_and_unfinished': True,
              'all_multiple_choice_questions_have_one_correct_offered_answer': True,
              'all_images_decoded_and_all_initial_manifest_hashes_verified': True,
              'binary_labels': binary, 'choice_gold_indices': choices,
              'preview_record_ids': selected, 'model_inference_run': False}
    (root/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
