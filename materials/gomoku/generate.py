#!/usr/bin/env python3
"""Create a reproducible, independently labelled freestyle Gomoku evaluation set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 15
BLACK, WHITE = 1, 2
NAMES = {BLACK: "Black", WHITE: "White"}
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
WIDTH, HEIGHT, X0, Y0, STEP = 720, 760, 80, 100, 40


def coordinate(r, c):
    return f"{chr(65+c)}{r+1}"


def completes_five(board, r, c, color):
    """Local neighbour counting; also works on a currently empty candidate point."""
    for dr, dc in DIRECTIONS:
        length = 1
        for sign in [-1, 1]:
            rr, cc = r + sign*dr, c + sign*dc
            while 0 <= rr < N and 0 <= cc < N and board[rr][cc] == color:
                length += 1
                rr, cc = rr + sign*dr, cc + sign*dc
        if length >= 5:
            return True
    return False


def winning_points(board, color):
    return [(r, c) for r in range(N) for c in range(N)
            if board[r][c] == 0 and completes_five(board, r, c, color)]


def already_won(board):
    return any(board[r][c] and completes_five(board, r, c, board[r][c])
               for r in range(N) for c in range(N))


def canonical(board):
    a = np.asarray(board, dtype=np.uint8)
    return min(np.rot90(b, k).tobytes() for b in [a, np.fliplr(a)] for k in range(4))


def threat_template(board, color, reserved, rng):
    """Four contiguous stones, one end blocked, the other end reserved empty."""
    for _ in range(300):
        dr, dc = rng.choice(DIRECTIONS)
        r, c = rng.randrange(N), rng.randrange(N)
        pts = [(r+i*dr, c+i*dc) for i in range(6)]
        if not all(0 <= rr < N and 0 <= cc < N for rr, cc in pts):
            continue
        if any(board[rr][cc] or (rr, cc) in reserved for rr, cc in pts):
            continue
        if rng.randrange(2):
            pts.reverse()
        board[pts[0][0]][pts[0][1]] = 3-color
        for rr, cc in pts[1:5]:
            board[rr][cc] = color
        reserved.add(pts[5])
        return pts[5]
    return None


def fill_board(board, target_counts, reserved, rng):
    for color in rng.sample([BLACK, WHITE], 2):
        needed = target_counts[color] - sum(row.count(color) for row in board)
        if needed < 0:
            return False
        candidates = [(r, c) for r in range(N) for c in range(N)
                      if board[r][c] == 0 and (r, c) not in reserved]
        rng.shuffle(candidates)
        for r, c in candidates:
            if not needed:
                break
            if completes_five(board, r, c, color):
                continue
            board[r][c] = color
            needed -= 1
        if needed:
            return False
    return True


def trajectory(board, rng, final=None):
    pools = {color: [(r, c) for r in range(N) for c in range(N)
                     if board[r][c] == color and (r, c) != final]
             for color in [BLACK, WHITE]}
    for pool in pools.values():
        rng.shuffle(pool)
    history = []
    total = sum(len(p) for p in pools.values())
    for i in range(total):
        color = BLACK if i % 2 == 0 else WHITE
        r, c = pools[color].pop()
        history.append({"ply": i+1, "color": NAMES[color].lower(), "row": r+1,
                        "column": c+1, "coordinate": coordinate(r, c)})
    if final is not None:
        r, c = final
        color = BLACK if total % 2 == 0 else WHITE
        assert board[r][c] == color
        history.append({"ply": total+1, "color": NAMES[color].lower(), "row": r+1,
                        "column": c+1, "coordinate": coordinate(r, c)})
    return history


def build_position(spec, rng):
    counts = {BLACK: spec["m"], WHITE: spec["m"] if spec["turn"] == BLACK else spec["m"]-1}
    for attempt in range(1000):
        board = [[0]*N for _ in range(N)]
        reserved = set()
        final = None
        if spec["terminal"]:
            # Positive and negative pairs have identical black/white counts.
            winner = WHITE if spec["turn"] == BLACK else BLACK
            final = threat_template(board, winner, reserved, rng)
            if final is None or already_won(board):
                continue
            pre_counts = dict(counts)
            pre_counts[winner] -= 1
            if not fill_board(board, pre_counts, reserved, rng):
                continue
            board[final[0]][final[1]] = winner
        else:
            good = True
            for color, wanted in [(spec["turn"], spec["can_win"]),
                                  (3-spec["turn"], spec["opponent_threat"])]:
                if wanted and threat_template(board, color, reserved, rng) is None:
                    good = False
                    break
            if not good or already_won(board) or not fill_board(board, counts, reserved, rng):
                continue
            own = winning_points(board, spec["turn"])
            opponent = winning_points(board, 3-spec["turn"])
            if bool(own) != spec["can_win"] or bool(opponent) != spec["opponent_threat"]:
                continue
            if spec["opponent_threat"] and len(opponent) != 1:
                continue  # A unique blocking point avoids ambiguous multi-threat labels.
        history = trajectory(board, rng, final)
        return board, history, counts
    raise RuntimeError(f"Could not construct requested stratum: {spec}")


def load_font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("A TrueType font is required for legible board coordinates")


def render(board, style, path):
    palettes = [("#f3e9d8", "#e8c58c"), ("#f3efdf", "#f0ddb7"),
                ("#f5f5f3", "#eeebe2")]
    outer, wood = palettes[style]
    image = Image.new("RGB", (WIDTH, HEIGHT), outer)
    d = ImageDraw.Draw(image)
    small, header = load_font(21), load_font(25)
    d.text((WIDTH//2, 27), "GOMOKU  |  15 x 15", font=header, fill="#29231e", anchor="mm")
    d.rectangle((X0-25, Y0-25, X0+14*STEP+25, Y0+14*STEP+25), fill=wood)
    for i in range(N):
        d.line((X0, Y0+i*STEP, X0+14*STEP, Y0+i*STEP), fill="#75614c", width=1)
        d.line((X0+i*STEP, Y0, X0+i*STEP, Y0+14*STEP), fill="#75614c", width=1)
        d.text((X0+i*STEP, Y0-43), chr(65+i), font=small, fill="#29231e", anchor="mm")
        d.text((X0-44, Y0+i*STEP), str(i+1), font=small, fill="#29231e", anchor="mm")
    for r, row in enumerate(board):
        for c, color in enumerate(row):
            if not color:
                continue
            x, y = X0+c*STEP, Y0+r*STEP
            fill = "#171717" if color == BLACK else "#fafafa"
            d.ellipse((x-16, y-16, x+16, y+16), fill=fill, outline="#353535", width=2)
    d.text((WIDTH//2, 710), "Columns A-O  |  Rows 1-15", font=small, fill="#29231e", anchor="mm")
    image.save(path)


def balanced_slots(n, rng):
    out = [i % 4 for i in range(n)]
    rng.shuffle(out)
    return out


def options_for_counts(count, slot, rng):
    nearby = [x for x in range(max(0, count-5), count+6) if x != count]
    other = rng.sample(nearby, 3)
    other.insert(slot, count)
    return [{"key": str(x), "text": f"{x} stones"} for x in other]


def move_options(board, correct_points, slot, rng):
    correct = rng.choice(correct_points)
    negatives = [(r, c) for r in range(N) for c in range(N)
                 if board[r][c] == 0 and (r, c) not in correct_points]
    # Prefer distractors near occupied points, rather than only distant trivial cells.
    near = [(r, c) for r, c in negatives
            if any(board[rr][cc] for rr in range(max(0, r-2), min(N, r+3))
                   for cc in range(max(0, c-2), min(N, c+3)))]
    pool = near if len(near) >= 3 else negatives
    candidates = rng.sample(pool, 3)
    candidates.insert(slot, correct)
    opts = [{"key": coordinate(r, c), "text": f"Place a stone at {coordinate(r,c)} "
             f"(column {chr(65+c)}, row {r+1})."} for r, c in candidates]
    return opts, coordinate(*correct)


def make_questions(rec, counters, slots, rng):
    board, meta = rec["board"], rec["metadata"]
    questions, answers = {}, {}

    def binary(qid, instruction, truth):
        questions[qid] = {"type": "noul", "instructions": instruction}
        answers[qid] = {"value": bool(truth), "label": int(truth), "label_text": "yes" if truth else "no"}

    def choice(qid, instruction, options, value, slot):
        questions[qid] = {"type": "choice", "instructions": instruction, "options": options}
        answers[qid] = {"value": value, "label_text": value, "gold_index": slot}

    binary("has_five", "Freestyle Gomoku: five or more consecutive stones of the same color "
           "in a horizontal, vertical, or diagonal line is a win. Does this board contain "
           "such a line for either Black or White?", meta["terminal"])
    for color in [BLACK, WHITE]:
        qid = NAMES[color].lower()+"_count"
        slot = slots[qid][counters[qid]]
        counters[qid] += 1
        count = meta["stone_counts"][NAMES[color].lower()]
        choice(qid, f"How many {NAMES[color].lower()} stones are visible on the Gomoku board?",
               options_for_counts(count, slot, rng), str(count), slot)
    if not meta["terminal"]:
        turn = BLACK if rec["side_to_move"] == "black" else WHITE
        own, opponent = winning_points(board, turn), winning_points(board, 3-turn)
        meta["winning_points"] = [coordinate(*p) for p in own]
        meta["opponent_winning_points"] = [coordinate(*p) for p in opponent]
        binary("can_win_now", f"Freestyle Gomoku. {NAMES[turn]} is to move. Can {NAMES[turn]} "
               "form a line of five or more stones by placing exactly one stone on an empty point?", bool(own))
        binary("opponent_has_immediate_win", f"Freestyle Gomoku. {NAMES[turn]} is to move. "
               f"Considering the current board, could {NAMES[3-turn]} form a line of five or more "
               "stones if allowed to place one stone on an empty point?", bool(opponent))
        for qid, points in [("winning_move", own), ("blocking_move", opponent)]:
            if not points:
                continue
            slot = slots[qid][counters[qid]]
            counters[qid] += 1
            options, key = move_options(board, points, slot, rng)
            if qid == "winning_move":
                instruction = f"Freestyle Gomoku. {NAMES[turn]} is to move. Which offered empty "
                instruction += f"point lets {NAMES[turn]} form five or more in a row immediately?"
            else:
                instruction = f"Freestyle Gomoku. {NAMES[turn]} is to move. Which offered empty "
                instruction += f"point, when occupied by {NAMES[turn]}, removes all of {NAMES[3-turn]}'s "
                instruction += "one-move five-in-a-row completion points? Evaluate only the opponent's lines, "
                instruction += "regardless of whether your own move also wins."
            choice(qid, instruction, options, key, slot)
    rec["questions"], rec["answers"] = questions, answers


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError(f"Output directory must be empty; refusing to overwrite {root}")
    (root/"images").mkdir()
    rng = random.Random(args.seed)
    strata = [(True, True)]*38 + [(True, False)]*37 + [(False, True)]*37 + [(False, False)]*38
    rng.shuffle(strata)
    turns = [BLACK]*75 + [WHITE]*75
    rng.shuffle(turns)
    specs = []
    for i, ((can_win, threat), turn) in enumerate(zip(strata, turns)):
        common = {"pair_id": f"pair_{i:03d}", "m": rng.randint(6, 26), "turn": turn, "style": i % 3}
        specs.append(dict(common, terminal=False, can_win=can_win, opponent_threat=threat))
        specs.append(dict(common, terminal=True, can_win=False, opponent_threat=False))
    rng.shuffle(specs)
    records, seen = [], set()
    for i, spec in enumerate(specs):
        for _ in range(100):
            board, history, counts = build_position(spec, rng)
            key = canonical(board)
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("Duplicate-board construction limit exceeded")
        record_id = f"gmv1_{i:04d}"
        winner = (WHITE if spec["turn"] == BLACK else BLACK) if spec["terminal"] else None
        rec = {"id": record_id, "game_id": record_id, "family": "gomoku_custom_v1", "split": "test",
               "images": [f"images/{record_id}.png"], "board": board, "moves": history,
               "side_to_move": None if spec["terminal"] else NAMES[spec["turn"]].lower(),
               "metadata": {"pair_id": spec["pair_id"], "terminal": spec["terminal"],
                            "winner": NAMES[winner].lower() if winner else None,
                            "stone_counts": {NAMES[c].lower(): counts[c] for c in [BLACK, WHITE]},
                            "style": spec["style"], "rules": "freestyle_five_or_more_no_forbidden_moves"}}
        records.append(rec)
        if (i+1) % 50 == 0:
            print(f"Constructed {i+1}/300 reachable positions", flush=True)
    slots = {k: balanced_slots(n, rng) for k, n in
             [("black_count", 300), ("white_count", 300), ("winning_move", 75), ("blocking_move", 75)]}
    counters = Counter()
    for rec in records:
        make_questions(rec, counters, slots, rng)
        render(rec["board"], rec["metadata"]["style"], root/rec["images"][0])
    with (root/"records.jsonl").open("w") as f, (root/"omnijev_inputs.jsonl").open("w") as api:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
            inp = {"id": rec["id"], "state": {"images": [str(root/p) for p in rec["images"]]},
                   "questions": rec["questions"]}
            api.write(json.dumps(inp, ensure_ascii=False)+"\n")
    summary = {"records": len(records), "questions": 0, "terminal": Counter(),
               "winner": Counter(), "question_counts": Counter(), "binary_labels": defaultdict(Counter),
               "choice_gold_indices": defaultdict(Counter)}
    for rec in records:
        summary["terminal"][str(rec["metadata"]["terminal"]).lower()] += 1
        if rec["metadata"]["winner"]:
            summary["winner"][rec["metadata"]["winner"]] += 1
        for qid, gold in rec["answers"].items():
            summary["questions"] += 1
            summary["question_counts"][qid] += 1
            if "label" in gold:
                summary["binary_labels"][qid][str(gold["label"])] += 1
            else:
                summary["choice_gold_indices"][qid][str(gold["gold_index"])] += 1
    write_json(root/"statistics.json", summary)
    sources = root/"provenance"
    sources.mkdir()
    for file in [Path(__file__), Path(__file__).with_name("validate.py")]:
        if file.exists():
            shutil.copy2(file, sources/file.name)
    manifest = {"dataset": "gomoku-eval-v1", "seed": args.seed, "board_size": N,
                "purpose": "New synthetic image benchmark; not the original OmniJev evaluation set",
                "board_encoding": {"0": "empty", "1": "black", "2": "white"},
                "binary_answer_encoding": {"0": "no/false", "1": "yes/true"},
                "coordinate_system": "columns A-O left to right; rows 1-15 top to bottom",
                "records": 300, "questions": summary["questions"], "image_size": [WIDTH, HEIGHT],
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(root.rglob("*")) if p.is_file()}}
    write_json(root/"manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
