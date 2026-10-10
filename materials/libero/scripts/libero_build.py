#!/usr/bin/env python3
"""Freeze an explicit custom LIBERO-10 diagnostic; does not run model inference."""
import collections
import hashlib
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/ssd-public/ai_infra/public/share/users/ziangsun')
DATA = ROOT/'datasets/LIBERO'
OUT = ROOT/'OmniJev-experiments/libero10-paired-v1-seed20261010'
SEED = 20261010
HORIZON = 5
STILL_METRES = .01
DIRECTION_KEYS = ['forward', 'backward', 'left', 'right', 'up', 'down', 'stay_still']
DIRECTION_TEXT = ['forward, away from the workspace camera', 'backward, toward the workspace camera',
                  'left in the workspace camera image', 'right in the workspace camera image',
                  'up in the workspace camera image', 'down in the workspace camera image',
                  'stay still (less than 1 cm on every camera-frame axis)']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotation(quat):
    w, x, y, z = np.asarray(quat, dtype=float)/np.linalg.norm(quat)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def camera_geometry(xml):
    tree = ET.fromstring(xml)
    cam = tree.find('./worldbody/camera[@name="agentview"]')
    if cam is None:
        raise ValueError('Agentview is not a worldbody camera; parent transform must be handled')
    assert cam.get('mode', 'fixed') == 'fixed'
    position = np.array([float(v) for v in cam.get('pos').split()])
    quat = [float(v) for v in cam.get('quat').split()]
    return position, rotation(quat), float(cam.get('fovy', '45'))

def classify(delta_world, rot):
    # MuJoCo camera looks along -Z; +X is image right, +Y image up, +Z toward viewer.
    delta = rot.T @ delta_world
    axis = int(np.argmax(np.abs(delta)))
    if float(np.max(np.abs(delta))) < STILL_METRES:
        key = 'stay_still'
    else:
        key = {0: ('left', 'right'), 1: ('down', 'up'), 2: ('forward', 'backward')}[axis][int(delta[axis] > 0)]
    ordered = sorted(np.abs(delta), reverse=True)
    return key, delta.tolist(), float(ordered[0]-ordered[1])

def project(pos, cam_pos, rot, fovy, width=128, height=128):
    p = rot.T @ (pos-cam_pos)
    assert p[2] < 0, p
    focal = .5*height/np.tan(np.deg2rad(fovy/2))
    return float(width/2 + focal*p[0]/(-p[2])), float(height/2 - focal*p[1]/(-p[2]))

def main():
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Frozen output exists; use it rather than silently replacing the test set')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    rng = random.Random(SEED)
    records = {'dev': [], 'test': []}
    audit = []
    episode_split = []
    command_audit = collections.defaultdict(list)
    for task_idx, file in enumerate(sorted((DATA/'libero_10').glob('*.hdf5'))):
        with h5py.File(file, 'r') as f:
            data = f['data']
            assert data.attrs['macros_image_convention'] == 'opengl'
            info = json.loads(data.attrs['problem_info'])
            instruction = info['language_instruction']
            names = sorted(data.keys(), key=lambda n: int(n.split('_')[-1]))
            rng.shuffle(names)
            assignment = {'dev': names[:2], 'test': names[2:7]}
            assert not set(assignment['dev']) & set(assignment['test'])
            episode_split.append({'task_id': task_idx, 'file': file.name, **assignment})
            for split, demos in assignment.items():
                for demo in demos:
                    g = data[demo]
                    pos = g['obs/ee_pos'][:]
                    actions = g['actions'][:]
                    fingers = g['obs/gripper_states'][:]
                    cam_pos, rot, fovy = camera_geometry(g.attrs['model_file'])
                    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-6)
                    assert set(np.unique(actions[:, 6])) <= {-1., 1.}
                    # Check next-command signs against subsequent changes in finger opening on dev only.
                    if split == 'dev':
                        widths = fingers[:, 0]-fingers[:, 1]
                        for t in range(len(actions)-5):
                            command_audit[str(int(actions[t+1, 6]))].append(float(widths[t+5]-widths[t]))
                    edges = np.linspace(5, len(pos)-HORIZON, 6, dtype=int)
                    for bin_idx in range(5):
                        t = rng.randrange(int(edges[bin_idx]), int(edges[bin_idx+1]))
                        uid = f'task{task_idx:02d}-{demo}-t{t:04d}'
                        direction, delta, margin = classify(pos[t+HORIZON]-pos[t], rot)
                        gripper = 'open' if actions[t+1, 6] < 0 else 'close'
                        paths = []
                        for sensor in ['agentview_rgb', 'eye_in_hand_rgb']:
                            path = OUT/'images'/f'{uid}-{sensor}.png'
                            # OpenGL origin is lower left. Convert once to upright display convention.
                            Image.fromarray(g[f'obs/{sensor}'][t][::-1].copy()).save(path)
                            paths.append(str(path))
                        # Use the released composition layout, but freeze a lossless shared PNG.
                        from mso.panels import compose
                        mosaic = compose(paths)
                        mosaic_path = OUT/'images'/f'{uid}-panels.png'
                        mosaic.save(mosaic_path)
                        prefix = ('The image contains two views of the same robot state: panel 1 is the fixed '
                                  'workspace camera and panel 2 is the wrist camera. Task: '+instruction+'.\n')
                        questions = {
                            'direction': {'type': 'choice', 'instructions': prefix+
                                'Over the next five recorded control steps, which direction will have the largest '
                                'NET displacement of the gripper in the demonstration? Use the coordinate frame '
                                'of the fixed workspace camera (panel 1): left/right and up/down are image axes; '
                                'forward is away from that camera and backward is toward it. Choose stay still '
                                'if displacement is below 1 cm along every camera-frame axis.',
                                'options': [{'key': k, 'text': v} for k, v in zip(DIRECTION_KEYS, DIRECTION_TEXT)]},
                            'gripper': {'type': 'choice', 'instructions': prefix+
                                'At the NEXT recorded control step, will the demonstration issue an OPEN or CLOSE '
                                'command to the gripper? A repeated OPEN command includes keeping it open; '
                                'a repeated CLOSE command includes keeping it closed. Predict the command, '
                                'not whether an object is currently held.',
                                'options': [{'key': 'open', 'text': 'open the gripper (or keep commanding it open)'},
                                            {'key': 'close', 'text': 'close the gripper (or keep commanding it closed)'}]},
                        }
                        row = {'uid': uid, 'split': split, 'task_id': task_idx, 'source_file': file.name,
                               'episode': demo, 'episode_uid': f'task{task_idx:02d}-{demo}', 'timestep': t,
                               'future_timestep': t+HORIZON, 'task_instruction': instruction,
                               'image': str(mosaic_path), 'image_sha256': sha(mosaic_path),
                               'camera_images': paths, 'questions': questions,
                               'labels': {'direction': direction, 'gripper': gripper},
                               'label_evidence': {'delta_world_m': (pos[t+HORIZON]-pos[t]).tolist(),
                                                  'delta_camera_m': delta, 'axis_margin_m': margin,
                                                  'next_gripper_action': float(actions[t+1, 6])}}
                        records[split].append(row)
                        if split == 'dev' and demo == demos[0] and bin_idx == 2:
                            before = Image.open(paths[0]).convert('RGB')
                            future = Image.fromarray(g['obs/agentview_rgb'][t+HORIZON][::-1].copy())
                            raw = Image.fromarray(g['obs/agentview_rgb'][t])
                            for im, p in [(before, pos[t]), (future, pos[t+HORIZON])]:
                                x, y = project(p, cam_pos, rot, fovy)
                                ImageDraw.Draw(im).ellipse((x-3, y-3, x+3, y+3), outline='red', width=2)
                            strip = Image.new('RGB', (512, 180), '#202020')
                            draw = ImageDraw.Draw(strip)
                            for j, (im, title) in enumerate([(raw, 'raw OpenGL'), (before, 'upright + eef'),
                                                          (future, 'future + eef'), (Image.open(paths[1]), 'wrist upright')]):
                                strip.paste(im, (j*128, 24)); draw.text((j*128+2, 4), title, fill='white')
                            draw.text((2, 158), f'{task_idx}: {demo} t={t} dir={direction} command={gripper}', fill='white')
                            audit.append(strip)
    assert len(records['test']) == 250 and len(records['dev']) == 100
    for split, rows in records.items():
        (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    sheet = Image.new('RGB', (512, len(audit)*180), '#202020')
    for i, strip in enumerate(audit): sheet.paste(strip, (0, i*180))
    sheet.save(OUT/'audit-contact-sheet.png')
    statistics = {split: {'states': len(rows), 'questions': len(rows)*2,
                         'episodes': len(set(r['episode_uid'] for r in rows)),
                         'label_counts': {q: dict(collections.Counter(r['labels'][q] for r in rows))
                                          for q in ['direction', 'gripper']}}
                  for split, rows in records.items()}
    protocol = {
        'name': 'Custom paired LIBERO-10 frame diagnostic v1; NOT reproduction of official robot_long scores',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': SEED,
        'data_revision': 'f13aa24a3da8c43c7225569f28c562979fa0e35a',
        'data_manifest_sha256': sha(DATA/'download_manifest.json'), 'builder_sha256': sha(__file__),
        'sampling': 'Random 2 dev and 5 test episodes per task; 5 random frames from separate temporal strata per episode; no dev/test episode overlap.',
        'image_preprocessing': 'Vertical flip from explicitly recorded OpenGL convention; agentview and wrist composed once with released mso.panels.compose into lossless PNG, identically reused by both models.',
        'direction_rule': {'horizon_recorded_steps': HORIZON, 'still_max_component_metres': STILL_METRES,
                           'transform': 'camera-to-world quaternion from episode XML; delta_camera = R.T @ (eef[t+5]-eef[t]); largest absolute component determines direction; camera +X right, +Y up, +Z toward viewer.',
                           'tie_break': 'numpy argmax first axis X then Y then Z',
                           'target': 'Future demonstrated net movement, not a unique optimal policy action.'},
        'gripper_rule': 'Observations were recorded after env.step(actions[t]); NEXT command is actions[t+1,6]. Panda sign -1=open,+1=close, including repeated commands.',
        'sources': ['https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py',
                    'https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py'],
        'abstention': 'Both choice pipelines include none-of-the-above. Abstention counts wrong. Forced choice and abstention rate also reported.',
        'dev_use': 'Coordinate/image/sign auditing only; no evaluation-driven threshold or prompt tuning.',
        'overlap_limitation': 'Released OmniJev training overlap with LIBERO demonstrations is unknown; local dev/test split does not establish unseen training examples.',
        'episode_split': episode_split, 'statistics': statistics,
        'dev_command_width_audit': {sign: {'n': len(v), 'median_width_change_over_5_steps_m': float(np.median(v))}
                                    for sign, v in command_audit.items()},
        'file_sha256': {f'{s}.jsonl': sha(OUT/f'{s}.jsonl') for s in records},
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(OUT), 'statistics': statistics,
                      'gripper_audit': protocol['dev_command_width_audit']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/data/users/sunziang/OmniJev')
    main()
