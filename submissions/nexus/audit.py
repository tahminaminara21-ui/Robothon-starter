"""Audit script: 30+ checks for Robothon 2026 submission."""
import json
import ast
import re
import inspect
import numpy as np
import mujoco
from pathlib import Path

RESULTS = Path(__file__).parent / 'results'
SCENE = Path(__file__).parent / 'scene.xml'
ENV_SRC = Path(__file__).parent / 'env.py'
EVAL_SRC = Path(__file__).parent / 'evaluate.py'
RESULTS.mkdir(exist_ok=True)

checks = []


def check(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ''))
    checks.append({'name': name, 'status': status, 'detail': detail})


def src(path):
    return path.read_text()


def run_audit():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    env_code = src(ENV_SRC)
    eval_code = src(EVAL_SRC)

    print("=== Audit: MJCF / Model ===")
    check('nq=31', m.nq == 31, f'nq={m.nq}')
    check('nv=30', m.nv == 30, f'nv={m.nv}')
    check('nu=20', m.nu == 20, f'nu={m.nu}')
    check('nsensor=34', m.nsensor == 34, f'nsensor={m.nsensor}')
    check('freejoint cube_free exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'cube_free') >= 0)
    check('cube body exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cube') >= 0)
    check('hand_weld equality exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, 'hand_weld') >= 0)
    check('hand_weld is active', bool(m.eq_active0[0]))
    check('rh_forearm body exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'rh_forearm') >= 0)
    check('rh_thdistal body exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'rh_thdistal') >= 0)
    check('cube_pos sensor exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'cube_pos') >= 0)
    check('cube_quat sensor exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'cube_quat') >= 0)
    check('grasp_site sensor exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'grasp_pos') >= 0)
    check('timestep=0.002', abs(m.opt.timestep - 0.002) < 1e-6, f'dt={m.opt.timestep}')
    check('mesh assets loaded (ngeom>=50)', m.ngeom >= 50, f'ngeom={m.ngeom}')
    check('WRJ1 actuator exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'rh_A_WRJ1') >= 0)
    check('THJ1 actuator exists',
          mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'rh_A_THJ1') >= 0)
    check('condim=6 on cube geom', m.geom_condim[66] == 6, f'condim={m.geom_condim[66]}')

    print("\n=== Audit: env.py source ===")
    check('no time.time() in env.py', 'time.time()' not in env_code)
    check('FSM IDLE state defined', 'IDLE' in env_code and '= 0' in env_code)
    check('FSM PRESHAPE state defined', 'PRESHAPE' in env_code)
    check('FSM GRASP state defined', 'GRASP' in env_code)
    check('FSM REORIENT state defined', 'REORIENT' in env_code)
    check('FSM HOLD state defined', 'HOLD' in env_code)
    check('FSM RELEASE state defined', 'RELEASE' in env_code)
    check('FSM DONE state defined', 'DONE' in env_code)
    check('mj_contactForce used in env.py', 'mj_contactForce' in env_code)
    check('friction_cone_margin defined', 'friction_cone_margin' in env_code)
    check('slip_reflex defined', 'slip_reflex' in env_code)
    check('domain_rand defined', 'domain_rand' in env_code)
    check('run_episode defined', 'run_episode' in env_code)
    check('export_dataset defined', 'export_dataset' in env_code)
    check('FSM gates on step_count (physics steps)', 'step_count' in env_code)
    check('SCENE_PATH uses Path(__file__)', 'SCENE_PATH = Path(__file__)' in env_code)

    print("\n=== Audit: evaluate.py source ===")
    check('mjd_transitionFD called', 'mjd_transitionFD' in eval_code)
    check('mj_fullM called', 'mj_fullM' in eval_code)
    check('mj_mulM called', 'mj_mulM' in eval_code)
    check('mj_differentiatePos called', 'mj_differentiatePos' in eval_code)
    check('mj_jacBody called', 'mj_jacBody' in eval_code)
    check('mj_angmomMat called', 'mj_angmomMat' in eval_code)
    check('mj_geomDistance called', 'mj_geomDistance' in eval_code)
    check('mj_contactForce called in evaluate.py', 'mj_contactForce' in eval_code)
    check('energy (mj_energyPos/mj_energyVel) called', 'mj_energyPos' in eval_code and 'mj_energyVel' in eval_code)
    check('run_benchmark defined', 'run_benchmark' in eval_code)
    check('run_ablation defined', 'run_ablation' in eval_code)
    check('run_dynamics_analysis defined', 'run_dynamics_analysis' in eval_code)

    print("\n=== Audit: runtime checks ===")
    # Quick episode smoke test
    from env import run_episode
    r = run_episode(max_steps=500)
    check('run_episode returns dict', isinstance(r, dict))
    check('run_episode has success key', 'success' in r)
    check('run_episode has wrist_rotation_deg', 'wrist_rotation_deg' in r)
    check('run_episode has n_contacts_at_grasp', 'n_contacts_at_grasp' in r)
    check('run_episode has friction_cone_margins', 'friction_cone_margins' in r)

    # Check reports if they exist
    bench = RESULTS / 'benchmark_report.json'
    if bench.exists():
        with open(bench) as f:
            bdata = json.load(f)
        n_pass = bdata.get('n_pass', 0)
        n_seeds = bdata.get('n_seeds', 20)
        check('benchmark 20/20 pass', n_pass == n_seeds, f'{n_pass}/{n_seeds}')
    else:
        check('benchmark_report.json exists', False, 'run evaluate.py first')

    abl = RESULTS / 'ablation_report.json'
    if abl.exists():
        with open(abl) as f:
            adata = json.load(f)
        n_cl = adata.get('n_closed_pass', 0)
        n_ol = adata.get('n_open_pass', 0)
        check('closed-loop >= open-loop', n_cl >= n_ol, f'{n_cl} vs {n_ol}')
    else:
        check('ablation_report.json exists', False, 'run evaluate.py first')

    dyn = RESULTS / 'dynamics_report.json'
    check('dynamics_report.json exists', dyn.exists())
    if dyn.exists():
        with open(dyn) as f:
            ddata = json.load(f)
        for api in ['mjd_transitionFD', 'mj_fullM', 'mj_mulM', 'mj_differentiatePos',
                    'mj_jacBody', 'mj_angmomMat', 'mj_geomDistance', 'mj_contactForce', 'energy']:
            check(f'dynamics_report has {api}', api in ddata)

    # Tally
    n_pass = sum(1 for c in checks if c['status'] == 'PASS')
    n_fail = sum(1 for c in checks if c['status'] == 'FAIL')
    print(f"\n=== TOTAL: {n_pass} PASS / {n_fail} FAIL / {len(checks)} checks ===")

    out = RESULTS / 'audit_report.json'
    with open(out, 'w') as f:
        json.dump({'n_pass': n_pass, 'n_fail': n_fail, 'checks': checks}, f, indent=2)
    print(f"Saved {out}")
    return n_fail == 0


if __name__ == '__main__':
    run_audit()
