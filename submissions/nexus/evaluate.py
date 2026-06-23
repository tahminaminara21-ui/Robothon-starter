"""Benchmark, ablation, dynamics analysis, dataset export, energy conservation."""
import json
import numpy as np
import mujoco
from pathlib import Path
from env import (ShadowHandEnv, run_episode, export_dataset, run_task_suite,
                 RESULTS, SCENE_PATH, OPEN_CTRL, PRESHAPE_CTRL, GRASP_CTRL,
                 ACT_WRJ1, STATE_NAMES, IDLE, PRESHAPE, GRASP, REORIENT, HOLD,
                 PLACE, RELEASE, DONE)

RESULTS.mkdir(exist_ok=True)


def run_benchmark(n_seeds=20, use_domain_rand=False, tag='benchmark'):
    results = []
    for seed in range(n_seeds):
        r = run_episode(seed=seed, use_domain_rand=use_domain_rand)
        r.pop('trajectory', None)
        results.append({'seed': seed, **r})
        status = 'PASS' if r['success'] else 'FAIL'
        print(f"  seed={seed:2d} {status} contacts={r['n_contacts_at_grasp']} "
              f"rot={r['wrist_rotation_deg']:.1f}deg")
    n_pass = sum(r['success'] for r in results)
    print(f"[{tag}] {n_pass}/{n_seeds} passed")
    out = RESULTS / f'{tag}_report.json'
    with open(out, 'w') as f:
        json.dump({'tag': tag, 'n_pass': n_pass, 'n_seeds': n_seeds, 'results': results}, f, indent=2)
    print(f"Saved {out}")
    return results


class OpenLoopEnv(ShadowHandEnv):
    """Ablation: slip_reflex disabled AND grasp gate is time-only (no force sensing)."""
    def slip_reflex(self, ctrl, threshold=0.0):
        return ctrl  # no grip correction

    def run(self, max_steps=4000, domain_rand=False, force_budget=None):
        self.reset(domain_rand=domain_rand)
        ctrl         = OPEN_CTRL.copy()
        grasp_locked = None

        for _ in range(max_steps):
            sc = self.step_count
            if self.state == IDLE:
                ctrl = OPEN_CTRL.copy()
                if sc >= 30: self.state = PRESHAPE
            elif self.state == PRESHAPE:
                t = min((sc - 30) / 100.0, 1.0)
                ctrl = OPEN_CTRL + t * (PRESHAPE_CTRL - OPEN_CTRL)
                if sc >= 140: self.state = GRASP
            elif self.state == GRASP:
                t = min((sc - 140) / 160.0, 1.0)
                ctrl = PRESHAPE_CTRL + t * (GRASP_CTRL - PRESHAPE_CTRL)
                # Open-loop: gate on time only, NO contact force check
                if sc >= 450:
                    grasp_locked = GRASP_CTRL.copy()
                    self._n_contacts_grasp = self._n_contacts_on_cube()
                    self._wrj1_start   = self.get_wrj1_pos()
                    self._cube_start_z = float(self.cube_pos()[2])
                    self.state = REORIENT
            elif self.state == REORIENT:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi
                # No slip reflex
                if sc >= 950: self.state = HOLD
            elif self.state == HOLD:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi
                self._peak_wrj1 = self.get_wrj1_pos()
                if sc > 1200: self.state = PLACE
            elif self.state == PLACE:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = 0.0
                if sc > 1350: self.state = RELEASE
            elif self.state == RELEASE:
                t = min((sc - 1350) / 100.0, 1.0)
                ctrl = grasp_locked + t * (OPEN_CTRL - grasp_locked)
                if sc > 1480: self.state = DONE
            self.step(ctrl)
            if self.state == DONE:
                break

        wrj1_peak = getattr(self, '_peak_wrj1', None) or self.get_wrj1_pos()
        wrj1_ref  = self._wrj1_start if self._wrj1_start is not None else wrj1_peak
        wrist_rot = float(np.degrees(abs(wrj1_peak - wrj1_ref)))
        success   = (self.state == DONE and self._n_contacts_grasp >= 3 and wrist_rot > 10.0)
        return {
            "success": success, "final_state": STATE_NAMES[self.state],
            "wrist_rotation_deg": round(wrist_rot, 2),
            "n_contacts_at_grasp": self._n_contacts_grasp,
            "slip_events": 0, "steps": self.step_count,
        }


def _run_episode_open_loop(seed=0):
    env = OpenLoopEnv(seed=seed)
    result = env.run(max_steps=1600, domain_rand=False)
    return result


def run_ablation(n_seeds=10):
    print("\n=== Ablation: closed-loop vs open-loop ===")
    closed, opened = [], []
    for seed in range(n_seeds):
        r_cl = run_episode(seed=seed, use_domain_rand=True, max_steps=1600)
        r_cl.pop('trajectory', None)
        closed.append(r_cl)
        r_ol = _run_episode_open_loop(seed=seed)
        opened.append(r_ol)
        print(f"  seed={seed} CL={'PASS' if r_cl['success'] else 'FAIL'}"
              f"({r_cl['wrist_rotation_deg']:.1f}deg)"
              f"  OL={'PASS' if r_ol['success'] else 'FAIL'}"
              f"({r_ol['wrist_rotation_deg']:.1f}deg)")
    n_cl = sum(r['success'] for r in closed)
    n_ol = sum(r['success'] for r in opened)
    print(f"  Closed-loop: {n_cl}/{n_seeds}  Open-loop: {n_ol}/{n_seeds}")
    # Sensor-cut delta: mean wrist rotation diff
    cl_rots = [r['wrist_rotation_deg'] for r in closed]
    ol_rots = [r['wrist_rotation_deg'] for r in opened]
    print(f"  Mean rot CL={np.mean(cl_rots):.2f}deg  OL={np.mean(ol_rots):.2f}deg"
          f"  delta={np.mean(cl_rots)-np.mean(ol_rots):.2f}deg")
    out = RESULTS / 'ablation_report.json'
    with open(out, 'w') as f:
        json.dump({
            'n_closed_pass': n_cl, 'n_open_pass': n_ol, 'n_seeds': n_seeds,
            'closed_loop_mean_rot': round(float(np.mean(cl_rots)), 3),
            'open_loop_mean_rot':   round(float(np.mean(ol_rots)), 3),
            'rot_delta_deg':        round(float(np.mean(cl_rots) - np.mean(ol_rots)), 3),
            'closed_loop': closed, 'open_loop': opened,
        }, f, indent=2)
    print(f"Saved {out}")
    return n_cl, n_ol


def run_dynamics_analysis():
    print("\n=== Dynamics Analysis (9 MuJoCo APIs) ===")
    m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    report = {}

    # 1. mjd_transitionFD — linearise sim to A/B matrices
    A = np.zeros((2 * m.nv, 2 * m.nv))
    B = np.zeros((2 * m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    report['mjd_transitionFD'] = {'A_shape': list(A.shape), 'B_shape': list(B.shape),
                                   'A_norm': round(float(np.linalg.norm(A)), 4)}
    print(f"  mjd_transitionFD: A{A.shape} norm={report['mjd_transitionFD']['A_norm']:.4f}")

    # 2. mj_fullM
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    report['mj_fullM'] = {'shape': list(M.shape), 'trace': round(float(np.trace(M)), 4)}
    print(f"  mj_fullM: M{M.shape} trace={report['mj_fullM']['trace']:.4f}")

    # 3. mj_mulM
    vec = np.ones(m.nv)
    out = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, out, vec)
    report['mj_mulM'] = {'out_norm': round(float(np.linalg.norm(out)), 4)}
    print(f"  mj_mulM: ||Mv||={report['mj_mulM']['out_norm']:.4f}")

    # 4. mj_differentiatePos
    qpos2 = d.qpos.copy(); qpos2[0] += 0.01
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, d.qpos, qpos2)
    report['mj_differentiatePos'] = {'dq_norm': round(float(np.linalg.norm(dq)), 6)}
    print(f"  mj_differentiatePos: ||dq||={report['mj_differentiatePos']['dq_norm']:.6f}")

    # 5. mj_jacBody (cube)
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, 28)
    report['mj_jacBody'] = {'jacp_norm': round(float(np.linalg.norm(jacp)), 4)}
    print(f"  mj_jacBody(cube): ||Jp||={report['mj_jacBody']['jacp_norm']:.4f}")

    # 6. mj_angmomMat
    H = np.zeros((3, m.nv))
    mujoco.mj_angmomMat(m, d, H, 1)
    report['mj_angmomMat'] = {'H_norm': round(float(np.linalg.norm(H)), 4)}
    print(f"  mj_angmomMat: ||H||={report['mj_angmomMat']['H_norm']:.4f}")

    # 7. mj_geomDistance
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(m, d, 26, 66, 0.1, fromto)
    report['mj_geomDistance'] = {'ff_cube_dist': round(float(dist), 4)}
    print(f"  mj_geomDistance(ff_distal,cube): {dist:.4f} m")

    # 8. mj_contactForce
    forces = []
    for i in range(d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        forces.append(round(float(np.linalg.norm(f)), 4))
    report['mj_contactForce'] = {'n_contacts': d.ncon, 'force_norms': forces}
    print(f"  mj_contactForce: {d.ncon} contacts")

    # 9. Energy conservation (mj_energyPos + mj_energyVel)
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    e0 = d.energy[0] + d.energy[1]
    # Step 2000 times and check energy drift
    for _ in range(2000): mujoco.mj_step(m, d)
    mujoco.mj_energyPos(m, d); mujoco.mj_energyVel(m, d)
    e1 = d.energy[0] + d.energy[1]
    energy_error_pct = abs(e1 - e0) / (abs(e0) + 1e-12) * 100
    report['energy'] = {
        'initial': round(float(e0), 6), 'final': round(float(e1), 6),
        'error_pct': round(float(energy_error_pct), 4)
    }
    print(f"  energy conservation: error={energy_error_pct:.4f}% over 2000 steps")

    out_path = RESULTS / 'dynamics_report.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_path}")
    return report


def main():
    print("=== Benchmark (20 seeds, no domain rand) ===")
    run_benchmark(n_seeds=20, use_domain_rand=False, tag='benchmark')

    print("\n=== Benchmark (20 seeds, domain rand ±30%) ===")
    run_benchmark(n_seeds=20, use_domain_rand=True, tag='benchmark_rand')

    print("\n=== Task Suite (20 tasks) ===")
    run_task_suite()

    run_ablation(n_seeds=10)
    run_dynamics_analysis()

    print("\n=== Exporting dataset (5 seeds) ===")
    export_dataset(n=5)

    print("\nAll done.")


if __name__ == '__main__':
    main()
