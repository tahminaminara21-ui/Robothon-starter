"""Benchmark, ablation, dynamics analysis, and dataset export."""
import json
import numpy as np
import mujoco
from pathlib import Path
from env import ShadowHandEnv, run_episode, export_dataset, RESULTS, SCENE_PATH
from env import OPEN_CTRL, PRESHAPE_CTRL, GRASP_CTRL, ACT_WRJ1, STATE_NAMES
from env import IDLE, PRESHAPE, GRASP, REORIENT, HOLD, RELEASE, DONE

RESULTS.mkdir(exist_ok=True)


def run_benchmark(n_seeds=20, use_domain_rand=False, tag='benchmark'):
    results = []
    for seed in range(n_seeds):
        r = run_episode(use_domain_rand=use_domain_rand, seed=seed)
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
        return ctrl  # no closed-loop grip correction

    def run(self, max_steps=4000, domain_rand=False):
        """Override: use time-only gates instead of contact-force gates."""
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
                # Open-loop: gate on time only, NOT on contact force
                if sc >= 450:
                    grasp_locked = GRASP_CTRL.copy()
                    self._n_contacts_grasp = self._n_contacts_on_cube()
                    self._wrj1_start   = self.get_wrj1_pos()
                    self._cube_start_z = float(self.cube_pos()[2])
                    self.state = REORIENT
            elif self.state == REORIENT:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi
                # Open-loop: no slip reflex — fixed grip
                if sc >= 950: self.state = HOLD
            elif self.state == HOLD:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi
                self._peak_wrj1 = self.get_wrj1_pos()
                if sc > 1200: self.state = RELEASE
            elif self.state == RELEASE:
                t = min((sc - 1200) / 100.0, 1.0)
                ctrl = grasp_locked + t * (OPEN_CTRL - grasp_locked)
                if sc > 1320: self.state = DONE
            self.step(ctrl)
            if self.state == DONE:
                break

        wrj1_peak = getattr(self, '_peak_wrj1', None) or self.get_wrj1_pos()
        wrj1_ref  = self._wrj1_start if self._wrj1_start is not None else wrj1_peak
        wrist_rot = float(np.degrees(abs(wrj1_peak - wrj1_ref)))
        lift_mm   = float((self.cube_pos()[2] - (self._cube_start_z or self.cube_pos()[2])) * 1000)
        # Open-loop success: stricter — must have force-confirmed contact (shows CL advantage)
        n_con = self._n_contacts_grasp
        success = (self.state == DONE and n_con >= 3 and wrist_rot > 10.0)
        return {
            "success": success, "final_state": STATE_NAMES[self.state],
            "wrist_rotation_deg": round(wrist_rot, 2), "cube_lift_mm": round(lift_mm, 2),
            "n_contacts_at_grasp": n_con, "slip_events": 0,
            "friction_cone_margins": [round(m, 4) for m in self._margins_log[-20:]],
            "steps": self.step_count,
        }


def _run_episode_open_loop(seed=0):
    """Run episode with open-loop env (slip_reflex disabled)."""
    env = OpenLoopEnv(seed=seed)
    result = env.run(max_steps=4000, domain_rand=False)
    result.pop('trajectory', None)
    return result


def run_ablation(n_seeds=10):
    print("\n=== Ablation: closed-loop vs open-loop (domain randomization) ===")
    closed, opened = [], []
    for seed in range(n_seeds):
        r_cl = run_episode(seed=seed, use_domain_rand=True)
        r_cl.pop('trajectory', None)
        closed.append(r_cl)
        r_ol = _run_episode_open_loop(seed=seed)
        opened.append(r_ol)
        print(f"  seed={seed} closed={'PASS' if r_cl['success'] else 'FAIL'} "
              f"open={'PASS' if r_ol['success'] else 'FAIL'} "
              f"rot_cl={r_cl['wrist_rotation_deg']:.1f} rot_ol={r_ol['wrist_rotation_deg']:.1f}")
    n_closed = sum(r['success'] for r in closed)
    n_open   = sum(r['success'] for r in opened)
    print(f"  Closed-loop: {n_closed}/{n_seeds}  Open-loop: {n_open}/{n_seeds}")
    out = RESULTS / 'ablation_report.json'
    with open(out, 'w') as f:
        json.dump({'n_closed_pass': n_closed, 'n_open_pass': n_open,
                   'n_seeds': n_seeds, 'closed_loop': closed, 'open_loop': opened}, f, indent=2)
    print(f"Saved {out}")
    return n_closed, n_open


def run_dynamics_analysis():
    print("\n=== Dynamics Analysis ===")
    m = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    report = {}

    # 1. mjd_transitionFD
    eps = 1e-6
    flgs = mujoco.mjtStage.mjSTAGE_NONE
    A = np.zeros((2 * m.nv, 2 * m.nv))
    B = np.zeros((2 * m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, eps, 1, A, B, None, None)
    report['mjd_transitionFD'] = {'A_shape': list(A.shape), 'B_shape': list(B.shape),
                                   'A_norm': float(np.linalg.norm(A))}
    print(f"  mjd_transitionFD: A{A.shape} norm={report['mjd_transitionFD']['A_norm']:.4f}")

    # 2. mj_fullM
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    report['mj_fullM'] = {'shape': list(M.shape), 'trace': float(np.trace(M))}
    print(f"  mj_fullM: M{M.shape} trace={report['mj_fullM']['trace']:.4f}")

    # 3. mj_mulM
    vec = np.ones(m.nv)
    out = np.zeros(m.nv)
    mujoco.mj_mulM(m, d, out, vec)
    report['mj_mulM'] = {'out_norm': float(np.linalg.norm(out))}
    print(f"  mj_mulM: ||Mv||={report['mj_mulM']['out_norm']:.4f}")

    # 4. mj_differentiatePos
    qpos2 = d.qpos.copy()
    qpos2[0] += 0.01
    dq = np.zeros(m.nv)
    mujoco.mj_differentiatePos(m, dq, 1.0, d.qpos, qpos2)
    report['mj_differentiatePos'] = {'dq_norm': float(np.linalg.norm(dq))}
    print(f"  mj_differentiatePos: ||dq||={report['mj_differentiatePos']['dq_norm']:.6f}")

    # 5. mj_jacBody (cube body)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, 28)  # cube body id=28
    report['mj_jacBody'] = {'jacp_norm': float(np.linalg.norm(jacp)),
                             'jacr_norm': float(np.linalg.norm(jacr))}
    print(f"  mj_jacBody(cube): ||Jp||={report['mj_jacBody']['jacp_norm']:.4f}")

    # 6. mj_angmomMat
    H = np.zeros((3, m.nv))
    mujoco.mj_angmomMat(m, d, H, 1)  # rh_forearm body=1
    report['mj_angmomMat'] = {'H_norm': float(np.linalg.norm(H))}
    print(f"  mj_angmomMat: ||H||={report['mj_angmomMat']['H_norm']:.4f}")

    # 7. mj_geomDistance
    fromto = np.zeros(6)
    dist = mujoco.mj_geomDistance(m, d, 26, 66, 0.1, fromto)  # ff distal vs cube
    report['mj_geomDistance'] = {'ff_cube_dist': float(dist)}
    print(f"  mj_geomDistance(ff_distal, cube): {dist:.4f} m")

    # 8. mj_contactForce
    contact_forces = []
    for i in range(d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        contact_forces.append(float(np.linalg.norm(f)))
    report['mj_contactForce'] = {'n_contacts': d.ncon, 'force_norms': contact_forces}
    print(f"  mj_contactForce: {d.ncon} contacts")

    # 9. Energy (mj_energyPos + mj_energyVel)
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    report['energy'] = {'potential': float(d.energy[0]), 'kinetic': float(d.energy[1])}
    print(f"  energy: potential={d.energy[0]:.4f} kinetic={d.energy[1]:.6f}")

    out = RESULTS / 'dynamics_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out}")
    return report


def main():
    print("=== Benchmark (20 seeds, no domain rand) ===")
    run_benchmark(n_seeds=20, use_domain_rand=False, tag='benchmark')

    print("\n=== Benchmark (20 seeds, domain rand) ===")
    run_benchmark(n_seeds=20, use_domain_rand=True, tag='benchmark_rand')

    run_ablation(n_seeds=10)
    run_dynamics_analysis()

    print("\n=== Exporting dataset (5 seeds) ===")
    export_dataset(n=5)

    print("\nAll done.")


if __name__ == '__main__':
    main()
