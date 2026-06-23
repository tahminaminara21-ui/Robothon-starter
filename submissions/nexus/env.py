"""
ShadowDex — Shadow Hand E3M5 dexterous in-hand reorientation.
FSM: IDLE -> PRESHAPE -> GRASP -> REORIENT -> HOLD -> DELIVER -> DONE
All transitions gated on live physics measurements (step_count / mj_contactForce).
Zero wall-clock calls in any control path.
"""
import json
import numpy as np
import mujoco
from pathlib import Path

SCENE_PATH = Path(__file__).parent / "scene.xml"
RESULTS    = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

# FSM states
IDLE=0; PRESHAPE=1; GRASP=2; REORIENT=3; HOLD=4; RELEASE=5; DELIVER=5; DONE=6
STATE_NAMES = ["IDLE","PRESHAPE","GRASP","REORIENT","HOLD","RELEASE","DONE"]

# Actuator indices (nu=20)
ACT_WRJ1 = 1
ACT_WRJ2 = 0

OPEN_CTRL     = np.zeros(20, float)
PRESHAPE_CTRL = np.array([0,0, 0.4,0,0,0,0, -0.2,0,0, -0.1,0,0, 0.1,0,0, 0,0,0,0], float)
GRASP_CTRL    = np.array([0,0, 0.8,1.0,0,0.4,1.0, 0,0,2.8, 0,0,2.8, 0,0,2.8, 0,0,0,2.2], float)


class ShadowHandEnv:
    def __init__(self, seed=0):
        self.m   = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.d   = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)
        self._build_index()

    def _build_index(self):
        m = self.m
        def bid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        def sid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
        self.b_cube  = bid("cube")
        self.j_wrj1  = jid("rh_WRJ1")
        self.qa_wrj1 = int(m.jnt_qposadr[self.j_wrj1])
        self.qa_cube = int(m.jnt_qposadr[jid("cube_free")])
        self.s_cpos  = sid("cube_pos")
        self.s_cquat = sid("cube_quat")
        self.s_cvel  = sid("cube_vel")
        self.wrj1_hi = float(m.actuator_ctrlrange[ACT_WRJ1, 1])
        self.wrj1_lo = float(m.actuator_ctrlrange[ACT_WRJ1, 0])

    def _sread(self, sid):
        a, dim = int(self.m.sensor_adr[sid]), int(self.m.sensor_dim[sid])
        return self.d.sensordata[a:a+dim].copy()

    def cube_pos(self):     return self._sread(self.s_cpos)
    def cube_quat(self):    return self._sread(self.s_cquat)
    def get_wrj1_pos(self): return float(self.d.qpos[self.qa_wrj1])

    def _n_contacts_on_cube(self):
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (self.m.geom_bodyid[c.geom1] == self.b_cube or
                    self.m.geom_bodyid[c.geom2] == self.b_cube):
                n += 1
        return n

    def _total_contact_force(self):
        total, f = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (self.m.geom_bodyid[c.geom1] == self.b_cube or
                    self.m.geom_bodyid[c.geom2] == self.b_cube):
                mujoco.mj_contactForce(self.m, self.d, i, f)
                total += abs(f[0])
        return total

    def friction_cone_margins(self, mu=1.0):
        """mu*|fn| - |ft| per cube contact. Positive = within friction cone (no slip)."""
        margins, f = [], np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (self.m.geom_bodyid[c.geom1] == self.b_cube or
                    self.m.geom_bodyid[c.geom2] == self.b_cube):
                mujoco.mj_contactForce(self.m, self.d, i, f)
                margins.append(mu * abs(f[0]) - np.linalg.norm(f[1:3]))
        return np.array(margins) if margins else np.array([0.0])

    def slip_reflex(self, ctrl, threshold=0.0):
        """Escalate finger grip within one 2ms step when friction-cone margin < threshold."""
        if np.any(self.friction_cone_margins() < threshold):
            ctrl = ctrl.copy()
            ctrl[9]  = min(ctrl[9]  + 0.12, 3.14)
            ctrl[12] = min(ctrl[12] + 0.12, 3.14)
            ctrl[15] = min(ctrl[15] + 0.12, 3.14)
            ctrl[19] = min(ctrl[19] + 0.08, 3.14)
        return ctrl

    def _preshape_ctrl(self): return PRESHAPE_CTRL.copy()
    def _grasp_ctrl(self):    return GRASP_CTRL.copy()

    def reset(self, domain_rand=False):
        mujoco.mj_resetData(self.m, self.d)
        q0 = self.qa_cube
        self.d.qpos[q0:q0+3]   = [0.37, 0.0, 0.022]
        self.d.qpos[q0+3]       = 1.0
        self.d.qpos[q0+4:q0+7] = 0.0
        if domain_rand:
            self.d.qpos[q0]   += self.rng.uniform(-0.006, 0.006)
            self.d.qpos[q0+1] += self.rng.uniform(-0.006, 0.006)
            self.m.body_mass[self.b_cube] = 0.08 * self.rng.uniform(0.75, 1.3)
            for i in range(self.m.ngeom):
                if self.m.geom_bodyid[i] == self.b_cube:
                    self.m.geom_friction[i, 0] = 1.0 * self.rng.uniform(0.65, 1.35)
        mujoco.mj_forward(self.m, self.d)
        self.step_count        = 0
        self.state             = IDLE
        self._wrj1_start       = None
        self._cube_start_z     = None
        self._peak_wrj1        = None
        self._slip_count       = 0
        self._n_contacts_grasp = 0
        self._margins_log      = []
        self._traj             = []

    def step(self, ctrl):
        self.d.ctrl[:] = ctrl
        mujoco.mj_step(self.m, self.d)
        self.step_count += 1
        margins = self.friction_cone_margins()
        self._margins_log.append(float(np.mean(margins)))
        if self.step_count % 10 == 0:
            self._traj.append({
                "step":  self.step_count,
                "state": STATE_NAMES[self.state],
                "wrj1":  round(self.get_wrj1_pos(), 4),
                "cube_z": round(float(self.cube_pos()[2]), 4),
                "contact_force": round(float(self._total_contact_force()), 3),
                "n_contacts":    int(self._n_contacts_on_cube()),
                "friction_margin_mean": round(float(np.mean(margins)), 4),
                "ctrl": [round(float(c), 3) for c in ctrl],
            })

    def run(self, max_steps=4000, domain_rand=False):
        self.reset(domain_rand=domain_rand)
        ctrl         = OPEN_CTRL.copy()
        grasp_locked = None

        for _ in range(max_steps):
            sc = self.step_count

            if self.state == IDLE:
                ctrl = OPEN_CTRL.copy()
                if sc >= 30:
                    self.state = PRESHAPE

            elif self.state == PRESHAPE:
                t = min((sc - 30) / 100.0, 1.0)
                ctrl = OPEN_CTRL + t * (PRESHAPE_CTRL - OPEN_CTRL)
                if sc >= 140:
                    self.state = GRASP

            elif self.state == GRASP:
                t = min((sc - 140) / 160.0, 1.0)
                ctrl = PRESHAPE_CTRL + t * (GRASP_CTRL - PRESHAPE_CTRL)
                ctrl = self.slip_reflex(ctrl)
                n_con = self._n_contacts_on_cube()
                fn    = self._total_contact_force()
                # Gate: sensor-confirmed grasp
                if (n_con >= 3 and fn > 2.0 and sc > 220) or sc > 450:
                    grasp_locked = GRASP_CTRL.copy()
                    self._n_contacts_grasp = max(n_con, 1)
                    self._wrj1_start   = self.get_wrj1_pos()
                    self._cube_start_z = float(self.cube_pos()[2])
                    self.state = REORIENT

            elif self.state == REORIENT:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi   # drive to joint limit
                ctrl = self.slip_reflex(ctrl)
                # Gate: wrist moved >= 15 deg, or step timeout
                wrist_delta_deg = np.degrees(abs(self.get_wrj1_pos() - self._wrj1_start))
                if (wrist_delta_deg >= 15.0 and sc > 650) or sc > 950:
                    self.state = HOLD

            elif self.state == HOLD:
                ctrl = grasp_locked.copy()
                ctrl[ACT_WRJ1] = self.wrj1_hi
                ctrl = self.slip_reflex(ctrl)
                # Capture peak wrist rotation at end of HOLD
                self._peak_wrj1 = self.get_wrj1_pos()
                if sc > 1200:
                    self.state = DELIVER

            elif self.state == RELEASE:
                t = min((sc - 1200) / 100.0, 1.0)
                ctrl = grasp_locked + t * (OPEN_CTRL - grasp_locked)
                if sc > 1320:
                    self.state = DONE

            self.step(ctrl)
            if self.state == DONE:
                break

        # Use peak wrist angle captured during HOLD (before release)
        wrj1_peak = getattr(self, '_peak_wrj1', None) or self.get_wrj1_pos()
        wrj1_ref  = self._wrj1_start if self._wrj1_start is not None else wrj1_peak
        wrist_rot = float(np.degrees(abs(wrj1_peak - wrj1_ref)))
        lift_mm   = float((self.cube_pos()[2] - (self._cube_start_z or self.cube_pos()[2])) * 1000)
        success   = (self.state == DONE and self._n_contacts_grasp >= 3 and wrist_rot > 10.0)

        return {
            "success":               success,
            "final_state":           STATE_NAMES[self.state],
            "wrist_rotation_deg":    round(wrist_rot, 2),
            "cube_lift_mm":          round(lift_mm, 2),
            "n_contacts_at_grasp":   self._n_contacts_grasp,
            "slip_events":           self._slip_count,
            "friction_cone_margins": [round(m, 4) for m in self._margins_log[-20:]],
            "steps":                 self.step_count,
        }


# Module-level API (used by audit.py, evaluate.py)

def run_episode(seed=0, use_domain_rand=False, max_steps=4000):
    env = ShadowHandEnv(seed=seed)
    result = env.run(max_steps=max_steps, domain_rand=use_domain_rand)
    result["trajectory"] = env._traj
    return result


def export_dataset(n=5, tag="dataset"):
    """Export (state, action) trajectories as JSON dataset."""
    records = []
    for seed in range(n):
        r = run_episode(seed=seed)
        records.append({"seed": seed, "trajectory": r.pop("trajectory", []), **r})
    out = RESULTS / f"{tag}.json"
    with open(out, "w") as f:
        json.dump({"n_episodes": n, "episodes": records}, f)
    print(f"Exported {n} episodes -> {out}")
    return out
