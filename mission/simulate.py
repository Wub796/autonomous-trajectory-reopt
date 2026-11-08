"""
simulate.py: fly whole missions (many at once) and score them.

A mission:
  * starts at Earth with the launch boost,
  * every dt hours, either follows the pre-made PLAN or asks the AI PILOT
    what to do (the switch happens at `takeover_hour`),
  * the thruster delivers  throttle x T0 x (true health)  of push,
  * ends after 460 days.  If the spacecraft ever gets within 577,000 km of
    Mars it counts as "captured" and the engine shuts off.

Optional real-world messiness (NoiseConfig), following the disturbance
types in Zavoli & Federici (2021):
  launch errors, thrust-size errors, pointing wobble, random engine
  outages, and navigation (GPS-like) errors in what the pilot sees.
"""
from dataclasses import dataclass, field
import numpy as np

from mission import physics as P
from mission import pilot as AI
from mission.thruster import T0, MDOT, WET_MASS, DRY_MASS, MISSION_HOURS, health

DT_HOURS = 12


@dataclass
class NoiseConfig:
    launch_pos_km: float = 0.0          # 1-sigma launch position error
    launch_vel_km_s: float = 0.0        # 1-sigma launch velocity error
    thrust_mag_pct: float = 0.0         # 1-sigma thrust size error (0.025 = 2.5%)
    pointing_deg: float = 0.0           # 1-sigma pointing wobble
    p_outage: float = 0.0               # chance per hour that the engine safes itself
    outage_hours: tuple = (2, 24)       # how long an outage lasts
    nav_pos_km: float = 0.0             # 1-sigma navigation error in position
    nav_vel_km_s: float = 0.0           # 1-sigma navigation error in velocity

    @staticmethod
    def zavoli_federici():
        """Disturbance levels similar to the original project's 'zavoli' preset."""
        return NoiseConfig(launch_pos_km=250.0, launch_vel_km_s=0.005, thrust_mag_pct=0.025,
                           pointing_deg=0.5, p_outage=0.005, outage_hours=(2, 24),
                           nav_pos_km=50.0, nav_vel_km_s=0.002)


@dataclass
class Scenario:
    """N missions' worth of fault + detector information."""
    onset_hour: np.ndarray
    severity: np.ndarray
    ramp_hours: np.ndarray
    believed: np.ndarray = None          # (N, H) believed thrust fraction (from the detector)
    alarm_hour: np.ndarray = None        # (N,) when the onboard detector fired
    takeover_hour: np.ndarray = None     # (N,) when the AI pilot takes control (0 = from launch)

    @property
    def n(self):
        return len(self.onset_hour)

    @staticmethod
    def healthy(n):
        return Scenario(np.full(n, np.inf), np.zeros(n), np.zeros(n))

    def subset_repeat(self, reps):
        """Repeat every scenario `reps` times in a row (for population x scenario grids)."""
        def rep(a):
            return None if a is None else np.repeat(a, reps, axis=0)
        return Scenario(rep(self.onset_hour), rep(self.severity), rep(self.ramp_hours),
                        rep(self.believed), rep(self.alarm_hour), rep(self.takeover_hour))

    def tile(self, reps):
        """Tile all scenarios `reps` times (block after block)."""
        def til(a):
            if a is None:
                return None
            return np.tile(a, (reps,) + (1,) * (a.ndim - 1))
        return Scenario(til(self.onset_hour), til(self.severity), til(self.ramp_hours),
                        til(self.believed), til(self.alarm_hour), til(self.takeover_hour))


def closest_approach(d0, d1):
    """
    Closest distance between two straight-line samples of the Mars-relative position.
    Relative position along the step:  d(s) = d0 + s (d1 - d0),  s in [0, 1].
    |d(s)|^2 is a parabola in s; its minimum is where the derivative is 0:
        s* = -(d0 . (d1-d0)) / |d1-d0|^2       (then clip s* to [0, 1])
    """
    seg = d1 - d0
    s = -np.sum(d0 * seg, -1) / np.maximum(np.sum(seg * seg, -1), 1e-12)
    s = np.clip(s, 0.0, 1.0)
    return np.sqrt(np.sum((d0 + s[:, None] * seg) ** 2, -1))


def fly(theta, scen: Scenario, plan=None, noise: NoiseConfig = None, rng=None,
        dt_hours=DT_HOURS, record=False):
    """
    Fly scen.n missions.

    theta : (243,) one pilot for all missions, or (N, 243) one per mission
    plan  : optional dict with 'throttle' (K,) and 'direction' (K, 3): the pre-made
            command list, followed until scen.takeover_hour
    Returns a dict of per-mission results (and the trajectory if record=True).
    """
    N = scen.n
    rng = rng if rng is not None else np.random.default_rng(0)
    noise = noise or NoiseConfig()
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (N, theta.size))

    dt = dt_hours * P.HOUR
    steps = MISSION_HOURS // dt_hours
    takeover = scen.takeover_hour if scen.takeover_hour is not None else np.zeros(N)
    if plan is None:
        takeover = np.zeros(N)
    believed_all = scen.believed

    # --- launch (with optional launch errors) ---
    r, v = P.earth_launch_state(N)
    r = r + noise.launch_pos_km * rng.standard_normal((N, 3))
    v = v + noise.launch_vel_km_s * rng.standard_normal((N, 3))
    m = np.full(N, WET_MASS)

    mars_r0, _ = P.mars_state(0.0)
    d_prev = mars_r0[None, :] - r
    min_dist = np.sqrt(np.sum(d_prev ** 2, -1))
    min_hour = np.zeros(N)
    captured = np.zeros(N, dtype=bool)
    capture_hour = np.full(N, np.inf)
    vrel_min = np.zeros(N)
    outage_left = np.zeros(N)              # hours of outage remaining
    piloted_hours = np.zeros(N)

    if record:
        traj = np.zeros((steps + 1, N, 3)); traj[0] = r
        cmds = np.zeros((steps, N, 4))
        dist_hist = np.zeros((steps + 1, N)); dist_hist[0] = min_dist

    for k in range(steps):
        hour = k * dt_hours
        t = hour * P.HOUR

        # ---------- decide what to do ----------
        h_true = health(hour, scen.onset_hour, scen.severity, scen.ramp_hours)
        h_believed = believed_all[:, hour] if believed_all is not None else np.ones(N)

        r_nav = r + noise.nav_pos_km * rng.standard_normal((N, 3))
        v_nav = v + noise.nav_vel_km_s * rng.standard_normal((N, 3))
        obs, (rh, th, nh) = AI.observe(r_nav, v_nav, m, t, h_believed)
        thr, alpha, beta = AI.act(theta, obs)
        direction = P.thrust_direction(alpha, beta, rh, th, nh)

        use_pilot = hour >= takeover
        if plan is not None:
            thr = np.where(use_pilot, thr, plan["throttle"][k])
            direction = np.where(use_pilot[:, None], direction, plan["direction"][k][None, :])
        piloted_hours += use_pilot * dt_hours
        thr = np.where(captured, 0.0, thr)     # mission accomplished: engine off

        # ---------- real-world messiness ----------
        on = np.ones(N)
        if noise.p_outage > 0:
            p_step = 1 - (1 - noise.p_outage) ** dt_hours
            start = (outage_left <= 0) & (rng.random(N) < p_step)
            outage_left = np.where(start, rng.uniform(*noise.outage_hours, N), outage_left)
            on = np.clip(1.0 - np.minimum(outage_left, dt_hours) / dt_hours, 0.0, 1.0)
            outage_left = np.maximum(outage_left - dt_hours, 0.0)
        mag_err = 1.0 + noise.thrust_mag_pct * rng.standard_normal(N)
        if noise.pointing_deg > 0:
            direction = direction + np.radians(noise.pointing_deg) * rng.standard_normal((N, 3))
            direction = direction / P.norm(direction)

        thrust = thr * T0 * h_true * mag_err * on
        mdot = thr * MDOT * on

        # ---------- physics ----------
        r, v, m = P.step(r, v, m, thrust, direction, mdot, dt, DRY_MASS)

        # ---------- how close to Mars? ----------
        mars_r, mars_v = P.mars_state(t + dt)
        d_now = mars_r[None, :] - r
        dist = closest_approach(d_prev, d_now)
        better = (dist < min_dist) & ~captured
        min_dist = np.where(better, dist, min_dist)
        min_hour = np.where(better, hour + dt_hours, min_hour)
        vrel_min = np.where(better, P.norm(mars_v[None, :] - v)[:, 0], vrel_min)
        new_cap = (dist < P.CAPTURE_RADIUS_KM) & ~captured
        capture_hour = np.where(new_cap, hour + dt_hours, capture_hour)
        captured |= new_cap
        d_prev = d_now

        if record:
            traj[k + 1] = r
            cmds[k, :, 0] = thr
            cmds[k, :, 1:] = direction
            dist_hist[k + 1] = np.sqrt(np.sum(d_now ** 2, -1))

    out = {
        "min_dist_km": min_dist,
        "min_hour": min_hour,
        "captured": captured,
        "capture_hour": capture_hour,
        "fuel_used_kg": WET_MASS - m,
        "vrel_km_s": vrel_min,
        "piloted_hours": piloted_hours,
    }
    if record:
        out.update(traj=traj, cmds=cmds, dist_hist=dist_hist)
    return out


def fitness(res, fuel_weight=0.3):
    """
    One score per mission (higher = better), used to train the pilot.

        score = -log10( max(closest distance, 150,000 km) / 577,000 km )
                + fuel_weight * (fraction of fuel left, only if captured)

    log10 makes "10x closer" always worth the same +1 point, whether you
    go from 100 million km to 10 million km or from 5 million to 500 thousand.
    Any capture scores >= 0; misses score < 0.
    """
    d = np.maximum(res["min_dist_km"], 150_000.0)
    score = -np.log10(d / P.CAPTURE_RADIUS_KM)
    fuel_left = 1.0 - res["fuel_used_kg"] / (WET_MASS - DRY_MASS)
    return score + fuel_weight * fuel_left * res["captured"]


def record_plan(theta, dt_hours=DT_HOURS):
    """Fly the pilot once on a healthy thruster and save its commands as the fixed PLAN."""
    res = fly(theta, Scenario.healthy(1), dt_hours=dt_hours, record=True)
    return {"throttle": res["cmds"][:, 0, 0].copy(),
            "direction": res["cmds"][:, 0, 1:].copy(),
            "result": res}
