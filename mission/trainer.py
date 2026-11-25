"""
trainer.py: teach the pilot with finite-difference evolution strategies and a
PPO-inspired per-parameter update cap.

1. Numerical derivative:
   The symmetric difference quotient approximates the derivative of mission
   score J(theta) along direction eps:

       dJ/d_eps ~ [ J(theta + sigma*eps) - J(theta - sigma*eps) ] / (2 * sigma)

   By testing P random directions eps_1, ..., eps_P and averaging, we estimate
   the gradient (the steepest uphill direction in weight space; Salimans et al., 2017).

2. Rank baseline (variance reduction; Schulman et al., 2016; Mnih et al., 2016):
   Instead of raw scores, we center and rank-normalize returns around the
   generation baseline (mean = 0, range [-0.5, +0.5]). This is a simple rank
   baseline, not GAE or an actor-critic advantage; it reduces trial-and-error
   variance across long 460-day trajectories.

3. PPO-inspired per-parameter update cap (Schulman et al., 2017):
   A tangent-line approximation is only accurate *close* to the
   current point. If a single update step is too large, the steering policy can
   become unstable. We borrow only the idea of a trust-region-style cap:
   a strict per-parameter limit (`clip_range`) on each update. This is not PPO's
   ratio clipping, and no PPO objective is computed:

       delta_theta = clip(learning_rate * grad, -clip_range, +clip_range)
       theta_new   = theta + delta_theta
"""
import time
import numpy as np

from mission import pilot as policy
from mission import simulate as S


def centered_ranks(x):
    """
    Center and scale returns around a zero baseline in [-0.5, +0.5].
    Acts as a simple rank baseline for variance reduction; it is not GAE.
    """
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def train_es(theta0, sample_scenarios, generations=200, pairs=24, sigma=0.05,
             learning_rate=0.03, clip_range=0.05, n_scen=4, plan=None, noise=None,
             seed=0, eval_fn=None, eval_every=10, log=print):
    """
    theta0           : starting weights (243,)
    sample_scenarios : function(rng, n) -> simulate.Scenario with n scenarios
    pairs            : number of +/- direction pairs per generation
    clip_range       : PPO-inspired per-parameter cap on each weight change (Schulman et al., 2017)
    n_scen           : missions flown by EVERY candidate per generation (fixed controls)
    eval_fn          : optional function(theta) -> dict of numbers to log
    Returns (best theta, history list of dicts).
    """
    rng = np.random.default_rng(seed)
    theta = theta0.copy()
    n = theta.size
    history = []
    best_theta, best_eval = theta.copy(), -np.inf
    t_start = time.time()

    for gen in range(generations + 1):
        if eval_fn is not None and (gen % eval_every == 0 or gen == generations):
            ev = eval_fn(theta)
            ev.update(generation=gen, seconds=time.time() - t_start)
            history.append(ev)
            log("  gen {generation:4d} | eval score {score:+.3f} | capture {capture_rate:5.1%} | "
                "median miss {median_miss_km:,.0f} km | {seconds:5.0f}s".format(**ev))
            if ev["score"] > best_eval:
                best_eval, best_theta = ev["score"], theta.copy()
        if gen == generations:
            break

        eps = rng.standard_normal((pairs, n))
        candidates = np.concatenate([theta + sigma * eps, theta - sigma * eps])   # (2*pairs, n)
        P = candidates.shape[0]

        scen = sample_scenarios(rng, n_scen)                 # n_scen missions
        # Layout: row j = candidate (j // n_scen) flying scenario (j % n_scen)
        grid = scen.tile(P)                                  # [s0, s1, .., s0, s1, ..]
        thetas = np.repeat(candidates, n_scen, axis=0)       # [c0, c0, .., c1, c1, ..]
        res = S.fly(thetas, grid, plan=plan, noise=noise, rng=rng)
        f = S.fitness(res).reshape(P, n_scen).mean(axis=1)   # average score per candidate

        u = centered_ranks(f)
        u_plus, u_minus = u[:pairs], u[pairs:]
        # Finite-difference gradient estimate: sum of [f(+) - f(-)] / (2 sigma) * eps
        grad = ((u_plus - u_minus)[:, None] * eps).sum(axis=0) / (2 * pairs * sigma)

        # PPO-inspired per-parameter cap; not PPO ratio clipping (Schulman et al., 2017)
        step = learning_rate * grad
        if clip_range is not None:
            step = np.clip(step, -clip_range, clip_range)
        theta = theta + step

    return best_theta, history


def train_policy(seed, stage1_seed, stage2_seed, validation_seed=999, log=print):
    """Run the two-stage training recipe shared by Experiments 3 and 7.

    Stage 1 optimises the nominal (healthy) mission for 400 generations and
    records its commands as the 920-step open-loop plan.  Stage 2 fine-tunes
    for 200 generations on faults that remove 10-25% of thrust, begin on days
    30-150, and ramp over 24-72 hours; the policy takes control at onset with
    the true health value, so adaptation is tested separately from detection.

    seed            : policy initialisation
    stage1_seed     : Stage 1 ES seed
    stage2_seed     : Stage 2 ES seed
    validation_seed : fixed ten-scenario validation set used for Stage 2 logging

    Returns (theta, plan, history_stage1, history_stage2).
    """
    theta0 = policy.init_params(np.random.default_rng(seed), scale=0.4)

    def evaluate(theta):
        res = S.fly(theta, S.Scenario.healthy(1), dt_hours=S.DT_HOURS)
        return {"score": float(S.fitness(res)[0]),
                "capture_rate": float(res["captured"][0]),
                "median_miss_km": float(res["min_dist_km"][0]),
                "fuel_used_kg": float(res["fuel_used_kg"][0])}

    def sample_nominal(r, n):
        return S.Scenario.healthy(n)

    log("\n--- Stage 1: Nominal Flight Optimization (400 generations) ---")
    best1, hist1 = train_es(
        theta0=theta0, sample_scenarios=sample_nominal, generations=400,
        pairs=64, sigma=0.035, learning_rate=0.025, n_scen=1,
        eval_fn=evaluate, eval_every=20, seed=stage1_seed, log=log)

    # Record the nominal reference plan from the best nominal pilot.
    plan = S.record_plan(best1, dt_hours=S.DT_HOURS)

    def sample_degraded(r, n):
        # Thruster loses 10-25% of thrust between day 30 and 150.
        onset = r.uniform(30 * 24, 150 * 24, n)
        sev = r.uniform(0.10, 0.25, n)
        ramp = r.uniform(24, 72, n)
        scen = S.Scenario(onset, sev, ramp)
        # Assume perfect detector feedback in training: believed health follows
        # the degraded value after onset, and the pilot takes control at onset.
        h = np.arange(S.MISSION_HOURS)
        believed = np.ones((n, S.MISSION_HOURS))
        for i in range(n):
            believed[i, h >= onset[i]] = 1.0 - sev[i]
        scen.believed = believed
        scen.takeover_hour = onset
        return scen

    def eval_robust(theta):
        # Ten fixed validation degradation scenarios.
        val_scen = sample_degraded(np.random.default_rng(validation_seed), 10)
        res = S.fly(theta, val_scen, plan=plan, dt_hours=S.DT_HOURS)
        return {"score": float(np.mean(S.fitness(res))),
                "capture_rate": float(np.mean(res["captured"])),
                "median_miss_km": float(np.median(res["min_dist_km"])),
                "fuel_used_kg": float(np.mean(res["fuel_used_kg"]))}

    log("\n--- Stage 2: Degradation Adaptation Training (200 generations) ---")
    best2, hist2 = train_es(
        theta0=best1, sample_scenarios=sample_degraded, generations=200,
        pairs=64, sigma=0.03, learning_rate=0.02, n_scen=2, plan=plan,
        eval_fn=eval_robust, eval_every=20, seed=stage2_seed, log=log)

    return best2, plan, hist1, hist2
