"""
trainer.py: teach the pilot using finite-difference policy gradients with a
proximal safety cap (Evolution Strategies + PPO trust-region clipping).

1. **Numerical Derivative (AP Calculus AB)**:
   The symmetric difference quotient approximates the derivative of mission
   score J(theta) along direction eps:

       dJ/d_eps ~ [ J(theta + sigma*eps) - J(theta - sigma*eps) ] / (2 * sigma)

   By testing P random directions eps_1, ..., eps_P and averaging, we estimate
   the gradient (the steepest uphill direction in weight space; Salimans et al., 2017).

2. **Advantage Baseline Subtraction (Schulman et al., 2016 [#27]; Mnih et al., 2016 [#30])**:
   Instead of raw scores, we center and rank-normalize returns around the
   generation baseline (mean = 0, range [-0.5, +0.5]). Subtracting the baseline
   reduces trial-and-error variance across long 460-day trajectories (#27, #30).

3. **Proximal Safety Cap / Clipped Policy Update (Schulman et al., 2017 [#19])**:
   In Calculus AB, a tangent-line approximation is only accurate *close* to the
   current point. If a single update step is too large, the steering policy can
   become unstable (#19). We place a strict safety cap (`clip_range`) on each
   parameter update:

       delta_theta = clip(learning_rate * grad, -clip_range, +clip_range)
       theta_new   = theta + delta_theta
"""
import time
import numpy as np

from mission import simulate as S


def centered_ranks(x):
    """
    Center and scale returns around a zero baseline in [-0.5, +0.5].
    Acts as a variance-reduced advantage estimate (#27, #30).
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
    clip_range       : proximal safety cap on per-step weight changes (#19 Schulman et al. 2017)
    n_scen           : missions flown by EVERY candidate per generation (fixed controls, #28)
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

        # Proximal clipped update (#19 Schulman et al., 2017)
        step = learning_rate * grad
        if clip_range is not None:
            step = np.clip(step, -clip_range, clip_range)
        theta = theta + step

    return best_theta, history
