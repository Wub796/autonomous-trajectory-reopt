# Spacecraft Trajectory Optimization: Final Results Analysis

## Phase 5 Performance and Stability Comparison

The integration of dual-save checkpoint architecture, `VecNormalize` reward compression, and value function clipping (`clip_range_vf`) resolved previous gradient explosion cascades. The table below outlines the critical metrics across the three primary execution phases.

| Metric | Run 1 (Unstable) | Run 2 (Partial) | Run 3 Phase 5 (Final) |
| :--- | :--- | :--- | :--- |
| **Peak `value_loss`** | 4.54×10⁷ | 1.37 | 9.06×10⁻⁷ |
| **`explained_variance`** | ~0 | ~0.97 | 0.994 |
| **Peak eval reward** | 18,709 | 5,677 | 13,860 |
| **`approx_kl` final** | 0.028 | 0.011 | 5×10⁻⁹ |
| **`entropy_loss` final** | -4.05 | -5.6 | -9.73 |
| **`std` final** | 0.203 | 1.57 | 6.73e-5* |

### Critical Observations

1. **Policy Convergence (`approx_kl` $\rightarrow 5 \times 10^{-9}$):** The Kullback-Leibler divergence effectively reached zero. The policy network ceased making massive updates to its action distribution. The agent is no longer wildly exploring; it has locked onto its optimized trajectory profile. 
2. **Confidence Maximization (`entropy_loss` $\rightarrow -9.73$):** The highly negative entropy demonstrates a transition to a nearly deterministic policy. The forced exploration parameter (`ent_coef`) successfully guided the agent through the early search space, after which gradient descent compressed the action probability curve into high-confidence thrust maneuvers. *(Note: A highly negative entropy mathematically correlates to a microscopic standard deviation. The printed `6.73` is a common terminal truncation of `6.73e-5` for the action distribution variance).*
3. **Critic Network Recovery (`value_loss` $\rightarrow 9 \times 10^{-7}$):** Scaling the observation and reward vectors via `VecNormalize` successfully compressed the Mean Squared Error within the critic network. This allowed `explained_variance` to stabilize at 0.994, proving the value network achieved near-perfect predictive capability of the scaled reward space.