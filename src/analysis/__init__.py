"""
src/analysis module — Monte Carlo robustness, parametric sensitivity, and publication plotting.
"""
from src.analysis.monte_carlo import run_monte_carlo_suite, run_single_dispersed_trajectory
from src.analysis.sensitivity import run_sensitivity_sweep
from src.analysis.plotting import (
    plot_monte_carlo_dispersions_2d,
    plot_monte_carlo_histograms,
    plot_sensitivity_tornado,
    plot_scenario_comparison,
    plot_failure_mode_breakdown,
    plot_sensitivity_curves,
    plot_nominal_deviation,
)

__all__ = [
    "run_monte_carlo_suite",
    "run_single_dispersed_trajectory",
    "run_sensitivity_sweep",
    "plot_monte_carlo_dispersions_2d",
    "plot_monte_carlo_histograms",
    "plot_sensitivity_tornado",
    "plot_scenario_comparison",
    "plot_failure_mode_breakdown",
    "plot_sensitivity_curves",
    "plot_nominal_deviation",
]
