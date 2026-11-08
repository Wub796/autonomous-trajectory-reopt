"""
thruster.py: the SPT-140 Hall-effect thruster, how it wears out, and how it can fail.

Three ideas live here:

1. **Nominal numbers** (same as the original project; Hargus & Fife 2000):
   thrust T0 = 0.289 N, specific impulse Isp0 = 1782 s.

2. **Slow wear** (aging).  Lab wear tests (Kamhawi et al. 2014) show thrust
   dropping a little at first and then more and more slowly.  A natural
   function with that shape is a logarithm:

        T(h) = T0 - k * ln(1 + h / tau)

   h = hours of operation, k = how much it wears, tau = how fast wear slows down.
   Its derivative  dT/dh = -k / (tau + h)  is the wear *rate*: it starts at
   -k/tau and shrinks toward 0 as h grows.  We find k and tau from the lab
   data with least-squares regression (fit_wear_model below).

3. **A sudden fault** (the problem our research is about).  At some random
   hour the thruster loses part of its push (e.g. electrode erosion or a
   cathode problem).  The propellant still flows at the same rate, so the
   thruster becomes less *efficient*: same fuel, less push.  The loss grows
   over a "ramp" of a few days and then stays.
"""
from dataclasses import dataclass
import numpy as np

from mission.physics import G0

# ---------------------------------------------------------------------------
# Spacecraft & thruster constants
# ---------------------------------------------------------------------------
T0 = 0.289                    # nominal thrust [N]
ISP0 = 1782.0                 # nominal specific impulse [s]
MDOT = T0 / (ISP0 * G0)       # propellant flow at full throttle [kg/s] (~1.65e-5)
WET_MASS = 2747.0             # spacecraft + full tank [kg]
DRY_MASS = 1648.0             # spacecraft with empty tank [kg]
MISSION_HOURS = 11040         # 460 days, same as the original project

# ---------------------------------------------------------------------------
# Lab wear-test data (approximate digitisation of Kamhawi et al. 2014,
# SPT-140 at 300 V / 4.5 kW).  Same numbers as the original project.
# ---------------------------------------------------------------------------
WEAR_HOURS = np.array([0, 100, 200, 300, 500, 700, 900, 1000, 1500, 2000], dtype=float)
WEAR_THRUST_MN = np.array([280.0, 279.4, 279.1, 278.8, 278.3, 277.9, 277.5, 277.2, 276.3, 275.5])
WEAR_ISP_S = np.array([1782.0, 1781.0, 1780.3, 1779.6, 1778.3, 1777.0, 1775.8, 1775.0, 1772.1, 1769.5])


def linear_regression(x, y):
    """
    Ordinary least-squares line y = a + b x (the AP Statistics formulas):

        b = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2)
        a = y_bar - b x_bar

    These come from setting the derivative of the total squared error
    to zero (see paper/math_primer.md, Section 5).
    """
    xb, yb = x.mean(), y.mean()
    b = np.sum((x - xb) * (y - yb)) / np.sum((x - xb) ** 2)
    a = yb - b * xb
    return a, b


def fit_wear_model(hours, values, tau_grid=None):
    """
    Fit  value(h) = P0 - k ln(1 + h/tau).

    Trick: if we *guess* tau, then x = ln(1 + h/tau) turns the model into a
    straight line  value = P0 - k x,  which linear_regression solves exactly.
    So we try many tau values and keep the one with the smallest
    sum of squared errors (SSE).

    Returns dict with P0, k, tau, sse, r2, and the SSE-vs-tau curve.
    """
    if tau_grid is None:
        tau_grid = np.geomspace(50, 50_000, 600)
    sse_curve = []
    best = None
    for tau in tau_grid:
        x = np.log1p(hours / tau)
        a, b = linear_regression(x, values)
        pred = a + b * x
        sse = float(np.sum((values - pred) ** 2))
        sse_curve.append(sse)
        if best is None or sse < best["sse"]:
            best = {"P0": a, "k": -b, "tau": float(tau), "sse": sse}
    sst = float(np.sum((values - values.mean()) ** 2))
    best["r2"] = 1 - best["sse"] / sst
    best["tau_grid"] = tau_grid
    best["sse_curve"] = np.array(sse_curve)
    return best


def wear_curve(hours, P0, k, tau):
    return P0 - k * np.log1p(np.asarray(hours, dtype=float) / tau)


# Fit once at import time so every module uses the same wear model.
_fit = fit_wear_model(WEAR_HOURS, WEAR_THRUST_MN)
WEAR_K_FRAC = _fit["k"] / _fit["P0"]   # wear as a fraction of thrust per ln-unit
WEAR_TAU = _fit["tau"]                 # [hours]


def wear_fraction(hours):
    """Fraction of the nominal thrust left after `hours` of normal aging (1.0 = new)."""
    return 1.0 - WEAR_K_FRAC * np.log1p(np.asarray(hours, dtype=float) / WEAR_TAU)


# ---------------------------------------------------------------------------
# Fault scenarios
# ---------------------------------------------------------------------------
@dataclass
class Fault:
    onset_hour: float = np.inf    # when the fault starts (inf = never)
    severity: float = 0.0         # fraction of thrust lost once fully developed (0.15 = 15%)
    ramp_hours: float = 0.0       # how long the loss takes to fully develop


def fault_fraction(hours, onset_hour, severity, ramp_hours):
    """
    Fraction of thrust left because of the fault (1.0 = no fault).
    Linear ramp from 0 loss at onset to full `severity` after `ramp_hours`.
    Works with numpy broadcasting (e.g. hours shape (T,), others shape (S, 1)).
    """
    hours = np.asarray(hours, dtype=float)
    progress = np.clip((hours - onset_hour) / np.maximum(ramp_hours, 1e-9), 0.0, 1.0)
    return 1.0 - severity * progress


def health(hours, onset_hour=np.inf, severity=0.0, ramp_hours=0.0):
    """Total thrust fraction available = (aging) x (fault)."""
    return wear_fraction(hours) * fault_fraction(hours, onset_hour, severity, ramp_hours)


def sample_faults(rng, n, p_fault=1.0, onset_days=(20, 200), severity=(0.05, 0.30), ramp_days=(0, 10)):
    """Draw n random fault scenarios.  With probability 1 - p_fault there is no fault."""
    has = rng.random(n) < p_fault
    onset = np.where(has, rng.uniform(*onset_days, n) * 24.0, np.inf)
    sev = np.where(has, rng.uniform(*severity, n), 0.0)
    ramp = rng.uniform(*ramp_days, n) * 24.0
    return onset, sev, ramp


# ---------------------------------------------------------------------------
# Telemetry: what the onboard sensors report every hour
# ---------------------------------------------------------------------------
THRUST_SENSOR_NOISE = 0.01     # 1% (1-sigma) measurement noise on thrust
ISP_SENSOR_NOISE = 0.01        # 1% on the derived Isp
TEMP_MEAN, TEMP_NOISE = 45.0, 0.5   # solar-array temperature [deg C] (nuisance signal)


def make_telemetry(rng, onset_hour, severity, ramp_hours, hours=MISSION_HOURS):
    """
    Hourly telemetry for S scenarios at once.

    Returns dict of arrays with shape (S, hours):
      true_frac  : true thrust fraction (aging x fault)
      thrust_mN  : measured thrust (noisy)
      isp_s      : measured specific impulse (noisy).  Same fuel flow, less push
                   means Isp drops by the same fraction as thrust.
      temp_C     : solar array temperature (just noise, should NOT trigger alarms)
      faulty     : True where the fault has started (the "answer key")
    """
    onset = np.atleast_1d(onset_hour)[:, None]
    sev = np.atleast_1d(severity)[:, None]
    ramp = np.atleast_1d(ramp_hours)[:, None]
    h = np.arange(hours, dtype=float)[None, :]
    S = onset.shape[0]

    frac = health(h, onset, sev, ramp)
    thrust = T0 * 1000 * frac * (1 + THRUST_SENSOR_NOISE * rng.standard_normal((S, hours)))
    isp = ISP0 * frac * (1 + ISP_SENSOR_NOISE * rng.standard_normal((S, hours)))
    temp = TEMP_MEAN + TEMP_NOISE * rng.standard_normal((S, hours))
    faulty = (h >= onset) & (sev > 0)
    return {"true_frac": frac, "thrust_mN": thrust, "isp_s": isp, "temp_C": temp,
            "faulty": np.broadcast_to(faulty, (S, hours))}
