"""
detector.py: the onboard "is my thruster OK?" monitor.

Why single sensor readings are not enough (Chandola et al., 2009 [#31]):
  Slow thruster wear is a *collective anomaly*. Any single hourly reading
  is buried in sensor noise and thermal fluctuations, whereas a sequence of
  readings over a 24-hour window reveals the collective downward drift.

We implement and compare three detectors at the AP Stats / intro ML level:

1. **PointIsolationForestDetector (w = 1 baseline)**:
   Feeds raw single-hour readings [thrust, Isp, solar_temp] into an Isolation
   Forest without time-window feature extraction. Demonstrates Chandola et al.
   (#31): without collective windowing, slow wear is frequently missed.

2. **ZScoreDetector (w = 24, AP Statistics rolling-window detector)**:
   - Predicts expected thrust from the wear curve: expected(h) = T0 * wear_fraction(h)
   - Computes residual: e(h) = measured(h) / expected(h) - 1
   - Averages the last w = 24 hours. By AP Stats, averaging w independent readings
     shrinks the noise standard deviation by sqrt(w) (Standard Error of the Mean).
   - Computes z = rolling_mean / (sigma / sqrt(w)).
   - Flags an alarm when z < -threshold for `persist` hours in a row.

3. **CollectiveIsolationForestDetector (w = 24, Fu et al., 2022 [#20]; Liu et al., 2008 [#26])**:
   Combines rolling-window feature extraction (#20, #31) with an unsupervised
   Isolation Forest of random binary decision trees (#26):
   - Feature 1: 24-hour rolling mean of wear-corrected thrust residual
   - Feature 2: 24-hour rolling mean of wear-corrected Isp residual
   - Feature 3: 24-hour secant slope (average rate of change de/dt over window w,
                from AP Calculus AB)
   - Feature 4: Solar-array temperature (included to verify that normal thermal
                fluctuations do not trick the detector into false alarms, #20)
   Trained only on the first 240 healthy hours of flight (no labeled failures needed, #1, #20).

After an alarm fires, the detector estimates remaining thruster health from the
24-hour rolling mean of measured / T0 and passes that belief to the AI pilot.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from mission.thruster import T0, ISP0, wear_fraction


def rolling_mean(x, w):
    """Causal rolling mean along the last axis (uses only past + current samples)."""
    c = np.cumsum(x, axis=-1)
    out = c.copy()
    out[..., w:] = c[..., w:] - c[..., :-w]
    counts = np.minimum(np.arange(1, x.shape[-1] + 1), w)
    return out / counts


def rolling_slope(x, w):
    """
    Causal secant slope (average rate of change over the last w hours):
        slope(t) = [ x(t) - x(t - w) ] / w
    In AP Calculus AB, this is the difference quotient over interval w.
    """
    out = np.zeros_like(x, dtype=float)
    out[..., w:] = (x[..., w:] - x[..., :-w]) / float(w)
    if w > 1:
        idx = np.arange(1, w)
        out[..., 1:w] = (x[..., 1:w] - x[..., :1]) / idx
    return out


def first_true_run(mask, persist):
    """
    Index of the first place where `mask` is True for `persist` samples in a row.
    Returns the index of the *last* sample of that run (when the alarm actually fires).
    np.inf if it never happens.  mask shape (S, H).
    """
    if persist <= 1:
        run_ok = mask
    else:
        run_ok = rolling_mean(mask.astype(float), persist) >= 1.0 - 1e-9
        run_ok[:, :persist - 1] = False
    any_ = run_ok.any(axis=1)
    idx = np.argmax(run_ok, axis=1).astype(float)
    idx[~any_] = np.inf
    return idx


def _believed_health(thrust_mN, alarm_hour, window=24):
    """Before alarm: nominal aging curve. After alarm: rolling measured thrust fraction."""
    H = thrust_mN.shape[-1]
    h = np.arange(H)
    believed = np.broadcast_to(wear_fraction(h), thrust_mN.shape).copy()
    measured = rolling_mean(thrust_mN / (T0 * 1000.0), window)
    after = h[None, :] >= alarm_hour[:, None]
    believed[after] = measured[after]
    return believed


class ZScoreDetector:
    def __init__(self, window=24, threshold=3.5, persist=3, baseline_hours=240, use_wear_model=True):
        self.window = window
        self.threshold = threshold
        self.persist = persist
        self.baseline_hours = baseline_hours
        self.use_wear_model = use_wear_model

    def residuals(self, thrust_mN):
        H = thrust_mN.shape[-1]
        h = np.arange(H)
        if self.use_wear_model:
            expected = T0 * 1000.0 * wear_fraction(h)
        else:
            expected = thrust_mN[..., :self.baseline_hours].mean(axis=-1, keepdims=True)
        return thrust_mN / expected - 1.0

    def scores(self, thrust_mN):
        """z-score for every hour, shape (S, H). Very negative = thrust too low."""
        e = self.residuals(thrust_mN)
        sigma = e[..., :self.baseline_hours].std(axis=-1, keepdims=True)
        m = rolling_mean(e, self.window)
        n = np.minimum(np.arange(1, e.shape[-1] + 1), self.window)
        return m / (sigma / np.sqrt(n))

    def run(self, thrust_mN):
        """
        Returns
          alarm_hour : first alarm per scenario (np.inf if none), shape (S,)
          alarms     : raw per-hour alarm flags (z < -threshold), shape (S, H)
          est_frac   : what the spacecraft *believes* its thrust fraction is,
                       shape (S, H). Before the alarm: wear model. After: measured.
        """
        thrust_mN = np.atleast_2d(thrust_mN)
        z = self.scores(thrust_mN)
        alarms = z < -self.threshold
        alarm_hour = first_true_run(alarms, self.persist)
        believed = _believed_health(thrust_mN, alarm_hour, self.window)
        return alarm_hour, alarms, believed


class PointIsolationForestDetector:
    """
    Point-only Isolation Forest (w = 1): feeds raw single-hour sensor readings
    without rolling-window feature extraction (#31 Chandola et al., 2009).
    """

    def __init__(self, contamination=0.01, baseline_hours=240, persist=3, seed=0):
        self.contamination = contamination
        self.baseline_hours = baseline_hours
        self.persist = persist
        self.seed = seed

    @staticmethod
    def features(tel, s):
        return np.column_stack([tel["thrust_mN"][s], tel["isp_s"][s], tel["temp_C"][s]])

    def run(self, tel):
        S, H = tel["thrust_mN"].shape
        alarms = np.zeros((S, H), dtype=bool)
        for s in range(S):
            X = self.features(tel, s)
            model = IsolationForest(contamination=self.contamination, random_state=self.seed)
            model.fit(X[:self.baseline_hours])
            alarms[s] = model.predict(X) == -1
        alarm_hour = first_true_run(alarms, self.persist)
        believed = _believed_health(tel["thrust_mN"], alarm_hour, window=24)
        return alarm_hour, alarms, believed


class IsolationForestDetector:
    """
    Collective-anomaly Isolation Forest (Fu et al., 2022 [#20]; Liu et al., 2008 [#26];
    Chandola et al., 2009 [#31]).

    Extracts rolling time-series features over `window` hours so random trees can
    isolate gradual thruster decay without being fooled by single-hour noise or
    normal solar-array temperature fluctuations.
    """

    def __init__(self, window=24, contamination=0.005, baseline_hours=240,
                 persist=4, n_estimators=50, seed=0):
        self.window = window
        self.contamination = contamination
        self.baseline_hours = baseline_hours
        self.persist = persist
        self.n_estimators = n_estimators
        self.seed = seed

    def features(self, tel, s):
        H = tel["thrust_mN"].shape[-1]
        h = np.arange(H)
        w_frac = wear_fraction(h)

        # Wear-corrected residuals for thrust and Isp
        e_thrust = tel["thrust_mN"][s] / (T0 * 1000.0 * w_frac) - 1.0
        e_isp = tel["isp_s"][s] / (ISP0 * w_frac) - 1.0

        # Collective window features (#20 Fu et al., #31 Chandola et al.)
        roll_thrust = rolling_mean(e_thrust, self.window)
        roll_isp = rolling_mean(e_isp, self.window)
        slope_thrust = rolling_slope(roll_thrust, self.window) * 100.0
        temp_norm = (tel["temp_C"][s] - 45.0) / 5.0

        return np.column_stack([roll_thrust, roll_isp, slope_thrust, temp_norm])

    def run(self, tel):
        """
        Returns
          alarm_hour : first alarm per scenario (np.inf if none), shape (S,)
          alarms     : per-hour anomaly flags, shape (S, H)
          believed   : believed thrust fraction (S, H) for the AI pilot
        """
        S, H = tel["thrust_mN"].shape
        alarms = np.zeros((S, H), dtype=bool)
        for s in range(S):
            X = self.features(tel, s)
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.seed
            )
            # Fit on healthy baseline window (after initial rolling warm-up)
            model.fit(X[self.window:self.baseline_hours])
            # Flag collective drops in thrust (roll_thrust < 0), ignoring upward noise spikes
            raw_flag = model.predict(X) == -1
            alarms[s] = raw_flag & (X[:, 0] < -0.015)

        alarm_hour = first_true_run(alarms, self.persist)
        believed = _believed_health(tel["thrust_mN"], alarm_hour, self.window)
        return alarm_hour, alarms, believed
