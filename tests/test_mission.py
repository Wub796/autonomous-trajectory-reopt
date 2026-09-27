"""
Unit tests for high-school level mission codebase.
Verifies physics, thruster models, detector logic, neural pilot, and simulation engine.
"""
import unittest
import numpy as np

from mission import physics as P
from mission import thruster as T
from mission import detector as D
from mission import pilot as AI
from mission import simulate as S
from mission import trainer as TR


class TestPhysics(unittest.TestCase):
    def test_earth_launch_state(self):
        r, v = P.earth_launch_state(1)
        self.assertEqual(r.shape, (1, 3))
        self.assertEqual(v.shape, (1, 3))
        # Radius ~ 1 AU
        self.assertAlmostEqual(P.norm(r)[0, 0], P.EARTH_R, places=2)
        # Velocity includes Earth orbital speed + v-inf
        self.assertAlmostEqual(P.norm(v)[0, 0], P.EARTH_SPEED + P.LAUNCH_VINF, places=2)

    def test_mars_state(self):
        r0, v0 = P.mars_state(0.0)
        self.assertEqual(r0.shape, (3,))
        self.assertEqual(v0.shape, (3,))
        # Orbit radius ~ 1.524 AU
        self.assertAlmostEqual(np.linalg.norm(r0), P.MARS_R, places=-2)
        # 3D inclination: z is non-zero
        self.assertNotEqual(r0[2], 0.0)

    def test_energy_conservation_ballistic(self):
        # Without thrust, orbital energy should remain nearly constant over a few orbits
        r, v = P.earth_launch_state(1)
        m = np.array([T.WET_MASS])
        E0 = P.specific_energy(r, v)[0]
        dt = 3600.0  # 1 hour
        for _ in range(240):  # 10 days
            rh, th, nh = P.local_frame(r, v)
            r, v, m = P.step(r, v, m, np.zeros(1), th, np.zeros(1), dt, T.DRY_MASS)
        E1 = P.specific_energy(r, v)[0]
        rel_diff = abs(E1 - E0) / abs(E0)
        self.assertLess(rel_diff, 1e-4)

    def test_local_frame_orthonormality(self):
        r = np.array([[1.0e8, 5.0e7, -2.0e7]])
        v = np.array([[-10.0, 25.0, 5.0]])
        rh, th, nh = P.local_frame(r, v)
        # Unit lengths
        self.assertAlmostEqual(P.norm(rh)[0, 0], 1.0, places=6)
        self.assertAlmostEqual(P.norm(th)[0, 0], 1.0, places=6)
        self.assertAlmostEqual(P.norm(nh)[0, 0], 1.0, places=6)
        # Mutually perpendicular
        self.assertAlmostEqual(np.sum(rh * th), 0.0, places=6)
        self.assertAlmostEqual(np.sum(rh * nh), 0.0, places=6)
        self.assertAlmostEqual(np.sum(th * nh), 0.0, places=6)


class TestThruster(unittest.TestCase):
    def test_linear_regression(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 3.0
        a, b = T.linear_regression(x, y)
        self.assertAlmostEqual(a, 3.0, places=6)
        self.assertAlmostEqual(b, 2.0, places=6)

    def test_wear_model_fit(self):
        fit = T.fit_wear_model(T.WEAR_HOURS, T.WEAR_THRUST_MN)
        self.assertGreater(fit["r2"], 0.99)
        self.assertGreater(fit["P0"], 279.0)
        self.assertGreater(fit["k"], 0.0)
        self.assertGreater(fit["tau"], 100.0)

    def test_fault_model(self):
        h = np.array([0.0, 100.0, 150.0, 200.0])
        frac = T.fault_fraction(h, onset_hour=100.0, severity=0.20, ramp_hours=50.0)
        self.assertAlmostEqual(frac[0], 1.0)
        self.assertAlmostEqual(frac[1], 1.0)
        self.assertAlmostEqual(frac[2], 0.80)
        self.assertAlmostEqual(frac[3], 0.80)

    def test_telemetry_generation(self):
        rng = np.random.default_rng(42)
        tel = T.make_telemetry(rng, onset_hour=1000.0, severity=0.15, ramp_hours=50.0, hours=2000)
        self.assertIn("thrust_mN", tel)
        self.assertIn("isp_s", tel)
        self.assertEqual(tel["thrust_mN"].shape, (1, 2000))
        # Pre-fault average should be around nominal thrust (289 mN or 280 mN empirical base)
        pre_mean = tel["thrust_mN"][0, :500].mean()
        self.assertGreater(pre_mean, 270.0)


class TestDetector(unittest.TestCase):
    def test_zscore_detector_alarm(self):
        rng = np.random.default_rng(42)
        tel = T.make_telemetry(rng, onset_hour=500.0, severity=0.20, ramp_hours=10.0, hours=1000)
        det = D.ZScoreDetector(window=24, threshold=3.5, persist=3, baseline_hours=240)
        alarm_hr, alarms, believed = det.run(tel["thrust_mN"])
        # Should flag an alarm after onset_hour=500
        self.assertFalse(np.isinf(alarm_hr[0]))
        self.assertGreaterEqual(alarm_hr[0], 500.0)
        self.assertLess(alarm_hr[0], 560.0)
        # Believed health should drop post-alarm
        self.assertLess(believed[0, -1], 0.85)

    def test_zscore_detector_nominal_no_false_alarm(self):
        rng = np.random.default_rng(42)
        tel = T.make_telemetry(rng, onset_hour=np.inf, severity=0.0, ramp_hours=0.0, hours=1000)
        det = D.ZScoreDetector(window=24, threshold=4.5, persist=4, baseline_hours=240)
        alarm_hr, _, _ = det.run(tel["thrust_mN"])
        self.assertTrue(np.isinf(alarm_hr[0]))

    def test_collective_isolation_forest_alarm(self):
        rng = np.random.default_rng(42)
        tel = T.make_telemetry(rng, onset_hour=500.0, severity=0.20, ramp_hours=10.0, hours=1000)
        det = D.IsolationForestDetector(window=24, contamination=0.005, baseline_hours=240, persist=4, seed=42)
        alarm_hr, alarms, believed = det.run(tel)
        self.assertFalse(np.isinf(alarm_hr[0]))
        self.assertGreaterEqual(alarm_hr[0], 500.0)
        self.assertLess(alarm_hr[0], 560.0)
        self.assertLess(believed[0, -1], 0.85)


class TestPilotAndSimulate(unittest.TestCase):
    def test_neural_pilot_forward(self):
        rng = np.random.default_rng(0)
        theta = AI.init_params(rng)
        self.assertEqual(theta.shape, (AI.N_PARAMS,))

        obs = np.ones((2, AI.N_IN))
        thr, alpha, beta = AI.act(np.tile(theta, (2, 1)), obs)
        self.assertEqual(thr.shape, (2,))
        self.assertEqual(alpha.shape, (2,))
        self.assertEqual(beta.shape, (2,))
        self.assertTrue(np.all((thr >= 0.0) & (thr <= 1.0)))

        # Compare with act_single
        thr_s, a_s, b_s = AI.act_single(theta, obs[0])
        self.assertAlmostEqual(thr[0], thr_s, places=6)
        self.assertAlmostEqual(alpha[0], a_s, places=6)
        self.assertAlmostEqual(beta[0], b_s, places=6)

    def test_simulate_fly(self):
        rng = np.random.default_rng(0)
        theta = AI.init_params(rng)
        scen = S.Scenario.healthy(2)
        res = S.fly(theta, scen, dt_hours=24)
        self.assertEqual(res["min_dist_km"].shape, (2,))
        self.assertEqual(res["fuel_used_kg"].shape, (2,))
        self.assertGreater(res["min_dist_km"][0], 0.0)
        self.assertGreater(res["fuel_used_kg"][0], 0.0)

    def test_trainer_ranks(self):
        scores = np.array([10.0, 50.0, 20.0, 100.0])
        ranks = TR.centered_ranks(scores)
        self.assertEqual(ranks.shape, (4,))
        self.assertAlmostEqual(ranks[0], -0.5)
        self.assertAlmostEqual(ranks[3], +0.5)


if __name__ == "__main__":
    unittest.main()
