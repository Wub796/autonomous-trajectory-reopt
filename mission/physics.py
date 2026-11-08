"""
physics.py: the "world" the spacecraft lives in.

Everything here is high-school physics + Calc AB:

* Newton's law of gravity (inverse-square law) in 3D
* F = m a  for the thruster push
* Velocity is the derivative of position, acceleration is the derivative of
  velocity.  We step forward in time with the tangent-line (linear)
  approximation:

        v(t + dt) ~ v(t) + a(t) * dt          (because dv/dt = a)
        r(t + dt) ~ r(t) + v(t + dt) * dt     (because dr/dt = v)

  Using the *new* velocity to move the position is called "semi-implicit
  Euler".  It keeps orbits from slowly spiraling outward the way the plain
  version does (see paper/math_primer.md, Section 2).

* The propellant use rate dm/dt = -T / (Isp * g0).

All arrays are "batched": shape (N, 3) for N spacecraft flying at the same
time.  That lets us simulate a whole population of AI pilots at once.

Units: kilometers, seconds, kilograms, Newtons.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_SUN = 1.32712440018e11      # G * M_sun   [km^3 / s^2]
AU = 1.495978707e8             # 1 astronomical unit [km]
G0 = 9.80665                   # standard gravity [m/s^2]  (used in Isp formula)
HOUR = 3600.0                  # [s]
DAY = 86400.0                  # [s]

# Earth: circular orbit, radius 1 AU, in the ecliptic plane (z = 0)
EARTH_R = 1.0 * AU
EARTH_SPEED = np.sqrt(MU_SUN / EARTH_R)          # ~29.78 km/s (from v^2/r = GM/r^2)

# Mars: circular orbit, radius 1.524 AU, tilted 1.85 degrees.
MARS_R = 1.523679 * AU
MARS_PERIOD = 686.98 * DAY
MARS_OMEGA = 2 * np.pi / MARS_PERIOD             # angular speed [rad/s]
MARS_INCLINATION = np.radians(1.85)              # tilt of Mars' orbit
MARS_NODE = np.radians(49.6)                     # where the tilted orbit crosses z = 0
# Mars' starting angle (measured along its orbit from the node line).
# Chosen so Mars arrives at heliocentric longitude ~203 deg around day 290,
# which is where a ~130-day thrust arc launched at t = 0 reaches 1.524 AU
# (see experiments/00_mission_design.py).  At that point Mars sits about
# 3.3 million km above the ecliptic, so the pilot must steer in 3D.
MARS_U0 = np.radians(1.4)

# Extra speed from the launch rocket ("v-infinity"), added along Earth's
# direction of motion.  A Hohmann transfer would need ~2.9 km/s here, so the
# electric thruster has to supply the rest of the energy plus the 3D steering.
LAUNCH_VINF = 2.0              # [km/s]

# Capture radius: Mars' sphere of influence (same as the original project)
CAPTURE_RADIUS_KM = 577_000.0


def mars_state(t):
    """
    Position and velocity of Mars at time t (seconds since launch).

    Position on a tilted circle of radius R, angle u = u0 + w t:
        x = R (cos N cos u - sin N sin u cos i)
        y = R (sin N cos u + cos N sin u cos i)
        z = R (sin u sin i)
    Velocity is just the derivative d/dt of each line (chain rule: du/dt = w).

    t can be a number or a numpy array.  Returns (pos, vel) with shape (..., 3).
    """
    t = np.asarray(t, dtype=float)
    u = MARS_U0 + MARS_OMEGA * t
    cN, sN = np.cos(MARS_NODE), np.sin(MARS_NODE)
    ci, si = np.cos(MARS_INCLINATION), np.sin(MARS_INCLINATION)
    cu, su = np.cos(u), np.sin(u)

    pos = MARS_R * np.stack([cN * cu - sN * su * ci,
                             sN * cu + cN * su * ci,
                             su * si], axis=-1)
    vel = MARS_R * MARS_OMEGA * np.stack([-cN * su - sN * cu * ci,
                                          -sN * su + cN * cu * ci,
                                          cu * si], axis=-1)
    return pos, vel


def earth_launch_state(n=1):
    """Spacecraft starts at Earth's position, moving with Earth's velocity plus the launch boost."""
    pos = np.tile([EARTH_R, 0.0, 0.0], (n, 1))
    vel = np.tile([0.0, EARTH_SPEED + LAUNCH_VINF, 0.0], (n, 1))
    return pos, vel


def norm(v):
    """Length of each 3D vector: sqrt(x^2 + y^2 + z^2).  Keeps the last axis."""
    return np.sqrt(np.sum(v * v, axis=-1, keepdims=True))


def gravity_accel(r):
    """
    Sun's gravity on the spacecraft (km/s^2).

    Size:      a = GM / |r|^2          (inverse-square law)
    Direction: toward the Sun = -r / |r|
    Together:  a = -GM * r / |r|^3
    """
    rmag = norm(r)
    return -MU_SUN * r / rmag**3


def local_frame(r, v):
    """
    Three perpendicular unit vectors attached to the spacecraft:

        r_hat : points away from the Sun          ("radial")
        n_hat : perpendicular to the orbit plane  ("normal")  = (r x v) / |r x v|
        t_hat : along the direction of travel     ("along-track") = n_hat x r_hat

    Thrusting along t_hat speeds you up (raises your orbit), along r_hat
    pushes you outward, along n_hat tilts your orbit up/down.
    """
    r_hat = r / norm(r)
    h = np.cross(r, v)
    n_hat = h / norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return r_hat, t_hat, n_hat


def thrust_direction(alpha, beta, r_hat, t_hat, n_hat):
    """
    Turn two steering angles into a 3D unit vector (like latitude/longitude):

        alpha: in-plane angle, measured from along-track toward radial
        beta : out-of-plane angle, measured up toward the orbit normal

        dir = cos(beta) * (cos(alpha) t_hat + sin(alpha) r_hat) + sin(beta) n_hat
    """
    a = alpha[:, None]
    b = beta[:, None]
    return np.cos(b) * (np.cos(a) * t_hat + np.sin(a) * r_hat) + np.sin(b) * n_hat


def step(r, v, m, thrust_N, direction, mdot_kg_s, dt, dry_mass):
    """
    Advance every spacecraft by one time step dt (seconds).

    thrust_N   : actual push the thruster delivers (N), shape (N,)
    direction  : unit vectors, shape (N, 3)
    mdot_kg_s  : propellant flow rate (kg/s), shape (N,)

    Returns new (r, v, m).  Spacecraft that are out of fuel get no thrust.
    """
    has_fuel = (m > dry_mass)
    thrust_N = np.where(has_fuel, thrust_N, 0.0)
    mdot_kg_s = np.where(has_fuel, mdot_kg_s, 0.0)

    # F = m a  ->  a = F / m   (N / kg = m/s^2, divide by 1000 for km/s^2)
    a_thrust = (thrust_N / m / 1000.0)[:, None] * direction
    a_total = gravity_accel(r) + a_thrust

    v_new = v + a_total * dt            # tangent-line step for velocity
    r_new = r + v_new * dt              # tangent-line step for position
    m_new = np.maximum(m - mdot_kg_s * dt, dry_mass)
    return r_new, v_new, m_new


def specific_energy(r, v):
    """Orbital energy per kg: E = v^2/2 - GM/r.  Constant when the engine is off."""
    return 0.5 * np.sum(v * v, axis=-1) - MU_SUN / norm(r)[..., 0]
