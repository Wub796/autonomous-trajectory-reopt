"""
pilot.py: the AI pilot, a tiny neural network written in plain numpy.

A neural network is a function built from simple pieces:

    hidden = tanh(W1 . inputs + b1)        (11 numbers in -> 16 numbers)
    output = W2 . hidden + b2              (16 numbers in ->  3 numbers)

W1, W2 are grids of "weights" and b1, b2 are "biases".  All 243 of these
numbers together are called theta.  Training means finding a theta that
makes the spacecraft fly well (see trainer.py).

The 3 outputs are squashed into sensible ranges:
    throttle = sigmoid(out0)            in (0, 1)       how hard to push
    alpha    = (pi/2) tanh(out1)        in (-90, 90) deg  steer in-plane
    beta     = (pi/2) tanh(out2)        in (-90, 90) deg  steer up / down

What the pilot "sees" (11 inputs, all scaled to be roughly between -1 and 1):
    0-2  where Mars is relative to me     (radial, along-track, normal parts) / 1 AU
    3-5  Mars' velocity relative to mine  (radial, along-track, normal parts) / 10 km/s
    6    my distance from the Sun  (in AU, minus 1.25)
    7    my outward speed / 5 km/s
    8    fuel left  (fraction of a full tank)
    9    time left  (fraction of the mission)
    10   how healthy I *believe* my thruster is, (fraction - 1) * 5
"""
import numpy as np

from mission import physics as P
from mission.thruster import WET_MASS, DRY_MASS, MISSION_HOURS

N_IN, N_HID, N_OUT = 11, 16, 3
N_PARAMS = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT     # = 243


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init_params(rng, scale=0.5):
    """Random hidden layer, zero output layer -> starts as 'half throttle, straight ahead'."""
    W1 = rng.standard_normal((N_IN, N_HID)) * scale / np.sqrt(N_IN)
    b1 = np.zeros(N_HID)
    W2 = np.zeros((N_HID, N_OUT))
    b2 = np.zeros(N_OUT)
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])


def unpack(theta):
    """theta shape (N, 243) -> W1 (N,11,16), b1 (N,16), W2 (N,16,3), b2 (N,3)."""
    i = 0
    W1 = theta[:, i:i + N_IN * N_HID].reshape(-1, N_IN, N_HID); i += N_IN * N_HID
    b1 = theta[:, i:i + N_HID]; i += N_HID
    W2 = theta[:, i:i + N_HID * N_OUT].reshape(-1, N_HID, N_OUT); i += N_HID * N_OUT
    b2 = theta[:, i:i + N_OUT]
    return W1, b1, W2, b2


def observe(r, v, m, t_sec, believed_health):
    """Build the 11 inputs for every spacecraft.  r, v: (N,3); m, believed_health: (N,)."""
    r_hat, t_hat, n_hat = P.local_frame(r, v)
    mars_r, mars_v = P.mars_state(t_sec)
    d = mars_r[None, :] - r
    dv = mars_v[None, :] - v

    def parts(vec):   # dot products with the 3 local unit vectors
        return np.stack([np.sum(vec * r_hat, -1), np.sum(vec * t_hat, -1), np.sum(vec * n_hat, -1)], -1)

    rmag = P.norm(r)[:, 0]
    obs = np.concatenate([
        parts(d) / P.AU * 2.0,
        parts(dv) / 10.0,
        (rmag / P.AU - 1.25)[:, None] * 2.0,
        (np.sum(v * r_hat, -1) / 5.0)[:, None],
        ((m - DRY_MASS) / (WET_MASS - DRY_MASS))[:, None],
        np.full((r.shape[0], 1), 1.0 - t_sec / (MISSION_HOURS * P.HOUR)),
        ((believed_health - 1.0) * 5.0)[:, None],
    ], axis=1)
    return obs, (r_hat, t_hat, n_hat)


def act(theta, obs):
    """
    Run the network.  theta: (N, 243) (one set of weights per spacecraft), obs: (N, 11).
    Returns throttle, alpha, beta, each shape (N,).
    """
    W1, b1, W2, b2 = unpack(theta)
    hidden = np.tanh(np.einsum("ni,nih->nh", obs, W1) + b1)
    out = np.einsum("nh,nho->no", hidden, W2) + b2
    throttle = sigmoid(out[:, 0])
    alpha = 0.5 * np.pi * np.tanh(out[:, 1])
    beta = 0.5 * np.pi * np.tanh(out[:, 2])
    return throttle, alpha, beta


def act_single(theta, obs):
    """Same network for ONE spacecraft using plain matrix math (used for the timing test)."""
    n = N_IN * N_HID
    W1 = theta[:n].reshape(N_IN, N_HID)
    b1 = theta[n:n + N_HID]
    W2 = theta[n + N_HID:n + N_HID + N_HID * N_OUT].reshape(N_HID, N_OUT)
    b2 = theta[-N_OUT:]
    out = np.tanh(obs @ W1 + b1) @ W2 + b2
    return sigmoid(out[0]), 0.5 * np.pi * np.tanh(out[1]), 0.5 * np.pi * np.tanh(out[2])
