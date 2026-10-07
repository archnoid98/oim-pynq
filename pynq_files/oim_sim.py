#!/usr/bin/env python3
# =============================================================================
#  oim_sim.py -- Frequency-tunable CMOS ring-oscillator Ising machine:
#                system simulation + waveform emulation + readout study.
#
#  Digital twin of:  Md. Mizanur Rahaman Nayan and Orchi Hassan,
#  "Frequency Tunable CMOS Ring Oscillator Based Ising Machine".
#
#  This one file covers the whole assignment:
#
#    STAGE 1  SYSTEM SIMULATION  (Ks, Kc)          -> phases phi_i(t)
#    STAGE 2  WAVEFORM EMULATION                   -> V_i(t) = VDD/2*sgn(sin(w_i t + phi_i)) + VDD/2
#    STAGE 3  READOUT            (XOR / DFF / RSFF / edge count)
#    STAGE 4  THE UNREADABLE BAND  (sweep dphi slightly and slightly)
#    STAGE 5  DEGENERACY         (2f frequency multiplication, frequency sweeping)
#    STAGE 6  FIXED-POINT TWIN   (bit-exact model of oim_pynq.v)
#
#  Run:
#     python3 oim_sim.py --all            every stage, every figure, all checks
#     python3 oim_sim.py --checks         self-test only (fast, no plots)
#     python3 oim_sim.py --stage 4        one stage
#     python3 oim_sim.py --all --save     also write figN_*.png
#
#  Conventions fixed once, used everywhere
#  ---------------------------------------
#    A        adjacency matrix, A[i,j] = 1 if edge (i,j).       SYMMETRIC, ZERO DIAGONAL.
#    J        Ising coupling, J = -A.  (ANTIFERROMAGNETIC: this is what makes
#             minimising the Hamiltonian identical to maximising the cut.)
#    H(s)     = -sum_{i != j} J_ij s_i s_j        (paper Eq. 1, ORDERED sum)
#    E(phi)   = -Kc sum_{i != j} J_ij cos(phi_i - phi_j) - Ks sum_i cos(2 phi_i)
#                                                 (paper Eq. 4, ORDERED sum)
#    phase    radians, unwrapped internally, reported mod 360 deg relative to a base.
#    spin     phi = 0 deg -> +1,  phi = 180 deg -> -1,  anything else -> 0 (FREE).
# =============================================================================

from __future__ import annotations

import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from itertools import product

import numpy as np

TWO_PI = 2.0 * np.pi

KS_DEFAULT = 0.30
#  Ks/Kc = 0.30 is the CENTRE of the measured useful window, not a guess.
#  Sweeping Ks over 400 runs per point on the size-6 graph (Kc = 1.0):
#
#      Ks/Kc   0.10  0.15  0.20  0.25  0.30  0.40  0.50  0.60  0.80  1.00  1.20
#      fixed   1.00  1.00  1.00  1.00  1.00  1.00  0.80  0.60  0.55  0.28  0.24
#      ramped  1.00  1.00  1.00  1.00  1.00  1.00  0.98  0.96  0.93  0.90  0.89
#
#  P(hit optimum) is 1.000 across 0.10-0.40 and falls away above it, which is
#  exactly the window the BUET slide quotes. An earlier version of this file
#  used Ks = 0.6 throughout -- outside the window, and measurably worse
#  (P(opt) 0.60 fixed / 0.96 ramped instead of 1.00).
#
#  LOWER bound too: an oscillator locks only while |dw_i| < Ks, so any
#  frequency-tuning sweep must keep its detuning under KS_DEFAULT or every
#  oscillator unlocks at once. That is why the tuning range is +-0.25 here.
#
#  FINE RE-MEASUREMENT (anneal WITH noise T = 60, relax with D = 0 T = 40,
#  Ks/Kc stepped by 0.025). The window is wider than 0.10 - 0.40 on BOTH sides:
#
#      Ks/Kc          0.000  0.025  0.050 .. 0.450  0.475  0.500  0.600
#      board  P(opt)  0.925  1.000  1.000 .. 1.000  0.775  0.650  0.425
#             binar.  0.417  0.921  1.000 .. 1.000  0.850  0.767  0.813
#      float  P(opt)  0.96   --     1.00  .. 0.99   --     0.80   0.54
#             binar.  0.52   --     1.00  .. 1.00   --     0.87   0.80
#
#  "board" is the fixed-point RTL model that reproduced a PYNQ-Z2 run point
#  for point (40 runs/point); "float" is ks_ratio_sweep (200 runs/point).
#  Useful = P(opt) >= 0.95 AND binarised >= 0.95. On the size-6 graph that is
#  0.05 - 0.45; KS_DEFAULT = 0.30 stays inside it and above KS_TUNE_MAX.
#
#  THE WINDOW IS GRAPH-DEPENDENT. Same criterion, float model, 200 runs:
#      size-6 0.05-0.45   K4 0.05-0.80   size-5 0.45-0.80
#      random-10 0.60-0.70   surrogate-16 0.50-0.80   King 3x4: none
#  On frustrated graphs a weak Ks has genuinely NON-binary fixed points (a
#  ten times longer relax changes nothing): the cut still rounds correctly
#  but the phases do not sit at 0/180, which is what the readout needs.
#  KS_WINDOW is therefore the size-6 window, not a universal constant.
KS_TUNE_MAX = 0.25
KS_WINDOW = (0.05, 0.45)

# =============================================================================
#  STAGE 0 -- GRAPHS AND THE ISING MAPPING
# =============================================================================


def _sym(mat):
    A = np.array(mat, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("adjacency must be symmetric")
    if not np.allclose(np.diag(A), 0.0):
        raise ValueError("adjacency must have zero diagonal (no self loops)")
    return A


def size6_graph():
    """Paper Fig. 1(a)/3(a): n=6, 9 edges, ONE optimal bipartition (no contention)."""
    return _sym([[0, 1, 0, 1, 0, 1],
                 [1, 0, 1, 1, 0, 1],
                 [0, 1, 0, 0, 0, 1],
                 [1, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0]])


def size5_graph():
    """Paper Fig. 3(e)/4: n=5, 5 edges. THE CONTENTION CASE madam drew.

    Optimal bipartitions include {{1,2,5},{3,4}} and {{1,5},{2,3,4}} -- they
    differ only in where vertex 2 goes, which is exactly why osc2 shows up as
    the frequency-multiplied / free oscillator in Fig. 4.
    """
    return _sym([[0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1],
                 [1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 0]])


def k4_graph():
    """Paper Fig. 6: K4, all-to-all n=4. Cut 4, THREE degenerate optima."""
    return _sym(np.ones((4, 4)) - np.eye(4))


_SURROGATE16_EDGES = [
    (1, 2), (1, 7), (1, 10), (1, 12), (1, 14), (1, 15), (2, 6), (2, 7), (2, 9),
    (2, 12), (2, 13), (2, 16), (3, 5), (3, 7), (3, 8), (3, 10), (3, 12),
    (3, 13), (3, 16), (4, 6), (4, 7), (4, 10), (4, 12), (4, 14), (4, 15),
    (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 16), (6, 9), (6, 10),
    (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (7, 9), (7, 10), (7, 15),
    (7, 16), (8, 10), (8, 13), (8, 14), (9, 10), (9, 12), (9, 13), (9, 14),
    (9, 15), (9, 16), (10, 14), (10, 15), (10, 16), (11, 13), (11, 15),
    (11, 16), (12, 14), (12, 15), (12, 16), (13, 14), (13, 15), (13, 16),
    (14, 15), (14, 16)]


def surrogate16_graph():
    """n=16, m=65 stand-in for the paper's Fig. 5(a) validation graph."""
    A = np.zeros((16, 16))
    for i, j in _SURROGATE16_EDGES:
        A[i - 1, j - 1] = A[j - 1, i - 1] = 1.0
    return _sym(A)


def random_graph(n, m, seed=0):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    A = np.zeros((n, n))
    for k in rng.choice(len(pairs), size=m, replace=False):
        i, j = pairs[k]
        A[i, j] = A[j, i] = 1.0
    return A


def adjacency_to_J(A):
    """J = -A.  See the header note on why the sign matters."""
    return -np.asarray(A, dtype=float)


# ----- Ising energy / cut -----------------------------------------------------

def H_pair(J, s):
    """H over UNORDERED pairs: -sum_{i<j} J_ij s_i s_j."""
    J = np.asarray(J, float)
    s = np.asarray(s, float)
    return -0.5 * (s @ J @ s)


def H_ordered(J, s):
    """Paper Eq. (1), ORDERED sum -> exactly 2 * H_pair."""
    return 2.0 * H_pair(J, s)


def cut_value(A, s):
    A = np.asarray(A, float)
    s = np.asarray(s, float)
    return 0.25 * np.sum(A * (1.0 - np.outer(s, s)))


def eq2_cut_from_H(A, s):
    """Paper Eq. (2) right-hand side. MUST equal cut_value() for every s."""
    total_w = 0.5 * np.sum(A)
    return 0.5 * total_w - 0.5 * H_pair(adjacency_to_J(A), s)


def enumerate_states(n, fix_first=True):
    rest = list(product([1, -1], repeat=n - 1))
    if fix_first:
        return np.array([[1] + list(r) for r in rest], dtype=float)
    return np.array([[a] + list(r) for a in (1, -1) for r in rest], dtype=float)


def brute_force_maxcut(A):
    """-> (best_cut, optimal_spin_vectors, degeneracy).  Global flip quotiented out."""
    S = enumerate_states(A.shape[0])
    cuts = np.array([cut_value(A, s) for s in S])
    best = cuts.max()
    opt = S[np.isclose(cuts, best)]
    return best, opt, len(opt)


def energy_landscape(A, ordered=True):
    S = enumerate_states(A.shape[0])
    J = adjacency_to_J(A)
    f = H_ordered if ordered else H_pair
    return S, np.array([f(J, s) for s in S])


def partition_of(s):
    s = np.asarray(s)
    return ([int(v) for v in np.flatnonzero(s > 0) + 1],
            [int(v) for v in np.flatnonzero(s < 0) + 1])


def spins_from_phase(phi, ref=0.0):
    """Ideal readout used ONLY as ground truth. The real one is STAGE 3."""
    return np.sign(np.cos(np.asarray(phi) - ref) + 1e-15)


def describe_graph(A, name=""):
    best, opt, ndeg = brute_force_maxcut(A)
    print(f"  graph {name}: n={A.shape[0]}  edges={int(A.sum()/2)}  "
          f"max degree={int(A.sum(1).max())}")
    print(f"    MaxCut = {best:.0f}   degeneracy = {ndeg}  "
          f"({'no contention' if ndeg == 1 else 'CONTENTION'})")
    for s in opt[:4]:
        a, b = partition_of(s)
        print(f"      {a} | {b}")
    if ndeg > 4:
        print(f"      ... and {ndeg - 4} more")
    return best, ndeg


# =============================================================================
#  STAGE 1 -- SYSTEM SIMULATION:  the phase dynamics driven by Kc and Ks
#
#    dphi_i/dt = dw_i - w_i [ Kc sum_j J_ij sin(phi_i - phi_j)
#                             + Ks(t) sin(2 phi_i) ]  + sqrt(2 D) xi(t)
#
#  which is minus the gradient of
#    E_lyap(phi) = -Kc/1 sum_{i<j}... (see E_paper below; E_lyap = E_paper / 2)
#  so E can only fall when D = 0 and dt is inside the stability limit.
# =============================================================================


@dataclass
class OIMParams:
    Kc: float = 1.0                 # coupling strength  (paper's K)
    Ks: float = 0.0                 # SHIL / 2f injection strength
    D: float = 0.0                  # noise strength
    dt: float = 2e-3
    T: float = 60.0
    omega: np.ndarray | None = None  # per-spin gain w_i  (default all 1)
    dw: np.ndarray | None = None     # per-spin detuning  (frequency tuning!)
    Ks_schedule: object = None       # callable t -> Ks, overrides Ks
    seed: int = 0

    def ks_at(self, t):
        return self.Ks if self.Ks_schedule is None else self.Ks_schedule(t)


def linear_ramp(Ks_final, t_on, t_full):
    """Ks(t): 0 until t_on, linear up to Ks_final at t_full, flat after."""
    def f(t):
        if t <= t_on:
            return 0.0
        if t >= t_full:
            return Ks_final
        return Ks_final * (t - t_on) / (t_full - t_on)
    return f


def lipschitz_L(A, Ks, Kc=1.0):
    """L = Kc*deg_max + 2*Ks -- the max curvature of the drift."""
    return Kc * float(np.asarray(A).sum(1).max()) + 2.0 * Ks


def stable_dt(A, Ks, Kc=1.0, margin=10.0):
    """dt < 2/L for forward Euler. margin=10 is the 10x safety used throughout.

    Above this the run does not blow up loudly -- it silently produces a
    trajectory whose energy is NOT monotone. Always check energy_monotone().
    """
    return (2.0 / lipschitz_L(A, Ks, Kc)) / margin


def _w(p, n):
    return np.ones(n) if p.omega is None else np.asarray(p.omega, float)


def _dw(p, n):
    return np.zeros(n) if p.dw is None else np.asarray(p.dw, float)


def E_paper(phi, J, Kc, Ks):
    """Paper Eq. (4): ORDERED double sum, no 1/2 on the SHIL term."""
    phi = np.asarray(phi, float)
    d = phi[:, None] - phi[None, :]
    return -Kc * np.sum(J * np.cos(d)) - Ks * np.sum(np.cos(2 * phi))


def E_lyap(phi, J, Kc, Ks):
    """The potential whose plain gradient flow is the paper's Eq. (3)."""
    return 0.5 * E_paper(phi, J, Kc, Ks)


def grad_E_lyap(phi, J, Kc, Ks):
    phi = np.asarray(phi, float)
    d = phi[:, None] - phi[None, :]
    return Kc * np.sum(J * np.sin(d), axis=1) + Ks * np.sin(2 * phi)


def drift(phi, J, Kc, Ks, w, dw=0.0):
    return dw - w * grad_E_lyap(phi, J, Kc, Ks)


def hessian_E_lyap(phi, J, Kc, Ks):
    """Curvature at a fixed point. lam_min = slowest decay rate per unit w."""
    phi = np.asarray(phi, float)
    J = np.asarray(J, float)
    d = phi[:, None] - phi[None, :]
    Hm = -Kc * J * np.cos(d)
    np.fill_diagonal(Hm, Kc * np.sum(J * np.cos(d), axis=1)
                     + 2.0 * Ks * np.cos(2 * phi))
    return Hm


@dataclass
class SimResult:
    t: np.ndarray
    phi: np.ndarray          # (steps, n) unwrapped radians
    E: np.ndarray            # (steps,)  E_paper along the trajectory
    params: OIMParams = None

    @property
    def phi_final(self):
        return self.phi[-1]

    def phi_rel(self, ref=0, deg=True):
        d = np.mod(self.phi - self.phi[:, [ref]], TWO_PI)
        return np.degrees(d) if deg else d

    def settled(self, tol=1e-3, frac=0.1):
        """madam's max|phi - phi| criterion, applied over the last 10% of the run."""
        k = max(2, int(len(self.t) * frac))
        return bool(np.abs(np.diff(self.phi[-k:], axis=0)).max() < tol)

    def energy_monotone(self, tol=1e-9):
        """True for any D=0 run AT FIXED Ks. If False there, dt is too big.

        NOT expected to hold while Ks is being ramped: E is evaluated with the
        current Ks, so a rising Ks lifts the potential under the trajectory.
        Measured on a ramped run, every rising step falls inside the ramp
        window. Only test this on fixed-Ks runs.
        """
        return bool(np.diff(self.E).max() < tol)


def simulate(J, p: OIMParams, phi0=None, record_every=1):
    """Euler-Maruyama.  phi <- phi + dt*drift + sqrt(2*D*dt)*N(0,1)."""
    J = np.asarray(J, float)
    n = J.shape[0]
    w, dw = _w(p, n), _dw(p, n)
    rng = np.random.default_rng(p.seed)
    phi = (rng.uniform(0, TWO_PI, n) if phi0 is None
           else np.array(phi0, dtype=float))
    nsteps = int(round(p.T / p.dt))
    sqrt2Ddt = np.sqrt(2.0 * p.D * p.dt)
    out_t, out_phi, out_E = [], [], []
    for k in range(nsteps + 1):
        Ks = p.ks_at(k * p.dt)
        if k % record_every == 0:
            out_t.append(k * p.dt)
            out_phi.append(phi.copy())
            out_E.append(E_paper(phi, J, p.Kc, Ks))
        if k == nsteps:
            break
        phi = phi + p.dt * drift(phi, J, p.Kc, Ks, w, dw)
        if p.D > 0:
            phi = phi + sqrt2Ddt * rng.standard_normal(n)
    return SimResult(np.array(out_t), np.array(out_phi), np.array(out_E), p)


def _drift_batch(phi, J, Kc, Ks, w, dw=0.0):
    d = phi[:, :, None] - phi[:, None, :]
    g = Kc * np.einsum('ij,bij->bi', J, np.sin(d)) + Ks * np.sin(2 * phi)
    return dw - w * g


def simulate_batch(J, p: OIMParams, B=100, phi0=None):
    """B independent runs, vectorised. Use for EVERY statistical claim."""
    J = np.asarray(J, float)
    n = J.shape[0]
    w, dw = _w(p, n), _dw(p, n)
    rng = np.random.default_rng(p.seed)
    phi = (rng.uniform(0, TWO_PI, (B, n)) if phi0 is None
           else np.broadcast_to(np.asarray(phi0, float), (B, n)).copy())
    sqrt2Ddt = np.sqrt(2.0 * p.D * p.dt)
    for k in range(int(round(p.T / p.dt))):
        phi = phi + p.dt * _drift_batch(phi, J, p.Kc, p.ks_at(k * p.dt), w, dw)
        if p.D > 0:
            phi = phi + sqrt2Ddt * rng.standard_normal((B, n))
    return phi


def _start(seed, n):
    return np.random.default_rng(seed).uniform(0, TWO_PI, n)


def settle_time(J, p: OIMParams, phi0=None, tol=1e-3):
    """madam's max|phi_i(t) - phi_i(inf)| dropping below tol.  D = 0 only."""
    r = simulate(J, p, phi0=phi0)
    dev = np.abs(r.phi - r.phi[-1]).max(axis=1)
    k = np.flatnonzero(dev > tol)
    if not k.size:
        return 0.0, r
    return (np.nan if k[-1] + 1 >= len(r.t) else float(r.t[k[-1] + 1])), r


def settling_constant(A, Ks=KS_DEFAULT, tol=1e-3, seeds=24, T=120.0, Kc=1.0):
    """w * t_settle at w = 1, counting only runs that reach a binarised optimum."""
    A = np.asarray(A, float)
    J = adjacency_to_J(A)
    n = A.shape[0]
    best = brute_force_maxcut(A)[0]
    dt = stable_dt(A, Ks, Kc)
    kept, kept_seeds, lam_ok, lam_bad = [], [], [], []
    for seed in range(seeds):
        t_s, r = settle_time(J, OIMParams(Kc=Kc, Ks=Ks, D=0.0, dt=dt, T=T),
                             phi0=_start(seed, n), tol=tol)
        rel = np.degrees(np.mod(r.phi_final - r.phi_final[0], TWO_PI))
        binary = np.minimum(rel % 180, 180 - (rel % 180)).max() < 1.0
        lam = np.linalg.eigvalsh(hessian_E_lyap(r.phi_final, J, Kc, Ks)).min()
        if binary and cut_value(A, spins_from_phase(r.phi_final)) >= best \
                and np.isfinite(t_s):
            kept.append(t_s)
            kept_seeds.append(seed)
            lam_ok.append(lam)
        else:
            lam_bad.append(lam)
    kept = np.array(kept)
    return dict(t_settle=kept, seeds=kept_seeds, n_tried=seeds,
                median=float(np.median(kept)), mean=float(kept.mean()),
                lam_optimum=float(np.median(lam_ok)) if lam_ok else np.nan,
                lam_other=float(np.median(lam_bad)) if lam_bad else np.nan)


def settling_vs_omega(A, Ks=KS_DEFAULT, tol=1e-3, ws=None, seed=0, Kc=1.0,
                      T_over_w=60.0):
    """Sweep the gain w over a decade. dt and T scale as 1/w, Ks held FIXED
    (a ramp is absolute-time and is the one thing that would not rescale)."""
    A = np.asarray(A, float)
    J = adjacency_to_J(A)
    n = A.shape[0]
    ws = np.logspace(-0.5, 1.0, 7) if ws is None else np.asarray(ws, float)
    dt0 = stable_dt(A, Ks, Kc)
    phi0 = _start(seed, n)
    t_set, curves = [], []
    for w in ws:
        p = OIMParams(Kc=Kc, Ks=Ks, D=0.0, dt=dt0 / w, T=T_over_w / w,
                      omega=np.full(n, w))
        t_s, r = settle_time(J, p, phi0=phi0, tol=tol)
        t_set.append(t_s)
        curves.append((r.t, np.abs(r.phi - r.phi[-1]).max(axis=1)))
    t_set = np.array(t_set)
    return dict(w=ws, t_settle=t_set, product=ws * t_set, curves=curves, tol=tol)


#  Measured Ks/Kc windows, criterion P(opt) >= 0.95 AND binarised >= 0.95.
#  The float column is ks_ratio_sweep at 200 runs/point; the board column was
#  measured on the PYNQ-Z2 at 40 runs/point.
#
#      graph          float        board        KS_DEFAULT = 0.30 inside?
#      size-6      0.05 - 0.45   0.05 - 0.45           yes
#      K4          0.05 - 0.80   0.05 - 0.80           yes
#      size-5      0.45 - 0.80   0.55 - 0.80           NO
#      random-10   0.60 - 0.70   0.60 - 0.80           NO
#      surrogate16 0.50 - 0.80   0.55 - 0.80           NO
#
#  KS_DEFAULT is a SIZE-6 value. On three of these five graphs it is below the
#  window: the cut still rounds out correctly, but the phases do not sit at
#  0/180 and the readout is left with unresolved spins. Frustrated graphs have
#  genuinely non-binary fixed points at weak Ks -- relaxing ten times longer
#  changes nothing.
KS_KNOWN_WINDOWS = {
    "size6": (0.05, 0.45), "k4": (0.05, 0.80), "size5": (0.55, 0.80),
    "surrogate16": (0.55, 0.80),
}


def _graph_key(A):
    """Cheap fingerprint so the table above can be looked up without a sweep."""
    A = np.asarray(A, float)
    for nm, G in (("size6", size6_graph()), ("size5", size5_graph()),
                  ("k4", k4_graph()), ("surrogate16", surrogate16_graph())):
        if A.shape == G.shape and np.array_equal(A, G):
            return nm
    return None


def ks_window_for(A, ratios=None, B=60, thresh=0.95, **kw):
    """MEASURE the useful Ks/Kc window for one graph.

    Useful = P(hit optimum) >= thresh AND binarised fraction >= thresh. The
    binarisation half is the one that bites: the cut can round out correctly
    while the phases sit nowhere near 0/180, and it is the phases the readout
    has to work with.
    """
    ratios = (np.round(np.arange(0.0, 0.851, 0.05), 3) if ratios is None
              else np.asarray(ratios, float))
    r = ks_ratio_sweep(A, ratios=ratios, B=B, **kw)
    ok = (r["p_opt"] >= thresh) & (r["binarised"] >= thresh)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    return (float(ratios[idx[0]]), float(ratios[idx[-1]]))


def recommend_ks(A, default=None, B=60, quiet=False):
    """The Ks/Kc to use for THIS graph: the centre of its measured window.

    Looks the graph up in KS_KNOWN_WINDOWS first (free); otherwise sweeps.
    Falls back to KS_DEFAULT with a warning if no window exists at all.
    """
    key = _graph_key(A)
    win = KS_KNOWN_WINDOWS.get(key) if key else None
    if win is None:
        win = ks_window_for(A, B=B)
    if win is None:
        if not quiet:
            print(f"   [recommend_ks: no Ks gives P>=0.95 AND binarised>=0.95 "
                  f"on this graph; falling back to {KS_DEFAULT}]")
        return KS_DEFAULT if default is None else default
    ks = round(0.5 * (win[0] + win[1]), 3)
    if not quiet:
        print(f"   [recommend_ks: window {win[0]:.2f}-{win[1]:.2f}"
              f"{' (' + key + ')' if key else ' (measured)'} -> Ks/Kc = {ks}]")
    return ks


def check_ks(A, Ks):
    """-> (ok, window) for a graph whose window is already known. None if not."""
    key = _graph_key(A)
    win = KS_KNOWN_WINDOWS.get(key) if key else None
    if win is None:
        return None, None
    return (win[0] <= Ks <= win[1]), win


def ks_ratio_sweep(A, ratios=None, B=200, T_anneal=60.0, T_relax=40.0,
                   D=0.004, tol_deg=5.0, Kc=1.0, seed=1):
    """Binarised fraction and P(hit optimum) vs Ks/Kc.

    Each point anneals WITH noise at fixed Ks, then relaxes with D = 0 so the
    phases are sitting on true fixed points when they are measured.
    """
    A = np.asarray(A, float)
    J = adjacency_to_J(A)
    best = brute_force_maxcut(A)[0]
    ratios = (np.linspace(0.0, 1.6, 17) if ratios is None
              else np.asarray(ratios, float))
    binar, popt = [], []
    for Ks in ratios:
        dt = stable_dt(A, max(Ks, 0.1), Kc)
        PH = simulate_batch(J, OIMParams(Kc=Kc, Ks=Ks, D=D, dt=dt, T=T_anneal,
                                         seed=seed), B=B)
        PH = simulate_batch(J, OIMParams(Kc=Kc, Ks=Ks, D=0.0, dt=dt, T=T_relax),
                            B=B, phi0=PH)
        rel = np.degrees(np.mod(PH - PH[:, [0]], TWO_PI))
        off = np.minimum(rel % 180, 180 - (rel % 180))
        cuts = np.array([cut_value(A, spins_from_phase(x)) for x in PH])
        binar.append(float(np.mean(off < tol_deg)))
        popt.append(float(np.mean(cuts >= best)))
    return dict(ratio=ratios, binarised=np.array(binar), p_opt=np.array(popt),
                tol_deg=tol_deg, B=B)


# =============================================================================
#  STAGE 2 -- WAVEFORM EMULATION
#
#  madam's formula, written on the page:
#
#      V(t) = (VDD/2) * sgn( sin( w t + phi ) ) + VDD/2
#
#  sgn(sin x) = +1 exactly when frac(x / 2pi) < 1/2, so the generalisation to a
#  non-ideal duty cycle d is  V = VDD  iff  frac(f t + phi/2pi) < d.
#  d = 0.5 recovers the formula above EXACTLY (asserted in the self-tests).
#
#  f_i is per oscillator: f_i = 2 is the "frequency multiplication" state that
#  the paper reports for osc2 and that madam circled on the 5-oscillator plot.
# =============================================================================

VDD = 0.8   # volts, matches the paper's 14 nm FinFET supply


def square_wave(t, f=1.0, phase=0.0, duty=0.5, vdd=VDD):
    """One oscillator's digital output. t is in periods of the base frequency."""
    x = np.mod(f * np.asarray(t, float) + phase / TWO_PI, 1.0)
    return np.where(x < duty, vdd, 0.0)


def madam_formula(t, w=TWO_PI, phase=0.0, vdd=VDD):
    """Literal transcription of the handwritten formula, for the identity test."""
    return (vdd / 2.0) * np.sign(np.sin(w * np.asarray(t, float) + phase)) + vdd / 2.0


def time_grid(n_periods=200, pts_per_period=256):
    return np.linspace(0, n_periods, n_periods * pts_per_period, endpoint=False)


def make_waveforms(phi, f=None, t=None, n_periods=200, pts_per_period=256,
                   duty=0.5, vdd=VDD):
    """Settled phases -> the digital waveforms a scope (or the FPGA) would see."""
    phi = np.asarray(phi, float)
    n = phi.size
    f = np.ones(n) if f is None else np.asarray(f, float)
    duty = np.full(n, duty) if np.isscalar(duty) else np.asarray(duty, float)
    if t is None:
        t = time_grid(n_periods, pts_per_period)
    V = np.stack([square_wave(t, f[i], phi[i], duty[i], vdd) for i in range(n)])
    return t, V


# =============================================================================
#  STAGE 3 -- READOUT.  Four detectors on the same two digital signals.
#
#    (a) XOR + average + threshold     <- the paper's Algorithm 1 (madam's bubble)
#    (b) D flip-flop sampler           <- 1-bit spin, sampled on the base edge
#    (c) RS flip-flop (set/reset) PD   <- sawtooth, 2x range, duty-INSENSITIVE
#    (d) rising-edge counter           <- frequency ratio, the only detector that
#                                          can see "frequency multiplication"
#
#  Every function takes LOGIC arrays (or volts; it thresholds at VDD/2).
# =============================================================================


def _logic(V, vdd=VDD):
    return np.asarray(V) > vdd / 2.0


def _rising(b):
    """Indices where a logic array goes 0 -> 1."""
    b = np.asarray(b, bool)
    return np.flatnonzero((~b[:-1]) & b[1:]) + 1


def _erode_short_pulses(x, min_len):
    """Detector dead zone: any XOR pulse shorter than min_len samples is lost."""
    y = np.asarray(x, bool).copy()
    i, N = 0, y.size
    while i < N:
        if y[i]:
            j = i
            while j < N and y[j]:
                j += 1
            if (j - i) < min_len:
                y[i:j] = False
            i = j
        else:
            i += 1
    return y


# ---- (a) XOR phase detector --------------------------------------------------

def pd_xor(Vb, Vi, vdd=VDD, deadzone=0.0, skew=0.0):
    """Mismatch coefficient C, in percent of the record. Paper's Algorithm 1 input.

    deadzone : fraction of the record; XOR pulses shorter than this vanish
               (models the finite rise/fall time of a real XOR gate).
    skew     : fraction of the record; static delay mismatch between the inputs.
    """
    b, i = _logic(Vb, vdd), _logic(Vi, vdd)
    x = np.logical_xor(b, i)
    if skew:
        x = np.roll(x, int(round(skew * x.size)))
    if deadzone:
        x = _erode_short_pulses(x, max(1, int(round(deadzone * x.size))))
    return 100.0 * x.mean()


def xor_closed_form(delta_deg, duty=0.5, duty_b=None):
    """C = 100 * min(2*theta, 2*d, 2*(1-theta)).  Trapezoid; d=0.5 -> triangle.

    THE key consequence: the anti-phase anchor sits at 200*d, not at 100.
    """
    d = duty if duty_b is None else min(duty, duty_b)
    th = np.mod(np.asarray(delta_deg, float) / 360.0, 1.0)
    return 100.0 * np.minimum(np.minimum(2 * th, 2 * (1 - th)), 2 * d)


# ---- (b) D flip-flop sampler -------------------------------------------------

def pd_dff(Vb, Vi, vdd=VDD, setup=1):
    """Sample V_i on every rising edge of V_base.

    Returns the bit sequence, its mean, and its TOGGLE RATE. The toggle rate is
    the discriminator the XOR average throws away: a phase-locked oscillator
    gives a constant bit stream (toggle 0), an unlocked one gives a drifting
    stream (toggle > 0).
    """
    b, i = _logic(Vb, vdd), _logic(Vi, vdd)
    idx = _rising(b) + setup
    idx = idx[idx < i.size]
    bits = i[idx].astype(int)
    if bits.size < 2:
        return dict(bits=bits, mean=np.nan, toggle=np.nan)
    return dict(bits=bits, mean=float(bits.mean()),
                toggle=float(np.mean(bits[1:] != bits[:-1])))


# ---- (c) RS (set/reset) flip-flop phase detector -----------------------------

def pd_rsff(Vb, Vi, vdd=VDD):
    """SET on a rising edge of V_base, RESET on a rising edge of V_i.

    Sawtooth characteristic: duty = dphi/360 over the FULL 0..360 range, and it
    is edge-triggered so it does not care about the input duty cycle at all.
    Price: it needs clean edges, and it mis-reads if edges are missing.
    """
    b, i = _logic(Vb, vdd), _logic(Vi, vdd)
    set_e = _rising(b)
    rst_e = _rising(i)
    if set_e.size == 0 or rst_e.size == 0:
        return np.nan
    N = b.size
    ev = np.concatenate([set_e, rst_e])
    val = np.concatenate([np.ones(set_e.size, int), np.zeros(rst_e.size, int)])
    order = np.argsort(ev, kind='stable')
    ev, val = ev[order], val[order]
    # collapse repeats: state only changes on a transition of `val`
    keep = np.concatenate([[True], val[1:] != val[:-1]])
    ev, val = ev[keep], val[keep]
    high = 0
    for k in range(ev.size - 1):
        if val[k] == 1:
            high += ev[k + 1] - ev[k]
    if val[-1] == 1:
        high += N - ev[-1]
    return 100.0 * high / (N - ev[0]) if N > ev[0] else np.nan


def rsff_closed_form(delta_deg):
    """duty = 1 - dphi/360 on (0, 360): a clean sawtooth, NO fold-back at 180.

    SIGN CONVENTION, measured not assumed: square_wave(t, f, phase=+d) puts the
    phase INSIDE sin(wt + phi), so a positive d makes V_i LEAD the base by d
    degrees (it rises at t = 1 - d/360 of a period). The RS output is therefore
    high for the fraction 1 - d/360.

    THE BLIND SPOT: at d = 0 exactly, the set and reset edges coincide and the
    output collapses to 0, i.e. the characteristic is DISCONTINUOUS by a full
    100 % precisely at the in-phase point. That is the one phase an Ising
    readout most needs to resolve, which is why this detector -- despite having
    twice the linear range of the XOR and being immune to duty distortion --
    is the wrong choice here. Verified by measurement in the self-tests.
    """
    d = np.mod(np.asarray(delta_deg, float), 360.0)
    return np.where(d == 0.0, 0.0, 100.0 * (1.0 - d / 360.0))


# ---- (d) rising-edge counter -------------------------------------------------

def pd_edge_ratio(Vb, Vi, vdd=VDD):
    """f_i / f_base from raw edge counts. The ONLY detector here that can
    separate 'frequency multiplication' from 'locked at 90 deg'."""
    nb = _rising(_logic(Vb, vdd)).size
    ni = _rising(_logic(Vi, vdd)).size
    return np.nan if nb == 0 else ni / nb


# =============================================================================
#  STAGE 4 -- THE UNREADABLE BAND
#
#  "shift the square wave slightly and slightly and get the range of
#   unreadable data."  That is exactly this section.
#
#  Algorithm 1 of the paper sorts the measured C into four classes:
#
#      C > 100 - M_th            -> spin -1      (anti-phase, 180 deg)
#      C <       M_th            -> spin +1      (in-phase, 0 deg)
#      50 - M_th < C < 50 + M_th -> spin  0      (FREE: belongs to both groups)
#      anything else             -> is_stable = 0 (UNREADABLE)
#
#  With a 50 % duty detector C = 100 * dphi/180 on the rising arm, so the class
#  boundaries land at fixed angles and the unreadable set is FOUR slivers:
#
#      (1.8 M , 90 - 1.8 M) and (90 + 1.8 M , 180 - 1.8 M), plus their mirrors.
#
#  Each sliver is (90 - 3.6 M) wide, so the unreadable set vanishes exactly at
#  M_th = 25 %. That is the headroom design rule this stage produces.
# =============================================================================

CLASS_PLUS, CLASS_MINUS, CLASS_FREE, CLASS_BAD = +1, -1, 0, 9
CLASS_NAME = {CLASS_PLUS: "+1", CLASS_MINUS: "-1", CLASS_FREE: "free",
              CLASS_BAD: "UNREADABLE"}


def classify_C(C, M_th=20.0):
    """The paper's Algorithm 1 decision, applied to one measured C."""
    C = np.asarray(C, float)
    out = np.full(C.shape, CLASS_BAD, dtype=int)
    out[(C > 100 - M_th)] = CLASS_MINUS
    out[(C < M_th)] = CLASS_PLUS
    free = (C > 50 - M_th) & (C < 50 + M_th) & (out == CLASS_BAD)
    out[free] = CLASS_FREE
    return out if out.shape else int(out)


def unreadable_bands_analytic(M_th=20.0, duty=0.5):
    """Closed-form class boundaries in degrees, from C = 100*min(2t,2d,2(1-t))."""
    peak = 200.0 * duty
    if 100.0 - M_th > peak:
        return dict(peak_C=peak, collapsed=True, plus=[], minus=[], free=[],
                    bad=[(0.0, 360.0)], bad_width=360.0,
                    note="anti-phase anchor unreachable: 200*duty < 100 - M_th")

    def ang(c):                       # C -> dphi on the rising arm
        return 1.8 * c

    m = ang(M_th)
    f_lo, f_hi = ang(50 - M_th), ang(50 + M_th)
    a = ang(100 - M_th)
    plus = [(0.0, m), (360.0 - m, 360.0)]
    minus = [(a, 360.0 - a)]
    free = [(f_lo, f_hi), (360.0 - f_hi, 360.0 - f_lo)]
    bad = []
    for lo, hi in [(m, f_lo), (f_hi, a), (360.0 - a, 360.0 - f_hi),
                   (360.0 - f_lo, 360.0 - m)]:
        if hi - lo > 1e-9:
            bad.append((lo, hi))
    return dict(peak_C=peak, collapsed=False, plus=plus, minus=minus,
                free=free, bad=bad,
                bad_width=float(sum(h - l for l, h in bad)),
                critical_M_th=25.0, note="")


def sweep_phase_shift(n_pts=721, duty=0.5, duty_b=None, deadzone=0.0,
                      n_periods=200, pts_per_period=256, M_th=20.0,
                      f_i=1.0, vdd=VDD):
    """MEASURE every detector while shifting one square wave past the other.

    This is the literal experiment on madam's page: build two square waves from
    her formula, slide one, and record what each readout circuit reports.
    """
    duty_b = duty if duty_b is None else duty_b
    deltas = np.linspace(0.0, 360.0, n_pts)
    t = time_grid(n_periods, pts_per_period)
    Vb = square_wave(t, 1.0, 0.0, duty_b, vdd)
    C = np.empty(n_pts)
    rs = np.empty(n_pts)
    dff_mean = np.empty(n_pts)
    dff_tog = np.empty(n_pts)
    ratio = np.empty(n_pts)
    for k, d in enumerate(deltas):
        Vi = square_wave(t, f_i, np.radians(d), duty, vdd)
        C[k] = pd_xor(Vb, Vi, vdd, deadzone=deadzone)
        rs[k] = pd_rsff(Vb, Vi, vdd)
        dd = pd_dff(Vb, Vi, vdd)
        dff_mean[k], dff_tog[k] = dd["mean"], dd["toggle"]
        ratio[k] = pd_edge_ratio(Vb, Vi, vdd)
    cls = classify_C(C, M_th)
    step = deltas[1] - deltas[0]
    return dict(delta=deltas, C=C, C_ideal=xor_closed_form(deltas, duty, duty_b),
                rsff=rs, rsff_ideal=rsff_closed_form(deltas),
                dff_mean=dff_mean, dff_toggle=dff_tog, edge_ratio=ratio,
                cls=cls, M_th=M_th, duty=duty, duty_b=duty_b,
                bad_width_measured=float(np.sum(cls == CLASS_BAD) * step),
                free_width_measured=float(np.sum(cls == CLASS_FREE) * step),
                analytic=unreadable_bands_analytic(M_th, min(duty, duty_b)))


def bands_from_classes(deltas, cls):
    """Contiguous [lo, hi) runs of each class -- for shading a plot."""
    out = {}
    for c in (CLASS_PLUS, CLASS_MINUS, CLASS_FREE, CLASS_BAD):
        m = (cls == c).astype(int)
        edges = np.diff(np.concatenate([[0], m, [0]]))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        out[c] = [(float(deltas[a]), float(deltas[min(b, len(deltas) - 1)]))
                  for a, b in zip(starts, stops)]
    return out


def unreadable_vs_Mth(M_list=None, duty=0.5, n_pts=721, **kw):
    """Width of the unreadable set as M_th is swept: the headroom design curve."""
    M_list = np.arange(2.0, 41.0, 2.0) if M_list is None else np.asarray(M_list, float)
    widths, frees = [], []
    for M in M_list:
        r = sweep_phase_shift(n_pts=n_pts, duty=duty, M_th=M, **kw)
        widths.append(r["bad_width_measured"])
        frees.append(r["free_width_measured"])
    return dict(M_th=M_list, bad_width=np.array(widths),
                free_width=np.array(frees), duty=duty)


# ---- Algorithm 1, multi-base, with the paper's typo repaired -----------------

def algorithm1(C_of, N, M_th=20.0, base_order=None):
    """Generalised post-processing (paper Algorithm 1), corrected.

    Two repairs against the printed listing:
      1. the printed test `C < 50 + M_th and m > 50 - M_th` uses an undefined
         `m`; it is C.  Implemented as 50 - M_th < C < 50 + M_th.
      2. the printed loop asserts S[b] = 1 for every new base, which silently
         re-labels already-resolved spins. Only the FIRST base is asserted; a
         later base contributes only if it is itself already resolved.
    """
    S = np.full(N, np.nan)
    is_stable = True
    order = list(range(N)) if base_order is None else list(base_order)
    bases_used = []
    S[order[0]] = 1.0
    for b in order:
        if not np.any(np.isnan(S) | (S == 0)):
            break
        if np.isnan(S[b]) or S[b] == 0:
            continue
        bases_used.append(b)
        for i in range(N):
            if not (np.isnan(S[i]) or S[i] == 0):
                continue
            C, ref = C_of(b, i), S[b]
            if C > 100 - M_th:
                S[i] = -ref
            elif C < M_th:
                S[i] = ref
            elif 50 - M_th < C < 50 + M_th:
                S[i] = 0.0
            else:
                is_stable = False
                S[i] = 0.0
    return dict(S=np.nan_to_num(S, nan=0.0), is_stable=is_stable,
                n_unresolved=int(np.sum(np.nan_to_num(S, nan=0.0) == 0)),
                bases_used=bases_used)


def spins_to_partition(S):
    """0 entries are FREE vertices: they may join EITHER group."""
    S = np.asarray(S)
    return ([int(v) for v in np.flatnonzero(S > 0) + 1],
            [int(v) for v in np.flatnonzero(S < 0) + 1],
            [int(v) for v in np.flatnonzero(S == 0) + 1])


def expand_free(S):
    """Every bipartition consistent with a spin vector containing free spins."""
    S = np.asarray(S, float)
    free = np.flatnonzero(S == 0)
    outs = []
    for bits in product([1.0, -1.0], repeat=len(free)):
        s = S.copy()
        s[free] = bits
        outs.append(s)
    return outs


def C_matrix_all_bases(phi, f=None, duty=0.5, deadzone=0.0, n_periods=200,
                       pts_per_period=256, vdd=VDD):
    """C[b][i] for every base b -- one waveform set, N*N comparisons."""
    t, V = make_waveforms(phi, f=f, duty=duty, n_periods=n_periods,
                          pts_per_period=pts_per_period, vdd=vdd)
    n = V.shape[0]
    return np.array([[pd_xor(V[b], V[i], vdd, deadzone=deadzone)
                      for i in range(n)] for b in range(n)])


def spin_error_vs_duty(PHI, duties, M_th=(20.0,), deadzone=0.0, n_periods=100,
                       pts_per_period=128):
    """Spin error rate vs detector duty, over settled phase vectors PHI.

    Error = the recovered spin disagrees with the ideal readout, or is left
    unresolved. The global flip is removed by asserting S[0] = +1 on both sides.
    """
    PHI = np.atleast_2d(np.asarray(PHI, float))
    duties = np.asarray(duties, float)
    M_th = np.atleast_1d(np.asarray(M_th, float))
    n = PHI.shape[1]
    err = np.zeros((M_th.size, duties.size))
    unres = np.zeros((M_th.size, duties.size))
    for jd, duty in enumerate(duties):
        for phi in PHI:
            C = C_matrix_all_bases(phi, duty=duty, deadzone=deadzone,
                                   n_periods=n_periods,
                                   pts_per_period=pts_per_period)
            s_true = spins_from_phase(phi)
            s_true = s_true * s_true[0]
            for jm, m in enumerate(M_th):
                S = algorithm1(lambda b, i: C[b][i], n, M_th=m)["S"]
                unres[jm, jd] += np.mean(S == 0)
                err[jm, jd] += np.mean((S == 0) | (S * s_true < 0))
    return dict(duty=duties, M_th=M_th, error=100.0 * err / PHI.shape[0],
                unresolved=100.0 * unres / PHI.shape[0],
                cliff=(100.0 - M_th) / 200.0)


# =============================================================================
#  STAGE 5 -- DEGENERACY, 2f FREQUENCY MULTIPLICATION, FREQUENCY SWEEPING
#
#  The paper names three stable oscillator states: 0/180 phase shift,
#  90 phase shift, and frequency multiplication. It maps the last two onto the
#  SAME readout value (C ~ 50 %) and says so explicitly -- the readout cannot
#  tell a 90 deg lock from an oscillator running at 2 f.
#
#  The rising-edge counter does tell them apart, in one extra counter per spin.
#  That is the whole point of adding detectors (d) and (b) to the readout.
# =============================================================================

STATE_INPHASE = "in-phase (0 deg)"
STATE_ANTI = "anti-phase (180 deg)"
STATE_QUAD = "quadrature (90 deg)"
STATE_MULT = "frequency multiplication (2 f)"
STATE_UNLOCK = "unlocked / drifting"
STATE_BAD = "UNREADABLE"


def classify_oscillator(Vb, Vi, M_th=20.0, vdd=VDD, ratio_tol=0.05,
                        toggle_tol=0.02, deadzone=0.0):
    """Full four-detector classification of one oscillator against the base.

    Order matters: frequency is checked BEFORE phase, because a 2f oscillator
    has no well-defined phase relative to the base and its C is meaningless.
    """
    C = pd_xor(Vb, Vi, vdd, deadzone=deadzone)
    ratio = pd_edge_ratio(Vb, Vi, vdd)
    dff = pd_dff(Vb, Vi, vdd)
    info = dict(C=C, edge_ratio=ratio, dff_mean=dff["mean"],
                dff_toggle=dff["toggle"])
    if not np.isfinite(ratio):
        return dict(state=STATE_BAD, spin=0.0, stable=False, **info)
    if abs(ratio - 1.0) > ratio_tol:
        return dict(state=STATE_MULT, spin=0.0, stable=True, **info)
    if np.isfinite(dff["toggle"]) and dff["toggle"] > toggle_tol:
        return dict(state=STATE_UNLOCK, spin=0.0, stable=False, **info)
    if C < M_th:
        return dict(state=STATE_INPHASE, spin=+1.0, stable=True, **info)
    if C > 100 - M_th:
        return dict(state=STATE_ANTI, spin=-1.0, stable=True, **info)
    if 50 - M_th < C < 50 + M_th:
        return dict(state=STATE_QUAD, spin=0.0, stable=True, **info)
    return dict(state=STATE_BAD, spin=0.0, stable=False, **info)


def classify_network(phi, f=None, base=0, M_th=20.0, duty=0.5,
                     n_periods=200, pts_per_period=256, deadzone=0.0):
    """Classify every oscillator of a settled network. Returns spins + states."""
    t, V = make_waveforms(phi, f=f, duty=duty, n_periods=n_periods,
                          pts_per_period=pts_per_period)
    n = V.shape[0]
    rows = [classify_oscillator(V[base], V[i], M_th=M_th, deadzone=deadzone)
            for i in range(n)]
    S = np.array([r["spin"] for r in rows])
    return dict(S=S, rows=rows, stable=all(r["stable"] for r in rows),
                states=[r["state"] for r in rows], V=V, t=t)


def lock_prediction(dw, Ks):
    """Paper / slide result: |dw| < Ks locks at phi* = 0.5 asin(dw/Ks);
    |dw| > Ks loses lock and the phase drifts at sqrt(dw^2 - Ks^2)."""
    dw = np.asarray(dw, float)
    locked = np.abs(dw) < Ks
    phi_star = np.where(locked, 0.5 * np.arcsin(np.clip(dw / max(Ks, 1e-12),
                                                        -1, 1)), np.nan)
    drift_rate = np.where(locked, 0.0,
                          np.sqrt(np.maximum(dw ** 2 - Ks ** 2, 0.0)))
    return dict(locked=locked, phi_star_deg=np.degrees(phi_star),
                drift_rate=drift_rate)


def detuning_sweep(Ks=KS_DEFAULT, ratios=None, T=400.0, dt=None):
    """Single spin under SHIL only: reproduce the locked / unlocked transition."""
    ratios = np.linspace(0, 2, 41) if ratios is None else np.asarray(ratios, float)
    dt = (0.2 / (2 * Ks)) if dt is None else dt
    J = np.zeros((1, 1))
    phi_star, rate = [], []
    for r in ratios:
        p = OIMParams(Kc=0.0, Ks=Ks, D=0.0, dt=dt, T=T, dw=np.array([r * Ks]))
        res = simulate(J, p, phi0=np.array([0.0]))
        tail = res.phi[-int(0.25 * len(res.t)):, 0]
        phi_star.append(np.degrees(np.mod(tail[-1] + np.pi, TWO_PI) - np.pi))
        rate.append((tail[-1] - tail[0]) / (0.25 * T))
    return dict(ratio=ratios, phi_star_deg=np.array(phi_star),
                drift_rate=np.abs(np.array(rate)), Ks=Ks,
                theory=lock_prediction(ratios * Ks, Ks))


def frequency_sweep_solutions(A, tunes=None, Ks=KS_DEFAULT, D=0.0, T=80.0,
                              mismatch=None, seed=0, Kc=1.0, verbose=False):
    """Paper Section V: sweep the tuning capacitance -> sweep the detuning
    vector -> start the network in a different part of the energy landscape ->
    recover a DIFFERENT global minimum each time.

    A single global scale on omega cannot do this (it only rescales time:
    w * t_settle = const). It is the RELATIVE detuning that moves the solution.
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    J = adjacency_to_J(A)
    best, opt, ndeg = brute_force_maxcut(A)
    tunes = np.linspace(-0.30, 0.30, 25) if tunes is None else np.asarray(tunes, float)
    if mismatch is None:
        rng = np.random.default_rng(seed)
        mismatch = rng.standard_normal(n)
        mismatch -= mismatch.mean()
        mismatch /= np.abs(mismatch).max()
    dt = stable_dt(A, Ks, Kc)
    found, cuts, phis = [], [], []
    for tune in tunes:
        p = OIMParams(Kc=Kc, Ks_schedule=linear_ramp(Ks, 0.05 * T, 0.20 * T),
                      D=D, dt=dt, T=T, dw=tune * mismatch, seed=seed)
        r = simulate(J, p, phi0=_start(seed, n))
        s = spins_from_phase(r.phi_final)
        s = s * s[0]
        cuts.append(cut_value(A, s))
        found.append(tuple(int(v) for v in s))
        phis.append(r.phi_final)
    uniq = {}
    for tune, key, c in zip(tunes, found, cuts):
        if c >= best:
            uniq.setdefault(key, []).append(float(tune))
    if verbose:
        print(f"    degeneracy = {ndeg}, distinct optima recovered = {len(uniq)}")
    return dict(tune=tunes, cut=np.array(cuts), spins=found,
                phi=np.array(phis), best=best, degeneracy=ndeg,
                unique_optima=uniq, mismatch=mismatch,
                coverage=len(uniq) / ndeg)


# =============================================================================
#  STAGE 6 -- BIT-EXACT FIXED-POINT TWIN OF oim_pynq.v
#
#  Every number below has exactly one representation, shared with the RTL.
#  If you change one, change BOTH. The self-test cross-checks this model
#  against the float model; tb-free RTL agreement is checked separately by
#  running the same vectors through Icarus.
#
#    phase   PW = 16 bits UNSIGNED. 2^16 == 2*pi, so the adder wraps for free.
#            1 LSB = 360/65536 = 0.00549 deg.
#    SIN     1024-entry signed 16-bit table, SIN[k] = round(32768*sin(2 pi k/1024))
#            clipped into signed 16-bit. Index = phase[15:6].
#    J       signed 4-bit integer per element (-8..+7). MaxCut uses J = -A.
#    KC,KS   unsigned Q4.12 (4096 == 1.0)
#    W       unsigned Q4.12 gain per spin (4096 == 1.0)
#    DW      signed Q1.15 detuning per spin (32768 == 1.0 rad per unit time)
#    DTQ     unsigned, DTQ = round((dt/pi) * 2^24)     [see derivation below]
#
#  Derivation of DTQ. The real update is  dphi = dt * r  radians, and
#  1 radian = 65536/(2 pi) LSB, so
#      delta_LSB = dt * (r_q/32768) * 65536/(2 pi) = r_q * dt/pi.
#  Hence delta = (DTQ * r_q) >> 24 with DTQ = round(dt/pi * 2^24).
#
#  Noise. The RTL adds a UNIFORM increment, not a Gaussian one. That is not an
#  approximation of convenience: Euler-Maruyama converges to the same SDE for
#  any zero-mean i.i.d. increment with matched variance, because it is the sum
#  over many steps that has to be Gaussian, and the CLT supplies that. Matching
#  the variance:  var = 2 D dt rad^2 -> a = sqrt(6 D dt) * 65536/(2 pi).
# =============================================================================

PW = 16
PHASE_MASK = (1 << PW) - 1
LUT_BITS = 10
LUT_N = 1 << LUT_BITS
SIN_SHIFT = PW - LUT_BITS          # = 6
SIN_SCALE = 32768
Q12 = 4096
DT_SHIFT = 24
RAD_PER_LSB = TWO_PI / (1 << PW)
LSB_PER_RAD = (1 << PW) / TWO_PI


def build_sin_lut():
    v = np.round(SIN_SCALE * np.sin(TWO_PI * np.arange(LUT_N) / LUT_N))
    return np.clip(v, -32768, 32767).astype(np.int64)


SIN_LUT = build_sin_lut()


def write_sin_lut_mem(path="sin_lut.mem"):
    """16-bit hex, one entry per line -- $readmemh format for the RTL."""
    with open(path, "w") as fh:
        for v in SIN_LUT:
            fh.write(f"{int(v) & 0xFFFF:04X}\n")
    return path


def _sra(x, n):
    """Arithmetic right shift matching Verilog's >>> on a signed value."""
    return int(np.floor(x / (1 << n))) if x < 0 else (int(x) >> n)


def q_dt(dt):
    return int(round((dt / np.pi) * (1 << DT_SHIFT)))


def q_noise_amp(D, dt):
    return int(round(np.sqrt(6.0 * D * dt) * LSB_PER_RAD))


def q_ks(ks):
    return int(round(ks * Q12))


def q_dw(dw):
    return int(round(dw * SIN_SCALE))


class LFSR32:
    """32-bit maximal-length Galois LFSR, taps 32,22,2,1 (poly 0x80200003).
    Identical construction to the one already in Jarif's pbit_engine.v."""

    def __init__(self, seed):
        self.s = int(seed) & 0xFFFFFFFF or 0x1

    def next(self):
        """Returns the CURRENT state, then advances.

        This ordering matters: in the RTL the datapath reads lfsr_q in the
        S_SHIL cycle and only then pulses lfsr_en, so the first spin of the
        first timestep sees the seed itself. Advancing first would put the
        Python twin one draw ahead of the hardware for every noisy run.
        """
        cur = self.s
        lsb = self.s & 1
        self.s >>= 1
        if lsb:
            self.s ^= 0xA3000000
        self.s &= 0xFFFFFFFF
        return cur


@dataclass
class FixedConfig:
    N: int
    J: np.ndarray                  # int, signed 4-bit range
    KC: int = Q12
    KS: int = 0
    W: np.ndarray = None           # Q4.12 per spin
    DW: np.ndarray = None          # Q1.15 per spin
    DTQ: int = 0
    NAMP: int = 0
    SEED: int = 1
    KS_START: int = 0
    KS_STOP: int = 0
    KS_STEPS: int = 0              # 0 = no ramp, Ks held at KS
    TOL: int = 60                  # settling tolerance, phase LSB (~0.33 deg)
    STRIDE: int = 64
    HOLD: int = 8

    def ks_at_step(self, k):
        if self.KS_STEPS <= 0:
            return self.KS
        if k >= self.KS_STEPS:
            return self.KS_STOP
        return self.KS_START + ((self.KS_STOP - self.KS_START) * k) // self.KS_STEPS


def fixed_step(phase, cfg: FixedConfig, ks_q, rng: LFSR32):
    """ONE timestep of the RTL datapath, integer-for-integer. Jacobi update."""
    N = cfg.N
    W = cfg.W if cfg.W is not None else np.full(N, Q12, dtype=np.int64)
    DW = cfg.DW if cfg.DW is not None else np.zeros(N, dtype=np.int64)
    new = np.empty(N, dtype=np.int64)
    for i in range(N):
        acc = 0
        for j in range(N):
            if j == i:
                continue
            d = (int(phase[i]) - int(phase[j])) & PHASE_MASK
            acc += int(cfg.J[i, j]) * int(SIN_LUT[d >> SIN_SHIFT])
        sh = int(SIN_LUT[((int(phase[i]) << 1) & PHASE_MASK) >> SIN_SHIFT])
        g = _sra(cfg.KC * acc, 12) + _sra(ks_q * sh, 12)
        wg = _sra(int(W[i]) * g, 12)
        r = int(DW[i]) - wg
        delta = _sra(cfg.DTQ * r, DT_SHIFT)
        if cfg.NAMP:
            u = rng.next() & 0xFFFF
            if u >= 0x8000:
                u -= 0x10000
            delta += _sra(u * cfg.NAMP, 15)
        new[i] = (int(phase[i]) + delta) & PHASE_MASK
    return new


def _wrapped_abs(a, b):
    d = (int(a) - int(b)) & PHASE_MASK
    if d >= 0x8000:
        d -= 0x10000
    return abs(d)


def fixed_run(cfg: FixedConfig, phi0_q, max_steps=20000):
    """Run to settling or the step budget. Returns the same flags the RTL exposes."""
    phase = np.array(phi0_q, dtype=np.int64) & PHASE_MASK
    rng = LFSR32(cfg.SEED)
    snap = phase.copy()
    hold = 0
    traj = [phase.copy()]
    settled_at = None
    for k in range(max_steps):
        phase = fixed_step(phase, cfg, cfg.ks_at_step(k), rng)
        traj.append(phase.copy())
        if (k + 1) % cfg.STRIDE == 0:
            dmax = max(_wrapped_abs(phase[i], snap[i]) for i in range(cfg.N))
            hold = hold + 1 if dmax < cfg.TOL else 0
            snap = phase.copy()
            if hold >= cfg.HOLD and settled_at is None:
                settled_at = k + 1
                break
    return dict(phase=phase, traj=np.array(traj), steps=k + 1,
                settled=settled_at is not None, settled_at=settled_at)


def fixed_to_rad(phase_q):
    return np.asarray(phase_q, float) * RAD_PER_LSB


def make_fixed_cfg(A, Ks=KS_DEFAULT, dt=None, D=0.0, Kc=1.0, seed=1, dw=None,
                   ramp=(0.0, None, 0)):
    """Build a FixedConfig from the same physical numbers the float model uses."""
    A = np.asarray(A, float)
    N = A.shape[0]
    ok, win = check_ks(A, Ks)
    if ok is False:
        import warnings
        warnings.warn(
            f"Ks/Kc = {Ks} is OUTSIDE this graph's measured window "
            f"{win[0]:.2f}-{win[1]:.2f}. The cut may still round out correctly "
            f"but the phases will not binarise, so the readout will leave "
            f"spins unresolved. Use recommend_ks(A).", RuntimeWarning,
            stacklevel=2)
    dt = stable_dt(A, Ks, Kc) if dt is None else dt
    J = np.round(adjacency_to_J(A)).astype(np.int64)
    if J.min() < -8 or J.max() > 7:
        raise ValueError("J does not fit signed 4-bit")
    ks_start, ks_stop, ks_steps = ramp
    cfg = FixedConfig(
        N=N, J=J, KC=q_ks(Kc), KS=q_ks(Ks),
        W=np.full(N, Q12, dtype=np.int64),
        DW=(np.zeros(N, dtype=np.int64) if dw is None
            else np.array([q_dw(x) for x in dw], dtype=np.int64)),
        DTQ=q_dt(dt), NAMP=q_noise_amp(D, dt), SEED=seed,
        KS_START=q_ks(ks_start),
        KS_STOP=q_ks(Ks if ks_stop is None else ks_stop),
        KS_STEPS=int(ks_steps))
    return cfg, dt


def rtl_vector_dump(cfg: FixedConfig, phi0_q, nsteps, path="golden.txt"):
    """Golden trace: the exact phase words after every timestep.
    The RTL must reproduce this file line for line."""
    phase = np.array(phi0_q, dtype=np.int64) & PHASE_MASK
    rng = LFSR32(cfg.SEED)
    with open(path, "w") as fh:
        fh.write(" ".join(f"{int(v):04X}" for v in phase) + "\n")
        for k in range(nsteps):
            phase = fixed_step(phase, cfg, cfg.ks_at_step(k), rng)
            fh.write(" ".join(f"{int(v):04X}" for v in phase) + "\n")
    return path


# =============================================================================
#  CALL TRACER -- see exactly which function ran, with what, and for how long.
#
#  Every public function in this module gets wrapped. The trace prints an
#  indented call tree with elapsed wall time, abbreviated arguments and an
#  abbreviated return value, then a summary table at the end.
#
#  Hot inner functions (square_wave, pd_xor, fixed_step, ...) are called
#  hundreds of thousands of times, so the tracer caps how many calls of each
#  name it prints and collapses the rest -- they still appear in the summary
#  with full call counts and total time.
#
#    python3 oim_sim.py --all              trace on (default)
#    python3 oim_sim.py --all --no-trace   silent
#    python3 oim_sim.py --all --trace-depth 5 --trace-cap 10   more detail
# =============================================================================

import inspect
import time

_TRACE = dict(on=False, depth=0, max_depth=3, cap=2, t0=None,
              counts={}, times={}, order=[])

#  never wrapped: tracer internals, plotting helpers, and trivial accessors
_NEVER_TRACE = {
    "_check", "_fmt", "_stage_fig", "_place", "_abbrev", "_abbrev_safe", "_stamp",
    "_trace_wrap", "install_tracer", "trace_summary", "_mpl", "figures",
    "gui_report", "_open_folder",
    "main", "_sym",
}


def _stamp():
    dt = 0.0 if _TRACE["t0"] is None else time.perf_counter() - _TRACE["t0"]
    return f"[{dt:7.2f}s] "


def _abbrev(v, maxlen=34):
    """Compact one-line rendering of an argument or return value.

    Wrapped in a blanket except by _abbrev_safe(): a diagnostic must never be
    able to crash the computation it is reporting on.
    """
    if isinstance(v, np.ndarray):
        if v.size <= 6 and v.ndim == 1:
            return "[" + ",".join(f"{float(x):.3g}" for x in v) + "]"
        return f"ndarray{tuple(v.shape)}"
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, str):
        return repr(v if len(v) <= maxlen else v[: maxlen - 3] + "...")
    if isinstance(v, dict):
        keys = ",".join(str(k) for k in list(v)[:4])   # keys need not be str
        return f"dict{{{keys}{',...' if len(v) > 4 else ''}}}"
    if isinstance(v, (list, tuple, set)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, OIMParams):
        return (f"OIMParams(Kc={v.Kc:g}, Ks={v.Ks:g}, D={v.D:g}, "
                f"dt={v.dt:.4g}, T={v.T:g})")
    if isinstance(v, SimResult):
        return f"SimResult(steps={len(v.t)}, n={v.phi.shape[1]})"
    if isinstance(v, FixedConfig):
        return f"FixedConfig(N={v.N}, DTQ={v.DTQ}, NAMP={v.NAMP})"
    if callable(v):
        return getattr(v, "__name__", "fn") + "()"
    return type(v).__name__


def _abbrev_safe(v):
    try:
        return _abbrev(v)
    except Exception:
        return type(v).__name__


def _trace_wrap(fn, name):
    sig_names = list(inspect.signature(fn).parameters)

    def wrapper(*a, **kw):
        st = _TRACE
        if not st["on"]:
            return fn(*a, **kw)
        n = st["counts"].get(name, 0)
        st["counts"][name] = n + 1
        if name not in st["times"]:
            st["times"][name] = 0.0
            st["order"].append(name)
        show = st["depth"] < st["max_depth"] and n < st["cap"]
        pad = "|  " * st["depth"]
        if show:
            parts = [f"{sig_names[i] if i < len(sig_names) else '*'}="
                     f"{_abbrev_safe(x)}" for i, x in enumerate(a)]
            parts += [f"{k}={_abbrev_safe(v)}" for k, v in kw.items()]
            print(f"{_stamp()}{pad}+- {name}(" + ", ".join(parts) + ")",
                  flush=True)
        elif n == st["cap"] and st["depth"] < st["max_depth"]:
            print(f"{_stamp()}{pad}   ... {name}(...) further calls collapsed "
                  f"-- see the summary", flush=True)
        st["depth"] += 1
        t = time.perf_counter()
        try:
            out = fn(*a, **kw)
        finally:
            st["depth"] -= 1
            el = time.perf_counter() - t
            st["times"][name] = st["times"].get(name, 0.0) + el
        if show:
            print(f"{_stamp()}{pad}`- {name} -> {_abbrev_safe(out)}"
                  f"   [{el*1000:.1f} ms]", flush=True)
        return out

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


def install_tracer(on=True, max_depth=3, cap=2):
    """Wrap every module-level function so calls print themselves."""
    _TRACE.update(on=on, max_depth=max_depth, cap=cap,
                  t0=time.perf_counter(), depth=0)
    _TRACE["counts"].clear()
    _TRACE["times"].clear()
    _TRACE["order"].clear()
    if not on:
        return
    g = globals()
    for k, v in list(g.items()):
        if not inspect.isfunction(v):
            continue
        if k.startswith("__") or k in _NEVER_TRACE:
            continue
        if getattr(v, "__wrapped__", None) is not None:
            continue
        g[k] = _trace_wrap(v, k)


def trace_summary(top=28):
    """Call counts and cumulative time, heaviest first."""
    if not _TRACE["counts"]:
        return
    rows = sorted(((n, _TRACE["counts"][n], _TRACE["times"].get(n, 0.0))
                   for n in _TRACE["counts"]), key=lambda r: -r[2])
    total = time.perf_counter() - _TRACE["t0"]
    print("\n" + "=" * 78)
    print(f" CALL SUMMARY   ({len(rows)} distinct functions, "
          f"{sum(r[1] for r in rows):,} calls, {total:.1f} s wall)")
    print("=" * 78)
    print(" cumulative time is INCLUSIVE of nested calls, so the shares of a")
    print(" parent and its children overlap and the column sums past 100 %.")
    print(" " + "-" * 74)
    print(f" {'function':<34}{'calls':>12}{'cumulative':>14}{'share':>10}")
    print(" " + "-" * 74)
    for name, cnt, tt in rows[:top]:
        print(f" {name:<34}{cnt:>12,}{tt:>12.2f} s{100*tt/total:>9.1f} %")
    if len(rows) > top:
        print(f" ... and {len(rows)-top} more")
    print("=" * 78 + "\n")


# =============================================================================
#  SELF-TESTS -- every claim this file makes about itself, checked.
#  Run:  python3 oim_sim.py --checks
# =============================================================================

_CHECKS = []


def _check(name, cond, detail=""):
    _CHECKS.append((name, bool(cond), detail))
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def run_checks(verbose=True):
    _CHECKS.clear()
    print("\n" + "=" * 78)
    print(" SELF-TESTS")
    print("=" * 78)

    # --- 1. the Ising <-> MaxCut mapping ------------------------------------
    for nm, A in [("size6", size6_graph()), ("size5", size5_graph()),
                  ("K4", k4_graph())]:
        S = enumerate_states(A.shape[0], fix_first=False)
        e = max(abs(cut_value(A, s) - eq2_cut_from_H(A, s)) for s in S)
        _check(f"Eq.(2) identity holds on {nm}", e < 1e-12, f"max error {e:.1e}")
    for nm, A, cut, deg in [("size6", size6_graph(), 7, 1),
                            ("size5", size5_graph(), 4, 3),
                            ("K4", k4_graph(), 4, 3)]:
        b, _, d = brute_force_maxcut(A)
        _check(f"{nm}: MaxCut={cut}, degeneracy={deg}", b == cut and d == deg,
               f"got {b:.0f}/{d}")
    _check("H_ordered == 2 * H_pair",
           abs(H_ordered(adjacency_to_J(size6_graph()), np.ones(6))
               - 2 * H_pair(adjacency_to_J(size6_graph()), np.ones(6))) < 1e-12)

    # --- 2. madam's square-wave formula -------------------------------------
    t = np.linspace(0, 10, 100003)
    worst = 0.0
    for ph in (0.0, 0.7, np.pi, 4.2):
        a = madam_formula(t, TWO_PI, ph)
        b = square_wave(t, 1.0, ph, 0.5)
        ok = np.abs(np.sin(TWO_PI * t + ph)) > 1e-9
        worst = max(worst, float(np.abs(a[ok] - b[ok]).max()))
    _check("VDD/2*sgn(sin(wt+phi))+VDD/2 == square_wave(duty=0.5)", worst == 0.0,
           f"max |diff| {worst:.1e} V")

    # --- 3. integrator stability limit --------------------------------------
    A = size6_graph()
    J = adjacency_to_J(A)
    Ks = KS_DEFAULT
    L = lipschitz_L(A, Ks)
    dt0 = stable_dt(A, Ks)
    phi0 = np.full(6, 0.3) + np.arange(6) * 0.4
    ok_in = simulate(J, OIMParams(Ks=Ks, dt=dt0, T=20), phi0=phi0).energy_monotone()
    bad_out = simulate(J, OIMParams(Ks=Ks, dt=14 * dt0, T=20),
                       phi0=phi0).energy_monotone()
    _check("E falls monotonically inside dt < 2/L", ok_in, f"dt={dt0:.4f}, 2/L={2/L:.4f}")
    _check("E is NOT monotone at 1.4 x (2/L)", not bad_out)

    # --- 4. XOR readout law and the unreadable band -------------------------
    r = sweep_phase_shift(n_pts=721, duty=0.5, n_periods=100,
                          pts_per_period=1024, M_th=20.0)
    _check("measured XOR matches C = 100*min(2t,2d,2(1-t))",
           np.abs(r["C"] - r["C_ideal"]).max() < 0.5,
           f"max |diff| {np.abs(r['C']-r['C_ideal']).max():.3f} %")
    an = r["analytic"]
    _check("unreadable width: measured == closed form",
           abs(r["bad_width_measured"] - an["bad_width"]) <= 2.0,
           f"{r['bad_width_measured']:.1f} deg vs {an['bad_width']:.1f} deg")
    _check("M_th=20 -> four 18 deg slivers, 72 deg total",
           abs(an["bad_width"] - 72.0) < 1e-6 and len(an["bad"]) == 4)
    _check("unreadable set closes exactly at M_th = 25 %",
           unreadable_bands_analytic(25.0, 0.5)["bad_width"] < 1e-9
           and unreadable_bands_analytic(24.0, 0.5)["bad_width"] > 0)
    _check("duty cliff: anti-phase anchor unreachable when 200d < 100-M_th",
           unreadable_bands_analytic(20.0, 0.35)["collapsed"]
           and not unreadable_bands_analytic(20.0, 0.45)["collapsed"])

    # --- 5. RS flip-flop: range, duty immunity, and the blind spot ----------
    t = time_grid(60, 1024)
    Vb = square_wave(t, 1.0, 0.0, 0.5)
    d = np.mod(r["delta"], 360.0)
    mask = (d > 3.0) & (d < 357.0)
    _check("RS-FF follows the sawtooth 1 - dphi/360 away from 0 deg",
           np.nanmax(np.abs(r["rsff"] - r["rsff_ideal"])[mask]) < 1.0,
           f"max |diff| {np.nanmax(np.abs(r['rsff']-r['rsff_ideal'])[mask]):.3f} %")
    at0 = pd_rsff(Vb, square_wave(t, 1.0, 0.0, 0.5))
    at1 = pd_rsff(Vb, square_wave(t, 1.0, np.radians(1.0), 0.5))
    _check("RS-FF has a ~100 % discontinuity AT dphi = 0 (blind spot)",
           at0 < 1.0 and at1 > 99.0, f"{at0:.2f} % -> {at1:.2f} %")
    dutyskew = [pd_rsff(Vb, square_wave(t, 1.0, np.radians(120.0), dd))
                for dd in (0.5, 0.45, 0.40, 0.35)]
    _check("RS-FF is immune to duty distortion (XOR is not)",
           max(dutyskew) - min(dutyskew) < 1.0,
           "spread %.2f %% over duty 0.35..0.50" % (max(dutyskew) - min(dutyskew)))
    # Duty distortion does NOT shift the XOR characteristic. It CLIPS THE PEAK.
    # On the rising arm C = 100*dphi/180 exactly, for any duty > dphi/360, so the
    # +1 branch and the 50 % free branch are untouched. Only the anti-phase
    # anchor moves, from 100 % down to 200*d -- which is why the spin-error
    # curve against duty is a cliff and not a slope.
    arm = [pd_xor(square_wave(t, 1.0, 0.0, dd),
                  square_wave(t, 1.0, np.radians(120.0), dd))
           for dd in (0.5, 0.45, 0.40, 0.35)]
    _check("XOR rising arm is duty-INDEPENDENT (C = 100*dphi/180)",
           max(arm) - min(arm) < 0.5 and abs(np.mean(arm) - 66.667) < 0.5,
           "C(120 deg) = %.2f %% at every duty" % np.mean(arm))
    peak = [pd_xor(square_wave(t, 1.0, 0.0, dd),
                   square_wave(t, 1.0, np.radians(180.0), dd))
            for dd in (0.5, 0.45, 0.40, 0.35)]
    _check("XOR anti-phase anchor collapses to 200*d",
           all(abs(p_ - 200 * dd) < 0.3
               for p_, dd in zip(peak, (0.5, 0.45, 0.40, 0.35))),
           "C(180 deg) = " + ", ".join(f"{p_:.1f}" for p_ in peak))

    # --- 6. the 2f discrimination -------------------------------------------
    t = time_grid(200, 256)
    Vb = square_wave(t, 1.0, 0.0, 0.5)
    Cq = pd_xor(Vb, square_wave(t, 1.0, np.radians(90), 0.5))
    Cm = pd_xor(Vb, square_wave(t, 2.0, np.radians(37), 0.5))
    _check("XOR alone CANNOT separate 90 deg lock from 2f (paper's limitation)",
           abs(Cq - Cm) < 0.5, f"C = {Cq:.2f} %% vs {Cm:.2f} %%")
    rq = pd_edge_ratio(Vb, square_wave(t, 1.0, np.radians(90), 0.5))
    rm = pd_edge_ratio(Vb, square_wave(t, 2.0, np.radians(37), 0.5))
    _check("edge counter DOES separate them", abs(rq - 1) < 0.05 and abs(rm - 2) < 0.05,
           f"ratio {rq:.3f} vs {rm:.3f}")
    tg_lock = pd_dff(Vb, square_wave(t, 1.0, np.radians(90), 0.5))["toggle"]
    tg_unlk = pd_dff(Vb, square_wave(t, 1.013, 0.0, 0.5))["toggle"]
    _check("D-FF toggle rate separates locked from drifting",
           tg_lock < 0.005 and tg_unlk > 0.01,
           f"toggle {tg_lock:.4f} (locked) vs {tg_unlk:.4f} (unlocked)")

    # --- 7. madam's circle diagram, end to end ------------------------------
    A5 = size5_graph()
    best5 = brute_force_maxcut(A5)[0]
    phi = np.radians([0.0, 37.0, 180.0, 180.0, 0.0])
    f = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    res = classify_network(phi, f=f, base=0, M_th=20.0)
    _check("size-5 readout gives S = [+1, 0, -1, -1, +1]",
           np.array_equal(res["S"], np.array([1, 0, -1, -1, 1.0])),
           f"S = {res['S'].astype(int)}")
    _check("osc2 is reported as frequency multiplication",
           res["states"][1] == STATE_MULT)
    exp = expand_free(res["S"])
    _check("free vertex expands to BOTH bipartitions the paper quotes",
           len(exp) == 2 and all(cut_value(A5, s) >= best5 for s in exp),
           " and ".join(str(partition_of(s)) for s in exp))

    # --- 8. locking / detuning theory ---------------------------------------
    ds = detuning_sweep(Ks=KS_DEFAULT, ratios=np.linspace(0, 1.9, 20), T=300.0)
    th = ds["theory"]
    lk = ds["ratio"] < 0.95
    err = np.abs(np.abs(ds["phi_star_deg"][lk]) - np.abs(th["phi_star_deg"][lk]))
    _check("locked branch: phi* = 0.5 asin(dw/Ks)", np.nanmax(err) < 3.0,
           f"max |diff| {np.nanmax(err):.2f} deg")
    ul = ds["ratio"] > 1.15
    err2 = np.abs(ds["drift_rate"][ul] - th["drift_rate"][ul])
    _check("unlocked branch: drift = sqrt(dw^2 - Ks^2)", np.nanmax(err2) < 0.05,
           f"max |diff| {np.nanmax(err2):.4f}")

    # --- 9. fixed-point twin vs float ---------------------------------------
    A = size6_graph()
    cfg, dt = make_fixed_cfg(A, Ks=KS_DEFAULT, D=0.0, seed=1)
    cfg.HOLD = 10 ** 9
    p0 = _start(0, 6)
    p0q = np.round(p0 * LSB_PER_RAD).astype(np.int64) & PHASE_MASK
    steps = int(round(60.0 / dt))
    fx = fixed_run(cfg, p0q, max_steps=steps)
    fl = simulate(adjacency_to_J(A), OIMParams(Ks=KS_DEFAULT, dt=dt, T=steps * dt), phi0=p0)
    sfx = spins_from_phase(fixed_to_rad(fx["phase"]))
    sfl = spins_from_phase(fl.phi_final)
    _check("fixed-point twin recovers the same spins as float",
           np.array_equal(sfx * sfx[0], sfl * sfl[0]))
    _check("fixed-point twin recovers the same cut",
           cut_value(A, sfx) == cut_value(A, sfl),
           f"cut {cut_value(A, sfx):.0f}")
    _check("sine LUT is 1024 x signed 16-bit and monotone on the first quadrant",
           SIN_LUT.size == 1024 and SIN_LUT.max() <= 32767
           and SIN_LUT.min() >= -32768
           and np.all(np.diff(SIN_LUT[:257]) >= 0))
    _check("DTQ derivation: delta_LSB = r_q * dt/pi (to the 2^-24 quantum)",
           abs(q_dt(0.03226) / (1 << DT_SHIFT) - 0.03226 / np.pi) <= 0.5 / (1 << DT_SHIFT),
           f"residual {abs(q_dt(0.03226)/(1<<DT_SHIFT)-0.03226/np.pi):.2e} "
           f"<= quantum {0.5/(1<<DT_SHIFT):.2e}")

    # --- 10. Algorithm 1 repairs --------------------------------------------
    Cm2 = np.array([[0, 100, 50], [100, 0, 50], [50, 50, 0]], float)
    out = algorithm1(lambda b, i: Cm2[b][i], 3, M_th=20.0)
    _check("Algorithm 1: free spin stays 0, is not promoted by a later base",
           np.array_equal(out["S"], np.array([1.0, -1.0, 0.0])),
           f"S = {out['S'].astype(int)}")

    # --- 11. Kim (Nature Electronics 2022) readout -------------------------
    tab = kim_code_table()
    got = [format(c, "04b") for c in sorted(tab)]
    _check("Kim phase-ID: exactly the 8 legal codes the paper prints",
           got == ["0000", "0001", "0011", "0111", "1000", "1100", "1110", "1111"],
           ", ".join(got))
    _check("Kim phase-ID windows tile the full circle",
           abs(sum(hi - lo for lo, hi, _ in tab.values()) - 360.0) < 0.5,
           f"{sum(hi-lo for lo,hi,_ in tab.values()):.1f} deg over {len(tab)} bins")
    ph = np.linspace(0, 359.9, 721)
    derr = np.abs(((kim_decode(kim_phase_id(ph)) - ph + 180) % 360) - 180)
    _check("Kim phase-ID decode is coarse but bounded (<= 30 deg)",
           derr.max() <= 30.5, f"max {derr.max():.1f} deg, mean {derr.mean():.1f} deg")
    Ak = kings_graph(41, 48)
    _check("King's graph 41x48 reproduces Kim's 1,968-node array",
           Ak.shape[0] == 1968 and int(Ak.sum(1).max()) == 8,
           f"n={Ak.shape[0]}, max degree={int(Ak.sum(1).max())}")
    # drift immunity, on a graph where 'far from the base' actually exists
    Ap = np.zeros((10, 10))
    for i in range(9):
        Ap[i, i + 1] = Ap[i + 1, i] = 1.0
    rr = compare_readouts_under_drift(A=Ap, amplitudes=np.arange(0, 181, 30.0),
                                      reps=4)
    _check("on a high-diameter graph Kim's neighbour scheme beats a fixed base",
           rr["kim"].mean() > rr["algorithm1"].mean(),
           f"Kim {rr['kim'].mean()/rr['best']:.3f} vs Algorithm 1 "
           f"{rr['algorithm1'].mean()/rr['best']:.3f} (normalised cut)")
    # Annealing Ks turns out to be a RESCUE for a badly chosen Ks, not a
    # benefit in its own right. At KS_DEFAULT it changes nothing; at Ks = 1.2
    # it lifts P(opt) from about 0.24 to about 0.89. An earlier version of
    # this file claimed annealing "DOES help" -- that was measured at Ks = 0.6,
    # i.e. outside the useful window, and the gain was really the window being
    # wrong. Corrected, this AGREES with Kim's null annealing result rather
    # than contradicting it.
    ab_ok = annealing_benefit(Ks=KS_DEFAULT, B=120)
    ab_hi = annealing_benefit(Ks=1.2, B=120)
    _check("at KS_DEFAULT, annealing Ks changes nothing (agrees with Kim)",
           abs(ab_ok["Ks annealed"]["p_opt"] - ab_ok["Ks fixed"]["p_opt"]) < 0.05,
           f"P(opt) {ab_ok['Ks fixed']['p_opt']:.3f} fixed -> "
           f"{ab_ok['Ks annealed']['p_opt']:.3f} annealed")
    _check("annealing only rescues a Ks chosen ABOVE the window",
           ab_hi["Ks annealed"]["p_opt"] > ab_hi["Ks fixed"]["p_opt"] + 0.2,
           f"Ks=1.2: P(opt) {ab_hi['Ks fixed']['p_opt']:.3f} fixed -> "
           f"{ab_hi['Ks annealed']['p_opt']:.3f} annealed")

    # --- 12. the Ks window, and the ramp caveat ---------------------------
    _check(f"KS_DEFAULT sits inside the measured window "
           f"{KS_WINDOW[0]:.2f} - {KS_WINDOW[1]:.2f}",
           KS_WINDOW[0] <= KS_DEFAULT <= KS_WINDOW[1], f"KS_DEFAULT = {KS_DEFAULT}")
    _check("tuning range stays below Ks, so oscillators keep lock",
           KS_TUNE_MAX < KS_DEFAULT,
           f"max|dw|/Ks = {KS_TUNE_MAX/KS_DEFAULT:.2f} < 1")
    A6 = size6_graph()
    J6 = adjacency_to_J(A6)
    dt6 = stable_dt(A6, KS_DEFAULT)
    ramped = simulate(J6, OIMParams(Ks_schedule=linear_ramp(KS_DEFAULT, 3, 12),
                                    D=0.0, dt=dt6, T=60), phi0=_start(0, 6))
    _check("a RAMPED run need not have monotone E (ramp moves the potential)",
           not ramped.energy_monotone(),
           "rising steps occur inside the ramp window, as expected")
    _check("at KS_DEFAULT the size-6 graph is solved with NO noise",
           cut_value(A6, spins_from_phase(ramped.phi_final))
           >= brute_force_maxcut(A6)[0],
           f"cut {cut_value(A6, spins_from_phase(ramped.phi_final)):.0f} of "
           f"{brute_force_maxcut(A6)[0]:.0f}")

    # --- 13. STAGE 8: the readout twin of oim_pynq.v v2 --------------------
    D_ = deg_to_phase_word
    dg = np.arange(15.0, 360.0, 15.0)
    rt = readout_twin([0] + [D_(x) for x in dg[:15]], [256] * 16)
    pc = readout_percent(rt)
    eC = np.max(np.abs(pc["C"][1:] - xor_closed_form(dg[:15], 0.5)))
    eD = np.max(np.abs(pc["D"][1:] - rsff_closed_form(dg[:15])))
    _check("readout twin: XOR C follows the triangle closed form",
           eC < 100.0 * 2 / 256, f"max error {eC:.3f} % (bound: 2 samples of 256)")
    _check("readout twin: RS-FF D follows the sawtooth closed form",
           eD < 100.0 * 1 / 256, f"max error {eD:.3f} % (bound: 1 sample of 256)")
    Ds = [readout_percent(readout_twin([0, D_(180)], [256, 256], duty_q=dq))["D"][1]
          for dq in (0x8000, 0x7333, 0x6666, 0x599A, 0x4CCD)]
    _check("RS-FF is duty-immune: D at 180 deg identical for duty 0.50..0.30",
           max(Ds) - min(Ds) == 0.0, f"D = {Ds[0]:.3f} % at every duty")
    rl = readout_twin([D_(x) for x in (123, 200, 10, 300, 250)], [256] * 5)
    _check("locked oscillators never toggle, whatever the base phase",
           rl["tog"].sum() == 0 and rl["tog2"].sum() == 0,
           f"tog {rl['tog'].tolist()}  tog2 {rl['tog2'].tolist()}")
    rd_ = readout_twin([0, D_(90)], [256, 260])
    _check("a 1.6 % drift toggles both D-FFs while the edge ratio stays near 1",
           rd_["tog"][1] > 0 and rd_["tog2"][1] > 0
           and abs(rd_["edge"][1] / rd_["edge"][0] - 1) < 0.05,
           f"tog {rd_['tog'][1]}, tog2 {rd_['tog2'][1]}, "
           f"ratio {rd_['edge'][1] / rd_['edge'][0]:.4f}")
    r2 = readout_twin([0, D_(37)], [256, 512])
    _check("SHIL-edge D-FF: 1x toggles 2W-1 times, locked 2f never",
           r2["togs"][0] == 399 and r2["togs"][1] == 0,
           f"togS 1x {r2['togs'][0]}, 2f {r2['togs'][1]}")
    pq = D_(np.arange(0.0, 360.0, 5.0))
    rk = readout_twin(np.concatenate([[0], pq]), [256] * (pq.size + 1), window=4)
    _check("readout twin: Kim phase ID equals kim_phase_id_fixed",
           np.array_equal(rk["pid"][1:], kim_phase_id_fixed(pq)),
           f"{pq.size} phases")

    # --- 13. the Ks window is graph-dependent ------------------------------
    _check("KS_DEFAULT is inside the size-6 window it was derived from",
           check_ks(size6_graph(), KS_DEFAULT)[0] is True,
           f"window {KS_KNOWN_WINDOWS['size6']}")
    bad = [k for k, w in KS_KNOWN_WINDOWS.items()
           if not (w[0] <= KS_DEFAULT <= w[1])]
    _check("and OUTSIDE at least one other graph's window (it is not universal)",
           len(bad) > 0, "outside for: " + ", ".join(bad))
    _check("recommend_ks returns the centre of each known window",
           all(abs(recommend_ks(G, quiet=True)
                   - 0.5*(KS_KNOWN_WINDOWS[k][0]+KS_KNOWN_WINDOWS[k][1])) < 1e-6
               for k, G in (("size6", size6_graph()), ("size5", size5_graph()),
                            ("k4", k4_graph()),
                            ("surrogate16", surrogate16_graph()))))
    import warnings as _w
    with _w.catch_warnings(record=True) as _rec:
        _w.simplefilter("always")
        make_fixed_cfg(surrogate16_graph(), Ks=KS_DEFAULT)
        fired = len(_rec) > 0
    with _w.catch_warnings(record=True) as _rec2:
        _w.simplefilter("always")
        make_fixed_cfg(size6_graph(), Ks=KS_DEFAULT)
        quiet = len(_rec2) == 0
    _check("make_fixed_cfg warns only when Ks is outside THAT graph's window",
           fired and quiet)

    n_pass = sum(c for _, c, _ in _CHECKS)
    print("-" * 78)
    print(f" {n_pass} / {len(_CHECKS)} checks passed")
    print("=" * 78 + "\n")
    return n_pass == len(_CHECKS)


# =============================================================================
#  FIGURES
# =============================================================================

_FIGS = []


def _place(fig, k, x0=30, y0=30, dx=42, dy=38):
    """Cascade the windows so all of them stay visible at once."""
    try:
        win = fig.canvas.manager.window
    except Exception:
        return
    x, y = x0 + dx * (k % 10), y0 + dy * (k % 10)
    for attempt in (lambda: win.wm_geometry(f"+{x}+{y}"),   # TkAgg
                    lambda: win.move(x, y),                 # Qt
                    lambda: win.Move((x, y))):              # WX
        try:
            attempt()
            return
        except Exception:
            continue


def _stage_fig(plt, fig, title):
    """Register a figure and, if a GUI exists, put it on screen immediately.

    Every figure stays open; the blocking plt.show() at the end of run_all()
    holds all of them on screen together.
    """
    _FIGS.append((title, fig))
    k = len(_FIGS) - 1
    try:
        fig.canvas.manager.set_window_title(f"Figure {k+1}  -  {title}")
    except Exception:
        pass
    if GUI_OK:
        _place(fig, k)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fig.show()
                plt.pause(0.001)          # let the toolkit paint it
            except Exception as e:
                print(f"   [figure {k+1} could not be shown: "
                      f"{type(e).__name__}: {e}]")
    print(f"   [figure {k+1:2d} {'open' if GUI_OK else 'built'}: {title}]")


def _fmt(ax, xlabel=None, ylabel=None, title=None, xticks=None, yticks=None,
         legend=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if legend:
        ax.legend(**legend)
    return ax


GUI_OK = None          # set by _mpl(): True if real windows are possible
_MPL_WHY = []          # (backend, reason) for each probe failure


def gui_report():
    """Say plainly whether this Python can open a window, and why not."""
    lines = []
    for mod, backend in (("tkinter", "TkAgg"), ("PyQt5", "Qt5Agg"),
                         ("PySide6", "QtAgg"), ("PyQt6", "QtAgg"),
                         ("wx", "WXAgg")):
        try:
            __import__(mod)
            lines.append(f"      {mod:<10} FOUND   -> backend {backend}")
        except Exception as e:
            lines.append(f"      {mod:<10} missing ({type(e).__name__})")
    return lines


def _mpl(headless=False):
    """Choose a backend and REPORT it. Never fail silently.

    The old version switched to Agg whenever stdout was not a tty, which is
    true in VS Code's integrated terminal -- so every figure was built and
    thrown away with no message. Now:
      * a GUI backend is only accepted after actually creating a test window,
        because matplotlib.use('TkAgg') succeeds even when tkinter is absent
        and only fails later;
      * if no GUI exists, that is printed loudly and the run switches to
        saving PNGs instead of pretending to draw.
    """
    global GUI_OK
    import matplotlib
    if headless:
        matplotlib.use("Agg", force=True)
        GUI_OK = False
    else:
        GUI_OK = False
        _why = []
        for cand in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg", "MacOSX"):
            try:
                matplotlib.use(cand, force=True)
                import matplotlib.pyplot as _probe
                _f = _probe.figure()
                _probe.close(_f)
                GUI_OK = True
                break
            except Exception as e:
                _why.append((cand, f"{type(e).__name__}: {e}"))
                continue
        if not GUI_OK:
            matplotlib.use("Agg", force=True)
            _MPL_WHY.extend(_why)
    import matplotlib.pyplot as plt
    print(f"   [matplotlib backend: {matplotlib.get_backend()}   "
          f"GUI windows: {'YES' if GUI_OK else 'NO'}]")
    if not GUI_OK and not headless:
        # Distinguish the two very different causes, because the fixes differ:
        #   (a) no toolkit installed        -> pip install PyQt5
        #   (b) toolkit present, no display -> you are on a headless/remote box
        toolkit = any("FOUND" in ln for ln in gui_report())
        headless_env = any("headless" in w or "DISPLAY" in w
                           for _, w in _MPL_WHY)
        print("   " + "!" * 70)
        if headless_env and toolkit:
            print("   GUI TOOLKIT IS INSTALLED, BUT THERE IS NO DISPLAY TO "
                  "DRAW ON.")
            print("   You are on a headless or remote session (SSH without "
                  "X forwarding,")
            print("   WSL without WSLg, a container, or a service account).")
            print("   Fix: run on the desktop session, or use ssh -X, or just")
            print("   read the PNGs this run is about to write.")
        elif not toolkit:
            print("   NO GUI TOOLKIT FOUND -- this Python cannot open a "
                  "window.")
            print("   Fix with ONE of:")
            print("      pip install PyQt5        (easiest, no reinstall)")
            print("      reinstall Python from python.org with "
                  "'tcl/tk and IDLE' ticked")
        else:
            print("   A GUI BACKEND COULD NOT BE STARTED.")
        for ln in gui_report():
            print(ln)
        if _MPL_WHY:
            print("   backend probe said:")
            for cand, w in _MPL_WHY[:3]:
                print(f"      {cand:<8} {w[:64]}")
        print("   Every figure will be WRITTEN TO PNG automatically.")
        print("   " + "!" * 70)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                         "grid.alpha": .3, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.autolayout": True})
    return plt


BAND_COLOR = {CLASS_PLUS: "tab:green", CLASS_MINUS: "tab:blue",
              CLASS_FREE: "tab:orange", CLASS_BAD: "tab:red"}


def fig1_problem(plt, A=None, name="size-5 (contention)"):
    """The problem: energy landscape, cut landscape, degeneracy."""
    A = size5_graph() if A is None else A
    best, opt, ndeg = brute_force_maxcut(A)
    S, H = energy_landscape(A)
    cuts = np.array([cut_value(A, s) for s in S])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(H, lw=1, marker='.', ms=4)
    for k in np.flatnonzero(np.isclose(H, H.min())):
        ax[0].plot(k, H[k], 'r*', ms=13)
    _fmt(ax[0], "spin state index", "H(s)",
         f"{name}: Ising energy landscape  [H(s) = H(-s)]")
    ax[1].plot(cuts, lw=1, marker='.', ms=4, color='tab:orange')
    ax[1].axhline(best, color='r', ls='--', lw=.8)
    _fmt(ax[1], "spin state index", "cut",
         f"MaxCut = {best:.0f}, degeneracy = {ndeg}")
    _stage_fig(plt, fig, "problem_landscape")
    return fig


def fig2_dynamics(plt, A=None, Ks=KS_DEFAULT, T=60.0, Ks_bad=1.2):
    """Stage 1: Ks INSIDE the window versus Ks above it.

    An earlier version of this figure contrasted a noiseless "cold" run against
    a noisy annealed one and showed the cold run failing. That contrast was an
    ARTEFACT of running at Ks = 0.6, outside the useful window: at Ks = 0.30
    the size-6 graph (and the 16-node one) is solved 100 % of the time with no
    noise at all. The real lesson is the window itself, so that is what this
    figure now shows.

    The energy panel uses FIXED Ks for both runs on purpose. E is evaluated
    with the current Ks, so while Ks is ramping the potential itself is moving
    and E is NOT expected to fall monotonically -- measured: 37 rising steps,
    all of them inside the ramp window. Monotonicity is a fixed-Ks guarantee.
    """
    A = size6_graph() if A is None else A
    J = adjacency_to_J(A)
    n = A.shape[0]
    best = brute_force_maxcut(A)[0]
    phi0 = _start(0, n)

    dt_ok = stable_dt(A, Ks)
    good = simulate(J, OIMParams(Ks_schedule=linear_ramp(Ks, 3, 12), D=0.0,
                                 dt=dt_ok, T=T), phi0=phi0)
    dt_hi = stable_dt(A, Ks_bad)
    bad = simulate(J, OIMParams(Ks=Ks_bad, D=0.0, dt=dt_hi, T=T), phi0=phi0)

    c_ok = cut_value(A, spins_from_phase(good.phi_final))
    c_hi = cut_value(A, spins_from_phase(bad.phi_final))

    # energy panel: fixed Ks on both, so "only falls" is actually true
    e_ok = simulate(J, OIMParams(Ks=Ks, D=0.0, dt=dt_ok, T=T), phi0=phi0)
    e_hi = simulate(J, OIMParams(Ks=Ks_bad, D=0.0, dt=dt_hi, T=T), phi0=phi0)

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    rg, rb = good.phi_rel(), bad.phi_rel()
    for i in range(n):
        ax[0].plot(good.t, rg[:, i], lw=1, label=f"Osc{i+1}")
        ax[1].plot(bad.t, rb[:, i], lw=1)
    _fmt(ax[0], "time (a.u.)", r"$\phi_i-\phi_1$ (deg)",
         f"$K_s/K_c$ = {Ks:.2f} (in window), annealed  ->  cut {c_ok:.0f}/{best:.0f}",
         yticks=[0, 90, 180, 270, 360], legend=dict(fontsize=6, ncol=2))
    _fmt(ax[1], "time (a.u.)", None,
         f"$K_s/K_c$ = {Ks_bad:.2f} (too high), fixed  ->  cut {c_hi:.0f}/{best:.0f}",
         yticks=[0, 90, 180, 270, 360])
    ax[2].plot(e_ok.t, e_ok.E, lw=1.2, label=f"$K_s/K_c$ = {Ks:.2f}")
    ax[2].plot(e_hi.t, e_hi.E, lw=1.2, label=f"$K_s/K_c$ = {Ks_bad:.2f}")
    _fmt(ax[2], "time (a.u.)", "E",
         "fixed $K_s$: E only falls  (a ramp breaks this)",
         legend=dict(fontsize=7))
    _stage_fig(plt, fig, "dynamics_and_energy")

    # the batch the later figures reuse
    PH = simulate_batch(J, OIMParams(Ks_schedule=linear_ramp(Ks, 3, 12),
                                     D=0.004, dt=dt_ok, T=T, seed=1), B=50)
    return fig, PH


def fig3_waveforms(plt):
    """Stage 2: madam's formula rebuilding the paper's Fig. 4(a), osc2 at 2f."""
    phi = np.radians([0.0, 37.0, 180.0, 180.0, 0.0])
    f = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    t, V = make_waveforms(phi, f=f, n_periods=5, pts_per_period=2048)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for i in range(5):
        ax.plot(t, V[i] + i * 1.0, lw=1.1, label=f"Osc {i+1}"
                + ("   <- 2f" if f[i] == 2 else ""))
    ax.set_yticks([i * 1.0 + 0.4 for i in range(5)])
    ax.set_yticklabels([f"Osc{i+1}" for i in range(5)])
    _fmt(ax, "time (a.u.)", "V",
         r"$V_i(t)=\frac{V_{DD}}{2}\,\mathrm{sgn}(\sin(\omega_i t+\phi_i))+\frac{V_{DD}}{2}$",
         legend=dict(fontsize=7, ncol=5, loc="upper center"))
    ax.set_ylim(-0.3, 5.6)
    _stage_fig(plt, fig, "emulated_waveforms")
    return fig


def fig4_unreadable(plt, M_th=20.0):
    """Stage 4: THE figure. Shift the square wave and watch the readout fail."""
    r = sweep_phase_shift(n_pts=721, duty=0.5, n_periods=100,
                          pts_per_period=1024, M_th=M_th)
    r47 = sweep_phase_shift(n_pts=721, duty=0.472, duty_b=0.472, n_periods=100,
                            pts_per_period=1024, M_th=M_th)
    bands = bands_from_classes(r["delta"], r["cls"])
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6))
    for c in (CLASS_PLUS, CLASS_MINUS, CLASS_FREE, CLASS_BAD):
        for lo, hi in bands[c]:
            ax[0].axvspan(lo, hi, color=BAND_COLOR[c], alpha=.18, lw=0)
    ax[0].plot(r["delta"], r["C"], lw=1.6, color='k', label="duty 0.500")
    ax[0].plot(r47["delta"], r47["C"], lw=1.2, color='tab:purple', ls='--',
               label="duty 0.472 (measured detector)")
    for y in (M_th, 50 - M_th, 50 + M_th, 100 - M_th):
        ax[0].axhline(y, color='k', lw=.5, ls=':')
    _fmt(ax[0], r"phase shift $\Delta\phi$ (deg)", "C (%)",
         f"XOR readout, $M_{{th}}$ = {M_th:.0f} %: "
         f"UNREADABLE = {r['bad_width_measured']:.0f} deg of 360",
         xticks=[0, 90, 180, 270, 360], legend=dict(fontsize=7, loc="lower center"))
    uv = unreadable_vs_Mth(np.arange(2.0, 41.0, 2.0), duty=0.5, n_pts=721,
                           n_periods=60, pts_per_period=512)
    ax[1].plot(uv["M_th"], uv["bad_width"], 'o-', color='tab:red', ms=4,
               label="UNREADABLE")
    ax[1].plot(uv["M_th"], uv["free_width"], '^-', color='tab:orange', ms=4,
               label="free (spin 0)")
    ax[1].axvline(25.0, color='k', ls='--', lw=.9)
    ax[1].annotate("closes at\n$M_{th}$ = 25 %", xy=(25, 10), xytext=(28, 120),
                   fontsize=7, arrowprops=dict(arrowstyle="->", lw=.7))
    _fmt(ax[1], r"$M_{th}$ (%)", "width (deg)",
         "headroom design curve", legend=dict(fontsize=7))
    _stage_fig(plt, fig, "unreadable_band")
    return fig, r


def fig5_detectors(plt, M_th=20.0):
    """Stage 3: all four detectors on the same sweep."""
    r = sweep_phase_shift(n_pts=361, duty=0.5, n_periods=100,
                          pts_per_period=1024, M_th=M_th)
    r2 = sweep_phase_shift(n_pts=361, duty=0.5, n_periods=100,
                           pts_per_period=1024, M_th=M_th, f_i=2.0)
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax[0].plot(r["delta"], r["C"], lw=1.5, label="XOR + average")
    ax[0].plot(r["delta"], r["rsff"], lw=1.5, label="RS flip-flop")
    ax[0].plot(r["delta"], 100 * r["dff_mean"], lw=1.2, ls='--',
               label="D-FF sample x100")
    _fmt(ax[0], r"$\Delta\phi$ (deg)", "output (%)", "detector characteristics",
         xticks=[0, 90, 180, 270, 360], legend=dict(fontsize=7))
    ax[1].plot(r["delta"], r["edge_ratio"], lw=1.6, label=r"$f_i=f_0$")
    ax[1].plot(r2["delta"], r2["edge_ratio"], lw=1.6, label=r"$f_i=2f_0$")
    ax[1].set_ylim(0.5, 2.5)
    _fmt(ax[1], r"$\Delta\phi$ (deg)", r"edge count $f_i/f_{base}$",
         "only the edge counter sees 2f", xticks=[0, 90, 180, 270, 360],
         legend=dict(fontsize=7))
    ax[2].plot(r["delta"], r["C"], lw=1.6, label=r"$f_i=f_0$")
    ax[2].plot(r2["delta"], r2["C"], lw=1.6, label=r"$f_i=2f_0$ (flat at 50 %)")
    ax[2].axhline(50, color='k', ls=':', lw=.8)
    _fmt(ax[2], r"$\Delta\phi$ (deg)", "C (%)",
         "XOR cannot: 2f reads 50 % at every phase",
         xticks=[0, 90, 180, 270, 360], legend=dict(fontsize=7))
    _stage_fig(plt, fig, "four_detectors")
    return fig


def fig6_duty_cliff(plt, PH=None):
    """Stage 4: spin error vs duty for three M_th -- the cliff."""
    if PH is None:
        A = size6_graph()
        dt = stable_dt(A, KS_DEFAULT)
        PH = simulate_batch(adjacency_to_J(A),
                            OIMParams(Ks_schedule=linear_ramp(KS_DEFAULT, 3, 12),
                                      D=0.004, dt=dt, T=60, seed=1), B=30)
    duties = np.linspace(0.30, 0.50, 21)
    se = spin_error_vs_duty(PH, duties, M_th=(20.0, 10.0, 5.0),
                            n_periods=60, pts_per_period=256)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for jm, mm in enumerate(se["M_th"]):
        ln, = ax[0].plot(se["duty"], se["error"][jm], lw=1.6,
                         label=f"$M_{{th}}$ = {mm:.0f}")
        ax[0].axvline(se["cliff"][jm], color=ln.get_color(), ls=':', lw=.9)
        ax[1].plot(se["duty"], se["unresolved"][jm], lw=1.6,
                   label=f"$M_{{th}}$ = {mm:.0f}")
    for a_ in ax:
        a_.axvline(0.472, color='k', ls='--', lw=.9)
    _fmt(ax[0], "detector duty cycle d", "spin error rate (%)",
         "dotted = predicted cliff (100-M)/200, dashed = BUET detector 0.472",
         legend=dict(fontsize=7))
    _fmt(ax[1], "detector duty cycle d", "unresolved spins (%)",
         "every error is an UNRESOLVED spin, not a flip", legend=dict(fontsize=7))
    _stage_fig(plt, fig, "duty_cliff")
    return fig, se


def fig7_ks_ratio(plt, A=None):
    A = size6_graph() if A is None else A
    kr = ks_ratio_sweep(A, ratios=np.linspace(0.0, 1.6, 13), B=100)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(kr["ratio"], kr["p_opt"], 's-', color='tab:orange', lw=1.6, ms=5,
            label="P(hit optimum)")
    ax.plot(kr["ratio"], kr["binarised"], '^-', color='tab:green', lw=1.6, ms=5,
            label="fraction of spins binarised")
    ax.axvspan(*KS_WINDOW, color='tab:blue', alpha=.10, lw=0)
    ax.set_ylim(0, 1.05)
    _fmt(ax, r"$K_s / K_c$", "fraction",
         f"{kr['B']} runs/point; shaded = the useful window "
         f"{KS_WINDOW[0]:.2f} - {KS_WINDOW[1]:.2f}",
         legend=dict(fontsize=8, loc="lower left"))
    _stage_fig(plt, fig, "ks_ratio_sweep")
    return fig, kr


def fig8_detuning(plt, Ks=KS_DEFAULT):
    ds = detuning_sweep(Ks=Ks, ratios=np.linspace(0, 2, 41), T=400.0)
    th = ds["theory"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(ds["ratio"], np.abs(ds["phi_star_deg"]), 'o', ms=4,
               label="simulated")
    ax[0].plot(ds["ratio"], np.abs(th["phi_star_deg"]), 'k-', lw=1,
               label=r"$\frac{1}{2}\arcsin(\Delta\omega/K_s)$")
    ax[0].axvline(1.0, color='r', ls=':', lw=.9)
    _fmt(ax[0], r"$\Delta\omega / K_s$", r"$\phi^*$ (deg)",
         "locked branch: steady phase offset", legend=dict(fontsize=7))
    ax[1].plot(ds["ratio"], ds["drift_rate"], 'o', ms=4, label="simulated")
    ax[1].plot(ds["ratio"], th["drift_rate"], 'k-', lw=1,
               label=r"$\sqrt{\Delta\omega^2-K_s^2}$")
    ax[1].axvline(1.0, color='r', ls=':', lw=.9)
    _fmt(ax[1], r"$\Delta\omega / K_s$", r"$\langle d\phi/dt\rangle$",
         "loss of lock -> frequency multiplication", legend=dict(fontsize=7))
    _stage_fig(plt, fig, "detuning_lock")
    return fig, ds


def fig9_frequency_sweep(plt, A=None):
    """Stage 5: frequency sweeping recovers ALL degenerate optima (K4)."""
    A = k4_graph() if A is None else A
    fs = frequency_sweep_solutions(A, tunes=np.linspace(-KS_TUNE_MAX, KS_TUNE_MAX, 29),
                                   Ks=KS_DEFAULT, T=80.0, seed=3)
    labels, first = [], {}
    for k, sp in enumerate(fs["spins"]):
        if fs["cut"][k] >= fs["best"]:
            first.setdefault(sp, len(first))
            labels.append(first[sp])
        else:
            labels.append(-1)
    labels = np.array(labels)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(fs["tune"], fs["cut"], 'o-', ms=4, lw=1.2)
    ax[0].axhline(fs["best"], color='r', ls='--', lw=.8)
    _fmt(ax[0], "frequency tuning (detuning scale)", "cut",
         f"MaxCut = {fs['best']:.0f}, degeneracy = {fs['degeneracy']}")
    for lab in sorted(set(labels[labels >= 0])):
        m_ = labels == lab
        key = [k for k, v in first.items() if v == lab][0]
        p, q = partition_of(np.array(key, float))
        ax[1].plot(fs["tune"][m_], np.full(m_.sum(), lab), 'o', ms=6,
                   label=f"{p} | {q}")
    _fmt(ax[1], "frequency tuning (detuning scale)", "optimum index",
         f"distinct optima recovered: {len(first)} of {fs['degeneracy']}",
         legend=dict(fontsize=7, loc="best"))
    _stage_fig(plt, fig, "frequency_sweep_degeneracy")
    return fig, fs


def fig10_settling(plt, A=None, Ks=KS_DEFAULT):
    A = size6_graph() if A is None else A
    sc = settling_constant(A, Ks=Ks, seeds=32)
    typ = sc["seeds"][int(np.argmin(np.abs(sc["t_settle"] - sc["median"])))]
    sw = settling_vs_omega(A, Ks=Ks, seed=typ)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for (tt, dv), w in zip(sw["curves"], sw["w"]):
        ax[0].semilogy(tt, np.maximum(dv, 1e-12), lw=1, label=f"w={w:.2f}")
    ax[0].axhline(sw["tol"], color='k', lw=.8, ls='--')
    ax[0].set_xlim(0, 25)
    ax[0].set_ylim(1e-5, 10)
    _fmt(ax[0], "time (a.u.)", r"max$|\phi-\phi_\infty|$ (rad)",
         r"madam's max$|\phi-\phi|$ settling metric", legend=dict(fontsize=6))
    ax[1].semilogx(sw["w"], sw["product"], 'o-', lw=1.2, ms=5)
    ax[1].axhline(sc["median"], color='r', ls='--', lw=.8,
                  label=f"median {sc['median']:.3f}")
    ax[1].set_ylim(0, 2 * np.nanmax(sw["product"]))
    _fmt(ax[1], r"gain $w$", r"$w \cdot t_{settle}$",
         "flat: raising w only rescales time", legend=dict(fontsize=7))
    _stage_fig(plt, fig, "settling_constant")
    return fig, sc, sw


def fig11_fixed_point(plt, A=None, Ks=KS_DEFAULT):
    """Stage 6: the RTL twin against the float model."""
    A = size6_graph() if A is None else A
    n = A.shape[0]
    cfg, dt = make_fixed_cfg(A, Ks=Ks, D=0.0, seed=1)
    cfg.HOLD = 10 ** 9
    p0 = _start(0, n)
    p0q = np.round(p0 * LSB_PER_RAD).astype(np.int64) & PHASE_MASK
    steps = int(round(60.0 / dt))
    fx = fixed_run(cfg, p0q, max_steps=steps)
    fl = simulate(adjacency_to_J(A), OIMParams(Ks=Ks, dt=dt, T=steps * dt),
                  phi0=p0)
    tq = np.arange(fx["traj"].shape[0]) * dt
    rq = np.degrees(np.mod(fixed_to_rad(fx["traj"])
                           - fixed_to_rad(fx["traj"])[:, [0]], TWO_PI))
    rf = fl.phi_rel()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for i in range(n):
        ax[0].plot(tq, rq[:, i], lw=1.0)
        ax[0].plot(fl.t, rf[:, i], lw=0.8, ls='--', color='k', alpha=.5)
    _fmt(ax[0], "time (a.u.)", r"$\phi_i-\phi_1$ (deg)",
         "solid = Q16 fixed point (RTL twin), dashed = float",
         yticks=[0, 90, 180, 270, 360])
    err = np.abs(((rq[:len(rf)] - rf + 180) % 360) - 180).max(axis=1)
    ax[1].semilogy(fl.t, np.maximum(err, 1e-6), lw=1.2)
    _fmt(ax[1], "time (a.u.)", "max phase difference (deg)",
         "quantisation error saturates; spins agree throughout")
    _stage_fig(plt, fig, "fixed_point_twin")
    return fig


def fig12_kim(plt):
    """Stage 7: the Chris Kim readout, and where each scheme wins."""
    tab = kim_code_table()
    codes = sorted(tab)
    ph = np.linspace(0, 359.9, 1441)
    dec = kim_decode(kim_phase_id(ph))
    Ap = np.zeros((10, 10))
    for i in range(9):
        Ap[i, i + 1] = Ap[i + 1, i] = 1.0
    rp = compare_readouts_under_drift(A=Ap, amplitudes=np.arange(0, 181, 20.0),
                                      reps=6)
    Ak = kings_graph(3, 4)
    rk = compare_readouts_under_drift(A=Ak, amplitudes=np.arange(0, 181, 20.0),
                                      reps=6)
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax[0].step(ph, dec, where='mid', lw=1.5, label="decoded")
    ax[0].plot(ph, ph, 'k:', lw=1, label="true")
    for lo, hi, c in tab.values():
        ax[0].axvline(np.mod(hi, 360.0), color='gray', lw=.4)
    _fmt(ax[0], "true phase (deg)", "decoded phase (deg)",
         f"Kim 4-bit phase ID: {len(codes)} bins, 45 deg mean",
         xticks=[0, 90, 180, 270, 360], legend=dict(fontsize=7))
    for a_, r_, ttl in ((ax[1], rp, "path-10 (diameter 9)"),
                        (ax[2], rk, "King's 3x4 (diameter 3)")):
        a_.plot(r_["amplitude"], r_["algorithm1"] / r_["best"], 'o-', lw=1.5,
                ms=4, label="Algorithm 1 (one fixed base)")
        a_.plot(r_["amplitude"], r_["kim"] / r_["best"], 's-', lw=1.5, ms=4,
                label="Kim (confidence-ranked neighbours)")
        a_.set_ylim(0.6, 1.05)
        _fmt(a_, "spatial phase drift (deg)", "normalised cut", ttl,
             legend=dict(fontsize=6.5, loc="lower left"))
    _stage_fig(plt, fig, "kim_readout_comparison")
    return fig, rp, rk


def figures():
    return list(_FIGS)


def save_figures(outdir="."):
    paths = []
    for k, (title, fig) in enumerate(_FIGS, 1):
        safe = re.sub(r"\W+", "_", title)
        fn = os.path.join(outdir, f"fig{k:02d}_{safe}.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        paths.append(fn)
        print(f"   wrote {fn}")
    return paths


def run_all(save=False, outdir=".", show=True, headless=False):
    plt = _mpl(headless=headless)
    _FIGS.clear()
    print("\n" + "=" * 78)
    print(" STAGE 0  THE PROBLEM")
    print("=" * 78)
    for nm, A in [("size-6 (Fig. 1a/3a)", size6_graph()),
                  ("size-5 (Fig. 3e/4)", size5_graph()),
                  ("K4     (Fig. 6)", k4_graph())]:
        describe_graph(A, nm)
    fig1_problem(plt)

    print("\n" + "=" * 78)
    print(" STAGE 1  SYSTEM SIMULATION (Ks, Kc)")
    print("=" * 78)
    A = size6_graph()
    dt = stable_dt(A, KS_DEFAULT)
    print(f"  deg_max = {A.sum(1).max():.0f}, Ks/Kc = {KS_DEFAULT}  ->  L = "
          f"{lipschitz_L(A,KS_DEFAULT):.2f}, 2/L = "
          f"{2/lipschitz_L(A,KS_DEFAULT):.4f}, "
          f"dt = {dt:.5f} (10x margin)")
    _, PH = fig2_dynamics(plt)

    print("\n" + "=" * 78)
    print(" STAGE 2  WAVEFORM EMULATION")
    print("=" * 78)
    print("  V_i(t) = VDD/2 * sgn(sin(w_i t + phi_i)) + VDD/2,  osc2 at 2 f")
    fig3_waveforms(plt)

    print("\n" + "=" * 78)
    print(" STAGE 3/4  READOUT AND THE UNREADABLE BAND")
    print("=" * 78)
    _, rr = fig4_unreadable(plt)
    b = bands_from_classes(rr["delta"], rr["cls"])
    for c, nm in [(CLASS_PLUS, "+1"), (CLASS_FREE, "free (0)"),
                  (CLASS_MINUS, "-1"), (CLASS_BAD, "UNREADABLE")]:
        print(f"   {nm:11s} " + ", ".join(f"[{a:.0f},{z:.0f})" for a, z in b[c]))
    print(f"   UNREADABLE total = {rr['bad_width_measured']:.0f} deg "
          f"({100*rr['bad_width_measured']/360:.0f} % of the circle)")
    fig5_detectors(plt)
    _, se = fig6_duty_cliff(plt, PH[:30])
    for jm, mm in enumerate(se["M_th"]):
        print(f"   M_th = {mm:4.1f}: cliff at d = {se['cliff'][jm]:.3f}, "
              f"error {se['error'][jm][-1]:5.1f} % at d=0.500 -> "
              f"{se['error'][jm][0]:5.1f} % at d=0.300")
    fig7_ks_ratio(plt)

    print("\n" + "=" * 78)
    print(" STAGE 5  DEGENERACY, 2f, FREQUENCY SWEEPING")
    print("=" * 78)
    phi = np.radians([0.0, 37.0, 180.0, 180.0, 0.0])
    fvec = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    res = classify_network(phi, f=fvec, base=0, M_th=20.0)
    for i, r_ in enumerate(res["rows"]):
        print(f"   osc{i+1}: C={r_['C']:6.2f} %  f_i/f_b={r_['edge_ratio']:.3f}  "
              f"-> {r_['state']}  spin {r_['spin']:+.0f}")
    print(f"   S = {res['S'].astype(int)}   free vertices "
          f"{spins_to_partition(res['S'])[2]}")
    for s in expand_free(res["S"]):
        p, q = partition_of(s)
        print(f"      {p} | {q}   cut {cut_value(size5_graph(), s):.0f}")
    fig8_detuning(plt)
    _, fs = fig9_frequency_sweep(plt)
    print(f"   K4: recovered {len(fs['unique_optima'])} of {fs['degeneracy']} "
          f"degenerate optima by frequency tuning")
    fig10_settling(plt)

    print("\n" + "=" * 78)
    print(" STAGE 6  FIXED-POINT TWIN OF oim_pynq.v")
    print("=" * 78)
    cfg, dtq = make_fixed_cfg(size6_graph(), Ks=KS_DEFAULT, D=0.0, seed=1)
    print(f"   DTQ = {cfg.DTQ}   KC = {cfg.KC}   KS = {cfg.KS}   "
          f"NAMP = {cfg.NAMP}   (dt = {dtq:.5f})")
    fig11_fixed_point(plt)

    print("\n" + "=" * 78)
    print(" STAGE 7  THE CHRIS KIM READOUT  (Nature Electronics 5, 310, 2022)")
    print("=" * 78)
    tabk = kim_code_table()
    print("   4-bit phase-ID codes: "
          + ", ".join(format(c, "04b") for c in sorted(tabk))
          + f"   ({len(tabk)} bins, {360/len(tabk):.0f} deg mean)")
    Ak = kings_graph(41, 48)
    print(f"   King's graph 41x48 -> n={Ak.shape[0]}, edges={int(Ak.sum()/2)}, "
          f"max degree={int(Ak.sum(1).max())}")
    _, rp, rk = fig12_kim(plt)
    print(f"   path-10   (diameter 9): Algorithm 1 "
          f"{rp['algorithm1'].mean()/rp['best']:.3f}  vs  Kim "
          f"{rp['kim'].mean()/rp['best']:.3f}")
    print(f"   King's 3x4 (diameter 3): Algorithm 1 "
          f"{rk['algorithm1'].mean()/rk['best']:.3f}  vs  Kim "
          f"{rk['kim'].mean()/rk['best']:.3f}")
    ab = annealing_benefit(B=200)
    for k in ("Ks fixed", "Ks annealed"):
        print(f"   {k:12s} P(optimum) = {ab[k]['p_opt']:.3f}, "
              f"mean cut {ab[k]['mean_cut']:.2f}")

    write_sin_lut_mem(os.path.join(outdir, "sin_lut.mem"))
    print(f"   wrote {os.path.join(outdir,'sin_lut.mem')} (1024 x 16-bit hex)")

    # If there is no GUI the figures would otherwise vanish, so write them
    # out whether or not --save was given. A run that produces nothing the
    # user can look at is a failed run.
    paths = []
    if save or not GUI_OK:
        if not save:
            print("\n   [no GUI -- saving the figures instead so the run is "
                  "not wasted]")
        paths = save_figures(outdir)

    trace_summary()

    print("=" * 78)
    if GUI_OK and show:
        print(f" {len(_FIGS)} FIGURES ARE OPEN SIMULTANEOUSLY "
              f"(Figure 1 ... Figure {len(_FIGS)}).")
        print(" Arrange them side by side. Closing the last window ends the run.")
        print("=" * 78)
        plt.show()                     # interactive mode is off -> this blocks
    else:
        print(f" {len(_FIGS)} figures written to {os.path.abspath(outdir)}")
        for q in paths:
            print("   " + os.path.abspath(q))
        if not GUI_OK:
            print("\n Install a GUI toolkit to see them live:  pip install PyQt5")
        print("=" * 78)
        if not GUI_OK and paths and "--open" in sys.argv:
            _open_folder(outdir)
    return figures()


def _open_folder(path):
    """Open the output folder in the OS file browser (--open)."""
    import subprocess
    path = os.path.abspath(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                                   # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        print(f"   [could not open {path}: {e}]")


def _argval(argv, flag, cast, default):
    return cast(argv[argv.index(flag) + 1]) if flag in argv else default


def main(argv=None):
    """
    python3 oim_sim.py                 run everything, trace on, all figures open
    python3 oim_sim.py --no-trace      same, silent
    python3 oim_sim.py --trace-depth 5 --trace-cap 10    deeper trace
    python3 oim_sim.py --save          also write figNN_*.png
    python3 oim_sim.py --no-show       headless (Agg), no windows
    python3 oim_sim.py --checks        self-tests only
    python3 oim_sim.py --stage 4       one stage
    python3 oim_sim.py --lut           write sin_lut.mem and exit
    """
    argv = sys.argv[1:] if argv is None else argv
    outdir = _argval(argv, "--outdir", str, ".")
    os.makedirs(outdir, exist_ok=True)

    # tracing is ON for normal runs, but OFF for --checks unless asked for:
    # a self-test sweep generates hundreds of thousands of calls and the trace
    # would bury the results it is supposed to illuminate.
    trace_on = ("--no-trace" not in argv
                and not ("--checks" in argv and "--trace" not in argv))
    install_tracer(on=trace_on,
                   max_depth=_argval(argv, "--trace-depth", int, 3),
                   cap=_argval(argv, "--trace-cap", int, 2))

    if "--checks" in argv:
        return 0 if run_checks() else 1
    if "--lut" in argv:
        print(write_sin_lut_mem(os.path.join(outdir, "sin_lut.mem")))
        return 0
    if "--gui" in argv:
        print(" GUI toolkit check for", sys.executable)
        for ln in gui_report():
            print(ln)
        _mpl(headless=False)
        return 0

    headless = ("--no-show" in argv) or ("--headless" in argv)

    if "--stage" in argv:
        plt = _mpl(headless=headless)
        k = int(argv[argv.index("--stage") + 1])
        {0: lambda: fig1_problem(plt), 1: lambda: fig2_dynamics(plt),
         2: lambda: fig3_waveforms(plt), 3: lambda: fig5_detectors(plt),
         4: lambda: fig4_unreadable(plt), 5: lambda: fig9_frequency_sweep(plt),
         6: lambda: fig11_fixed_point(plt), 7: lambda: fig12_kim(plt)}[k]()
        if ("--save" in argv) or not GUI_OK:
            save_figures(outdir)
        trace_summary()
        if GUI_OK and not headless:
            plt.show()
        return 0

    run_all(save=("--save" in argv), outdir=outdir,
            show=not headless, headless=headless)
    return 0




# =============================================================================
#  STAGE 7 -- THE CHRIS KIM READOUT  (Moy, Ahmed, Chiu, Moy, Sapatnekar & Kim,
#  "A 1,968-node coupled ring oscillator circuit for combinatorial optimization
#  problem solving", Nature Electronics 5, 310-317, 2022)
#
#  Their chip is the state of the art this thesis is measured against, and it
#  reads its oscillators out in a completely different way from the XOR chain.
#  Two ideas are worth taking, and one measured result of theirs contradicts a
#  rule we wrote, so it gets tested rather than believed.
#
#  IDEA 1 -- THE 4-BIT PHASE ID (their Fig. 3e / 4a).
#  Each unit tile samples its own ROSC output tapped after 0, 2, 4 and 6
#  inverter stages, all latched on ONE globally shared clock edge. In a
#  9-stage ROSC a 2-stage delay is T/9 = 40 deg, so the four taps span 120 deg
#  and the result is a thermometer code with exactly eight legal values:
#      0000 0001 0011 0111 1000 1100 1110 1111
#  That is a pseudo-analogue phase measurement, roughly 45 deg resolution, and
#  because every oscillator is latched by the same edge there is no
#  measurement-induced phase skew between them.
#
#  IDEA 2 -- CONFIDENCE-RANKED, NEAREST-NEIGHBOUR POST-PROCESSING (their Fig. 4c).
#  Their measured phase maps show ABSOLUTE phase drifting across the array
#  while NEAREST-NEIGHBOUR differences stay correct. So they never compare
#  distant oscillators. They rank each edge by how close its average phase
#  difference sits to 0 or 180 deg, resolve spins from the most confident edges
#  first, and simply stop once every spin is assigned -- the least confident
#  edges are never used.
#
#  This matters here: the paper's Algorithm 1 picks ONE base oscillator and
#  compares everything to it. Under a spatially varying drift that is exactly
#  the wrong thing to do, and Section 7.4 measures how wrong.
# =============================================================================

#  Legal thermometer codes -> the phase window each one reports, in degrees.
#  Derived, not tabulated: tap k samples at phase x + k*tap_deg, and a 50 %
#  duty square wave is high for x in [0,180).
KIM_TAPS = 4
KIM_TAP_DEG = 40.0


def kim_phase_id(phase_deg, tap_deg=KIM_TAP_DEG, taps=KIM_TAPS, duty=0.5):
    """The 4-bit thermometer code one tile would latch for a given phase.

    Bit k is the oscillator's level at (phase + k*tap_deg). Returned MSB-first
    as S1..S4 so it matches the paper's printed code table.
    """
    x = np.asarray(phase_deg, float)
    bits = []
    for k in range(taps):
        frac = np.mod(x + k * tap_deg, 360.0) / 360.0
        bits.append((frac < duty).astype(int))
    out = np.zeros(np.shape(x), dtype=int)
    for k, b in enumerate(bits):
        out = out | (b << (taps - 1 - k))
    return out if out.shape else int(out)


def deg_to_phase_word(deg):
    """Degrees -> the 16-bit phase word the RTL actually holds."""
    return int(round(np.asarray(deg, float) * (1 << PW) / 360.0)) & PHASE_MASK \
        if np.isscalar(deg) else \
        (np.round(np.asarray(deg, float) * (1 << PW) / 360.0).astype(np.int64)
         & PHASE_MASK)


KIM_PHTAP_Q = 0x1C71      # 7281 LSB = 39.996 deg, the RTL default
KIM_DUTY_Q = 0x8000       # 50 % duty threshold


def kim_phase_id_fixed(phase_q, phtap_q=KIM_PHTAP_Q, duty_q=KIM_DUTY_Q,
                       taps=KIM_TAPS):
    """INTEGER-domain twin of the RTL sampler. Use this to compare against RTL.

    The float kim_phase_id() and this differ on exactly one thing: where a tap
    lands within a couple of LSB of the duty threshold. At phase 100 deg with
    40 deg taps, tap 2 sits 2 LSB (0.011 deg) below 32768 in the hardware and
    exactly ON 180 deg in the float model, so the two disagree by one code.
    That is a bin boundary, not a defect -- but a twin that is only almost
    bit-exact is not a twin, so this version exists and is what the RTL is
    checked against.
    """
    pq = np.asarray(phase_q, np.int64) & PHASE_MASK
    out = np.zeros(pq.shape, dtype=np.int64)
    for k in range(taps):
        b = (((pq + k * phtap_q) & PHASE_MASK) < duty_q).astype(np.int64)
        out = out | (b << (taps - 1 - k))
    return out if out.shape else int(out)


def kim_code_table(tap_deg=KIM_TAP_DEG, taps=KIM_TAPS, duty=0.5, n=36000):
    """code -> (lo, hi, centre) in degrees, measured by sweeping the phase."""
    ph = np.linspace(0, 360, n, endpoint=False)
    ids = kim_phase_id(ph, tap_deg, taps, duty)
    tab = {}
    for c in np.unique(ids):
        sel = ph[ids == c]
        # the 1111 window straddles 0, so unwrap before averaging
        d = np.diff(sel)
        if d.size and d.max() > 2 * (360.0 / n):
            cut = int(np.argmax(d)) + 1
            sel = np.concatenate([sel[cut:] - 360.0, sel[:cut]])
        tab[int(c)] = (float(sel.min()), float(sel.max()),
                       float(np.mod((sel.min() + sel.max()) / 2, 360.0)))
    return tab


_KIM_TABLE = None


def kim_decode(code, tap_deg=KIM_TAP_DEG, taps=KIM_TAPS, duty=0.5):
    """Thermometer code -> the centre of the phase window it represents."""
    global _KIM_TABLE
    if _KIM_TABLE is None:
        _KIM_TABLE = kim_code_table(tap_deg, taps, duty)
    c = np.asarray(code, int)
    flat = np.array([_KIM_TABLE.get(int(v), (np.nan,) * 3)[2] for v in c.ravel()])
    return flat.reshape(c.shape) if c.shape else float(flat[0])


def kim_sample_network(phi, tap_deg=KIM_TAP_DEG, taps=KIM_TAPS, duty=0.5,
                       drift_deg=None, jitter_deg=0.0, seed=0):
    """Latch every oscillator on one clock edge -> codes and decoded phases.

    drift_deg : per-oscillator ABSOLUTE phase offset, the effect their measured
                phase maps show. It is added before sampling, exactly as a real
                array would experience it.
    """
    phi = np.degrees(np.asarray(phi, float)) if np.max(np.abs(phi)) <= TWO_PI \
        else np.asarray(phi, float)
    n = phi.size
    if drift_deg is not None:
        phi = phi + np.asarray(drift_deg, float)
    if jitter_deg:
        phi = phi + np.random.default_rng(seed).normal(0, jitter_deg, n)
    codes = kim_phase_id(phi, tap_deg, taps, duty)
    return dict(code=codes, phase_deg=kim_decode(codes, tap_deg, taps, duty),
                true_deg=np.mod(phi, 360.0))


def _fold180(d):
    """Distance of a phase difference from the nearer of 0 or 180 degrees."""
    d = np.mod(np.asarray(d, float), 360.0)
    d = np.where(d > 180.0, 360.0 - d, d)
    return d


def kim_postprocess(A, phase_deg, samples=None):
    """Confidence-ranked nearest-neighbour spin extraction (their Fig. 4c).

    Only EDGES of the graph are ever used, never a distant reference, which is
    what makes it immune to a drift that varies across the array.

    confidence = how close the edge's phase difference sits to 0 or 180 deg.
    Edges are consumed most-confident-first; the graph is grown outward from
    the single most confident edge and the rest are discarded.
    """
    A = np.asarray(A, float)
    n = A.shape[0]
    if samples is None:
        samples = np.atleast_2d(np.asarray(phase_deg, float))
    else:
        samples = np.atleast_2d(np.asarray(samples, float))

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                continue
            d = np.mod(samples[:, i] - samples[:, j], 360.0)
            # circular mean of the differences across repeated samples
            ang = np.radians(d)
            mean = np.degrees(np.arctan2(np.sin(ang).mean(),
                                         np.cos(ang).mean())) % 360.0
            folded = _fold180(mean)
            conf = abs(90.0 - folded)          # 90 deg = useless, 0/180 = certain
            same = folded < 90.0               # in phase -> same spin
            edges.append((conf, i, j, same, mean))
    edges.sort(key=lambda e: -e[0])

    S = np.zeros(n)
    used = []
    if not edges:
        return dict(S=S, used_edges=used, n_edges=0)
    # seed on the most confident edge
    c, i, j, same, mean = edges[0]
    S[i] = 1.0
    S[j] = 1.0 if same else -1.0
    used.append((i, j, c))
    changed = True
    while changed and np.any(S == 0):
        changed = False
        for c, i, j, same, mean in edges:
            if (S[i] != 0) == (S[j] != 0):
                continue                        # both known or both unknown
            if S[i] != 0:
                S[j] = S[i] if same else -S[i]
            else:
                S[i] = S[j] if same else -S[j]
            used.append((i, j, c))
            changed = True
    return dict(S=S, used_edges=used, n_edges=len(edges),
                n_unresolved=int(np.sum(S == 0)))


def kings_graph(rows, cols):
    """Kim's array topology: every node couples to 2 vertical, 2 horizontal and
    4 diagonal neighbours -- the moves of a chess king. Their chip is 41 x 48."""
    n = rows * cols
    A = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            a = r * cols + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        A[a, rr * cols + cc] = 1.0
    return _sym(A)


def drift_profile(n, kind="linear", amplitude=60.0, seed=0):
    """Spatially varying absolute-phase drift, of the kind their phase maps show.

    Their uniform problem drifts monotonically across the array; their
    chequerboard problem spirals. Both leave nearest-neighbour differences
    essentially correct while the absolute phase wanders.
    """
    x = np.arange(n) / max(n - 1, 1)
    if kind == "linear":
        return amplitude * (x - 0.5) * 2.0
    if kind == "spiral":
        return amplitude * np.sin(2 * np.pi * 1.5 * x)
    if kind == "random":
        return np.random.default_rng(seed).normal(0, amplitude / 2, n)
    raise ValueError(kind)


def compare_readouts_under_drift(A=None, amplitudes=None, kind="linear",
                                 M_th=20.0, reps=8, seed=0):
    """THE comparison: one fixed base (paper's Algorithm 1) versus Kim's
    confidence-ranked nearest-neighbour scheme, as spatial drift grows."""
    A = size6_graph() if A is None else A
    n = A.shape[0]
    best = brute_force_maxcut(A)[0]
    amplitudes = (np.arange(0, 121, 10.0) if amplitudes is None
                  else np.asarray(amplitudes, float))
    s_true = brute_force_maxcut(A)[1][0]
    phi_true = np.where(s_true > 0, 0.0, 180.0)
    a1_cut, kim_cut = [], []
    for amp in amplitudes:
        c1, c2 = [], []
        for r in range(reps):
            dr = drift_profile(n, kind, amp, seed=seed + r)
            ph = np.radians(phi_true + dr)
            # --- paper Algorithm 1: XOR against ONE base ---
            C = C_matrix_all_bases(ph, duty=0.5, n_periods=60,
                                   pts_per_period=256)
            S1 = algorithm1(lambda b, i: C[b][i], n, M_th=M_th)["S"]
            c1.append(max(cut_value(A, s) for s in expand_free(S1))
                      if np.any(S1 == 0) else cut_value(A, S1))
            # --- Kim: sampled phase IDs + confidence-ranked neighbours ---
            ks = kim_sample_network(phi_true + dr)
            S2 = kim_postprocess(A, ks["phase_deg"],
                                 samples=ks["phase_deg"][None, :])["S"]
            c2.append(cut_value(A, S2) if np.all(S2 != 0)
                      else max(cut_value(A, s) for s in expand_free(S2)))
        a1_cut.append(np.mean(c1))
        kim_cut.append(np.mean(c2))
    return dict(amplitude=amplitudes, algorithm1=np.array(a1_cut),
                kim=np.array(kim_cut), best=best, kind=kind)


def annealing_benefit(A=None, Ks=KS_DEFAULT, B=200, T=60.0, D=0.004, seed=5):
    """Kim measured that EXTERNAL ANNEALING of the coupling bought them almost
    nothing (about 1.0 % average, 0.04 % best) and suggested the annealing
    already happening inside the chip may be enough.

    Our Rule 2 says the opposite about annealing Ks. The two claims are about
    different knobs -- they ramped Kc, we ramp Ks -- so rather than argue, this
    measures our own claim directly.
    """
    A = size6_graph() if A is None else A
    J = adjacency_to_J(A)
    best = brute_force_maxcut(A)[0]
    dt = stable_dt(A, Ks)
    out = {}
    for name, sched in (("Ks fixed", None),
                        ("Ks annealed", linear_ramp(Ks, 0.05 * T, 0.20 * T))):
        p = OIMParams(Kc=1.0, Ks=Ks, Ks_schedule=sched, D=D, dt=dt, T=T,
                      seed=seed)
        PH = simulate_batch(J, p, B=B)
        PH = simulate_batch(J, OIMParams(Kc=1.0, Ks=Ks, D=0.0, dt=dt, T=0.5 * T),
                            B=B, phi0=PH)
        cuts = np.array([cut_value(A, spins_from_phase(x)) for x in PH])
        rel = np.degrees(np.mod(PH - PH[:, [0]], TWO_PI))
        off = np.minimum(rel % 180, 180 - (rel % 180))
        out[name] = dict(p_opt=float(np.mean(cuts >= best)),
                         mean_cut=float(cuts.mean()),
                         binarised=float(np.mean(off.max(axis=1) < 5.0)))
    out["best"] = best
    out["B"] = B
    return out


# =============================================================================
#  STAGE 8 -- BIT-EXACT TWIN OF oim_readout  (RTL v2, all seven detectors)
#
#  Tick-for-tick model of the hardware readout, so the notebook can check the
#  board the same way section 2 checks the engine: every counter equal, not
#  approximately equal. The detectors are
#
#    xor    XOR + counter, every tick                       -> C = 100*xor/total
#    edge   rising edges of V_i                             -> f_i / f_base
#    tog    D-FF on the base rising edge, toggle count      -> lock
#    rs     RS flip-flop high ticks (SET base, RESET V_i,   -> D = 100*rs/total
#           RESET wins a tie)
#    tog2   D-FF on the QUADRATURE base edge (base + 90 deg) -> min(tog, tog2)
#    togs   D-FF on the SHIL edge (2 f0, phase 0)           -> 1x vs locked 2f
#    pid    Kim 4-bit thermometer code on the last base edge
#
#  Timing, exactly as the RTL runs it. At the start edge every accumulator is
#  seeded with its phase word; one priming cycle copies the levels into the
#  edge detectors without counting; then run tick k = 1, 2, ... sees
#      wacc_k = phase + FW*(k-1)          V_k = wacc_k < DUTY
#  and the record ends ON the tick of the WINDOW-th base rising edge, which is
#  counted. A toggle is only ever counted between two real samples of the same
#  sampling clock; the first edge arms the counter.
# =============================================================================

def _rises(v):
    """v: bool array (..., T) of run ticks 1..T. Rising edge at tick k needs
    V_k = 1 and V_{k-1} = 0, with V_0 = V_1 (the priming copy)."""
    prev = np.concatenate([v[..., :1], v[..., :-1]], axis=-1)
    return v & ~prev


def _dff_track(v, ticks, prime):
    """Sample every row of v at the given ticks. Returns (last sample, toggles)."""
    if ticks.size == 0:
        return prime.astype(int), np.zeros(v.shape[0], dtype=np.int64)
    smp = v[:, ticks]
    tog = np.sum(smp[:, 1:] != smp[:, :-1], axis=1).astype(np.int64)
    return smp[:, -1].astype(int), tog


def readout_twin(phase_q, fw, base=0, duty_q=0x8000, window=200, fw0=256,
                 phtap_q=KIM_PHTAP_Q):
    """Every readout counter of oim_pynq.v v2, integer for integer.

    phase_q : phase words the engine holds (one per oscillator)
    fw      : FW[i] frequency words, same length
    Returns a dict keyed like the registers: total, xor, edge, tog, rs, tog2,
    togs, dff, dff2, dffs, pid.
    """
    M = PHASE_MASK
    p = np.asarray(phase_q, np.int64) & M
    f = np.asarray(fw, np.int64) & M
    n = p.size
    if f[base] == 0 or not (0 < duty_q <= M):
        raise ValueError("the base never rises: the hardware would never finish")
    # enough ticks for WINDOW base periods plus the first partial one
    per = -(-(1 << PW) // int(f[base]))
    tmax = (int(window) + 2) * per + 2
    k = np.arange(tmax, dtype=np.int64)
    wb = (p[base] + f[base] * k) & M
    vb = wb < duty_q
    br = np.flatnonzero(_rises(vb))
    if br.size < max(int(window), 1):
        raise RuntimeError("record too short -- widen tmax")
    T = int(br[max(int(window), 1) - 1]) + 1                  # ticks in record
    k = k[:T]
    w = (p[:, None] + f[:, None] * k[None, :]) & M             # (n, T)
    v = w < duty_q
    vr = _rises(v)
    vb, brise = v[base], vr[base]
    bt = np.flatnonzero(brise)                                 # base edge ticks

    out = dict(total=T)
    out["xor"] = np.sum(v ^ vb[None, :], axis=1).astype(np.int64)
    out["edge"] = np.sum(vr, axis=1).astype(np.int64)
    out["dff"], out["tog"] = _dff_track(v, bt, v[:, 0])

    wq = (w[base] - (1 << (PW - 2))) & M
    bq = wq < (1 << (PW - 1))
    out["dff2"], out["tog2"] = _dff_track(v, np.flatnonzero(_rises(bq)), v[:, 0])

    ws = (((2 * int(fw0)) & M) * k) & M
    vs = ws < (1 << (PW - 1))
    out["dffs"], out["togs"] = _dff_track(v, np.flatnonzero(_rises(vs)), v[:, 0])

    # RS: state during tick k is set by the last event on a tick < k.
    rs_init = (p[base] < p)
    reset_ev = vr
    set_ev = brise[None, :] & ~vr
    idx = np.broadcast_to(k[None, :] + 1, (n, T))
    last_set = np.maximum.accumulate(np.where(set_ev, idx, 0), axis=1)
    last_rst = np.maximum.accumulate(np.where(reset_ev, idx, 0), axis=1)
    after = np.where(last_set > last_rst, 1, 0)
    none = (last_set == 0) & (last_rst == 0)
    after = np.where(none, rs_init[:, None].astype(int), after)
    state = np.concatenate([rs_init[:, None].astype(int), after[:, :-1]], axis=1)
    out["rs"] = np.sum(state, axis=1).astype(np.int64)

    tl = T - 1
    code = np.zeros(n, dtype=np.int64)
    for j in range(KIM_TAPS):
        b = (((w[:, tl] + j * int(phtap_q)) & M) < duty_q).astype(np.int64)
        code |= b << (KIM_TAPS - 1 - j)
    out["pid"] = code
    return out


def readout_percent(r):
    """The derived quantities the pipeline reads: C, D, ratio, toggle rates."""
    T = float(r["total"])
    return dict(C=100.0 * r["xor"] / T, D=100.0 * r["rs"] / T)


if __name__ == "__main__":
    sys.exit(main())
