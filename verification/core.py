"""Shared objects for the numerical-verification companion of
'Finite-size entropic security for random-slot hyperentangled direct communication' (v4_final).

Ordering of the local photon space P (x) S:  |H0>, |H1>, |V0>, |V1>.
Equation numbers refer to the v4_final manuscript.
"""
import itertools
import numpy as np

S2 = 1 / np.sqrt(2)
H0, H1, V0, V1 = np.eye(4)
# local Bell detector basis D, Eqs. (15)-(16); rows d00, d01, d10, d11  (index 2u+v)
BELL = np.array([S2 * (H0 + V1), S2 * (H1 + V0), S2 * (H0 - V1), S2 * (H1 - V0)])
_Q = {"Z": np.eye(2), "X": np.array([[S2, S2], [S2, -S2]])}
Xp = np.array([[0, 1], [1, 0]]); Zp = np.diag([1.0, -1.0]); I2 = np.eye(2)


def product_basis(xy):
    """Rows: the four states of basis xy, Eqs. (6)-(9) (first letter polarization)."""
    return np.array([np.kron(_Q[xy[0]][i], _Q[xy[1]][j]) for i in (0, 1) for j in (0, 1)])


BASES = {"D": BELL, "ZZ": product_basis("ZZ"), "ZX": product_basis("ZX"),
         "XZ": product_basis("XZ"), "XX": product_basis("XX")}
C1, C2 = BASES["ZX"], BASES["XZ"]          # Eq. (10)
PHI16 = np.eye(4).reshape(16) / 2          # |Phi+>_P |Phi+>_S between photons A and B, Eq. (1)


def enc(a, w):
    """Z^a X^w on the polarization of Alice's photon, Eq. (2)."""
    return np.kron(np.linalg.matrix_power(Zp, a) @ np.linalg.matrix_power(Xp, w), I2)


# ------------------------------------------------------------------ information measures
def h2(e):
    e = np.clip(np.asarray(e, float), 1e-300, 1)
    f = np.clip(1 - e, 1e-300, 1)
    return -(e * np.log2(e) + f * np.log2(f))


def r_rate(e, q=2.0):
    """Eq. (22) (q = 2) and Eq. (43) (general q)."""
    e = np.asarray(e, float)
    val = q - h2(e) - e * np.log2(3)
    return np.where(e <= 0.75, np.maximum(val, 0.0), 0.0)


def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


def cond_entropy_cq(blocks):
    """H(X|E) for a cq state sum_x |x><x| (x) blocks[x] (subnormalised, total trace 1)."""
    rhoE = sum(blocks)
    s = 0.0
    for b in blocks:
        p = np.trace(b).real
        if p > 1e-15:
            s += -p * np.log2(p) + p * vn(b / p)
    return s - vn(rhoE)


def haar_isometry(din, dout, rng):
    z = (rng.normal(size=(dout, din)) + 1j * rng.normal(size=(dout, din))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def gue(d, rng):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / 2
    return (z + z.conj().T) / np.sqrt(2 * d)


# ------------------------------------------------------------------ finite-size bound, Sec. VI
LOG2_51 = np.log2(51)


def g_eps(eps):
    """Eq. (60), written in the cancellation-free form -log2(1 - sqrt(1-e^2)) = log2((1 + sqrt(1-e^2)) / e^2)."""
    return np.log2((1 + np.sqrt(1 - eps**2)) / eps**2)


def tangent(et):
    """Eq. (46): r_t, Lambda_t, beta_t."""
    rt = float(2 - h2(et) - et * np.log2(3))
    lam = float(np.log2(3 * (1 - et) / et))
    return rt, lam, rt + et * lam


def first_order(mu, eta_min, etol, et=None, pdc=0.0):
    """Worst value of the lifted affine function over the acceptance polytope (53)-(54)
    (or (72)-(73) when pdc > 0), divided by nothing: entropy per physical round.
    At et = etol this equals mu (1 - pdc) eta_min r(etol), Eqs. (55), (74)."""
    et = etol if et is None else et
    _, lam, beta = tangent(et)
    # vertices of {ql <= (1-pdc)(1-eta_min), qe + etol ql <= etol(1-pdc) + 3pdc/4, ql, qe >= 0}
    # (all quantities normalised by gamma); f = mu[beta(1-pdc) + 3pdc lam/4 - beta ql - lam qe]
    qlmax = (1 - pdc) * (1 - eta_min); b = etol * (1 - pdc) + 0.75 * pdc
    verts = [(0.0, b), (qlmax, max(0.0, b - etol * qlmax))]
    worst = max(beta * ql + lam * qe for ql, qe in verts)
    return mu * (beta * (1 - pdc) + 0.75 * pdc * lam - worst)


def delta_L(L, alpha, mu, gamma, lam, eps_s, eps_a):
    """Eqs. (57), (59), (61) with Lambda_tol replaced by the tangent slope actually used."""
    VG = LOG2_51 + np.sqrt(2 + mu**2 * lam**2 / gamma)
    XG = 2 * np.log2(5) + mu * lam
    d = (alpha - 1) / (2 - alpha)
    K = (2 - alpha) ** 3 / (6 * (3 - 2 * alpha) ** 3 * np.log(2)) * 2 ** (d * XG) * np.log(2 ** XG + np.e**2) ** 3
    return L * d * np.log(2) / 2 * VG**2 + (g_eps(eps_s) + alpha * np.log2(1 / eps_a)) / (alpha - 1) + L * d**2 * K


def hmin_rate(L, mu, gamma, eta_min, etol, alpha, et=None, eps_s=1e-10, eps_a=1e-10, pdc=0.0):
    """Lower bound (58) divided by L."""
    et = etol if et is None else et
    _, lam, _ = tangent(et)
    return first_order(mu, eta_min, etol, et, pdc) - delta_L(L, alpha, mu, gamma, lam, eps_s, eps_a) / L
