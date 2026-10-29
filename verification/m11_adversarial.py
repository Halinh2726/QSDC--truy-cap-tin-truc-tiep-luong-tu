"""M11  Adversarial search over single-round attacks.

Instead of sampling attacks at random, minimise the gap
      g(V) = H(D | E', det) - r(e(V)),      e = (e_1 + e_2) / 2,
over isometries V : Q -> B (x) E', B = C^4 (+) |vac>, dim E' = dE, by numerical optimisation.
V is parametrised by an unconstrained complex matrix X in C^{(5 dE) x 4} through the polar map
V = X (X^dag X)^{-1/2}, which is smooth and surjective onto isometries.  A penalty
w (e - e*)^2 steers the search to a target mismatch e*.

Control experiment: the same optimiser applied to the invalid test (Z_P (x) Z_S mismatch, Lemma 2)
must find violations; this calibrates the power of the search.
"""
import numpy as np
from scipy.optimize import minimize
from core import BELL, BASES, C1, C2, r_rate
from m2_quantum import virtual_state, H_D_given_E, mismatch

DB = 5


def iso_from_params(x, dE):
    n = DB * dE * 4
    X = (x[:n] + 1j * x[n:]).reshape(DB * dE, 4)
    w, U = np.linalg.eigh(X.conj().T @ X)
    V = X @ (U * (1 / np.sqrt(np.maximum(w, 1e-12)))) @ U.conj().T
    return V.reshape(DB, dE, 4)


def evaluate(x, dE, test="mixed"):
    V = iso_from_params(x, dE)
    psi, pdet = virtual_state(V)
    H = H_D_given_E(psi)
    if test == "mixed":
        e = 0.5 * (mismatch(psi, C1) + mismatch(psi, C2))
    else:
        e = mismatch(psi, BASES["ZZ"])
    return H, e, pdet


def search(e_target, dE, test, rng, w=200.0, pdet_min=0.05, maxiter=300):
    def obj(x):
        H, e, pdet = evaluate(x, dE, test)
        pen = w * (e - e_target) ** 2 + (100.0 * (pdet_min - pdet) ** 2 if pdet < pdet_min else 0.0)
        return H - float(r_rate(e)) + pen
    x0 = rng.normal(size=2 * DB * dE * 4)
    # start near the identity channel so that low-error regions are reachable
    n = DB * dE * 4
    I = np.zeros((DB, dE, 4)); I[:4, 0, :] = np.eye(4)
    x0[:n] = 3 * I.reshape(-1) + 0.3 * x0[:n]; x0[n:] *= 0.3
    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": 40000})
    H, e, pdet = evaluate(res.x, dE, test)
    return dict(test=test, dE=dE, e_target=e_target, e=e, H=H, r=float(r_rate(e)), gap=H - float(r_rate(e)),
                pdet=pdet, nit=res.nit)


TARGETS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6)


def run(seed=21):
    """dE = 2, 4 on all targets (2 and 3 starts), dE = 8 on three targets; ZZ control for dE = 2, 4."""
    rng = np.random.default_rng(seed)
    rows = []
    plan = [(2, TARGETS, 2), (4, TARGETS, 3), (8, (0.05, 0.2, 0.45), 1)]
    for dE, targets, starts in plan:
        for et in targets:
            for _ in range(starts):
                rows.append(search(et, dE, "mixed", rng, maxiter=600))
    for dE in (2, 4):
        for et in (0.0, 0.05, 0.1, 0.2):
            rows.append(search(et, dE, "ZZ", rng, maxiter=600))
    return rows


if __name__ == "__main__":
    import pandas as pd, time
    t = time.time()
    rng = np.random.default_rng(0)
    print(pd.DataFrame([search(0.1, 2, "mixed", rng), search(0.0, 2, "ZZ", rng)]).to_string(), time.time() - t)
