"""M4  Calibrated receiver model, Sec. V-B to V-D.

(a) Prop. V.2, Eqs. (31)-(33): trusted random thinning makes detection basis independent.
    Exact POVM algebra + Monte Carlo of click statistics before/after thinning.
(b) Eqs. (34)-(36): exclusive-branch dark events; estimators eta_sig, e_sig from test statistics.
(c) Eqs. (38)-(40): symmetric cross-talk relation and the worst-case LP (39) with a finite-data box.
"""
import numpy as np
from scipy.optimize import linprog
from core import BASES, BELL


def thinning(rng, n=400_000):
    eff = {y: rng.uniform(0.55, 0.95, 4) for y in ("D", "C1", "C2")}
    eta0 = min(v.min() for v in eff.values())
    Pi = {"D": BELL, "C1": BASES["ZX"], "C2": BASES["XZ"]}
    # exact: retained POVM elements theta * eta * Pi = eta0 * Pi
    dev = 0.0
    for y in eff:
        for z in range(4):
            M = eff[y][z] * np.outer(Pi[y][z], Pi[y][z])
            Mt = (eta0 / eff[y][z]) * M
            dev = max(dev, np.abs(Mt - eta0 * np.outer(Pi[y][z], Pi[y][z])).max())
    # Monte Carlo: input state = a fixed non-uniform state; click probability per analyzer
    psi = np.array([0.8, 0.1j, 0.5, -0.3]); psi /= np.linalg.norm(psi)
    out = {}
    for y in eff:
        pz = np.abs(Pi[y].conj() @ psi) ** 2
        z = rng.choice(4, size=n, p=pz)
        click = rng.random(n) < eff[y][z]
        keep = click & (rng.random(n) < eta0 / eff[y][z])
        out[y] = dict(p_click_raw=click.mean(), p_click_thinned=keep.mean(),
                      tv_retained_vs_ideal=0.5 * np.abs(np.bincount(z[keep], minlength=4) / keep.sum() - pz).sum())
    return dict(eta0=eta0, povm_dev=dev, mc=out)


def dark(rng, L=2_000_000, trials=20):
    rows = []
    for pdc, eta_sig, e_sig in [(0.0, 0.9, 0.02), (1e-3, 0.8, 0.02), (1e-2, 0.6, 0.03), (5e-2, 0.5, 0.05)]:
        est = []
        for _ in range(trials):
            g = rng.random(L) < pdc                                     # dark flag G_i, Eq. (34)
            sig = ~g & (rng.random(L) < eta_sig / (1 - pdc))            # ordinary branch: detected
            err = np.where(g, rng.random(L) < 0.75, sig & (rng.random(L) < e_sig))
            q_loss = np.mean(~g & ~sig); q_err = np.mean(err)
            eta_hat = 1 - pdc - q_loss                                   # Eq. (35)
            est.append((eta_hat, (q_err - 0.75 * pdc) / eta_hat))       # Eq. (36)
        est = np.array(est)
        rows.append(dict(pdc=pdc, eta_sig=eta_sig, e_sig=e_sig, eta_hat=est[:, 0].mean(), eta_sd=est[:, 0].std(),
                         e_hat=est[:, 1].mean(), e_sd=est[:, 1].std(),
                         naive_e=(est[:, 1] * est[:, 0] + 0.75 * pdc).mean() / (est[:, 0].mean() + pdc)))
    return rows


def worst_intrinsic(N, phat, box):
    """LP (39): maximise 1 - (1/4) sum_z p_{z|z} over intended conditionals p_z in the simplex
    with N p_z inside the box [phat_z - box, phat_z + box] (componentwise)."""
    n = 4
    c = np.zeros(n * n)
    for z in range(n):
        c[z * n + z] = 0.25                   # minimise (1/4) sum p_{z|z}
    A, b = [], []
    for z in range(n):
        for i in range(n):
            row = np.zeros(n * n); row[z * n:(z + 1) * n] = N[i]
            A.append(row); b.append(phat[z][i] + box)
            A.append(-row); b.append(-(phat[z][i] - box))
    Aeq = np.zeros((n, n * n)); beq = np.ones(n)
    for z in range(n):
        Aeq[z, z * n:(z + 1) * n] = 1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=beq, bounds=[(0, 1)] * (n * n))
    return 1 - res.fun if res.success else np.nan


def crosstalk(rng):
    rows = []
    for chi in [0.0, 0.02, 0.05, 0.1, 0.3]:
        for e in [0.01, 0.03, 0.08]:
            N = np.full((4, 4), chi / 3); np.fill_diagonal(N, 1 - chi)
            # intended conditionals: correct with 1-e, errors spread uniformly
            P = np.full((4, 4), e / 3); np.fill_diagonal(P, 1 - e)
            phat = [N @ P[z] for z in range(4)]
            e_obs = 1 - np.mean([phat[z][z] for z in range(4)])
            e_from40 = chi + (1 - 4 * chi / 3) * e
            # Monte Carlo of recorded labels
            n = 400_000
            zint = np.where(rng.random(n) < e, rng.integers(1, 4, n), 0)    # intended label error
            flip = rng.random(n) < chi
            zrec = np.where(flip, (zint + rng.integers(1, 4, n)) % 4, zint)
            e_mc = np.mean(zrec != 0)
            e_inv = (e_obs - chi) / (1 - 4 * chi / 3)
            rows.append(dict(chi=chi, e=e, e_obs=e_obs, e_eq40=e_from40, e_mc=e_mc,
                             lp_exact=worst_intrinsic(N, phat, 0.0), e_inverted=e_inv,
                             lp_box_1e3=worst_intrinsic(N, phat, 1e-3), lp_box_1e2=worst_intrinsic(N, phat, 1e-2)))
    return rows


def run(seed=5):
    rng = np.random.default_rng(seed)
    return dict(thinning=thinning(rng), dark=dark(rng), crosstalk=crosstalk(rng))


if __name__ == "__main__":
    import json, pandas as pd
    o = run()
    print(json.dumps(o["thinning"], indent=1, default=float))
    print(pd.DataFrame(o["dark"]).to_string()); print(pd.DataFrame(o["crosstalk"]).to_string())
