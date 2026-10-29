"""M7  Completeness, Eqs. (69) and (75)-(76): Monte Carlo abort probability of the acceptance test
(53)-(54) / (72)-(73) for an honest i.i.d. channel, compared with the Hoeffding/union bound."""
import numpy as np


def abort_mc(L, gamma, eta_hon, e_hon, eta_min, etol, pdc, trials, rng):
    probs = np.array([gamma * (1 - pdc) * eta_hon * (1 - e_hon),      # test, signal, ok
                      gamma * (1 - pdc) * eta_hon * e_hon,            # test, signal, err
                      gamma * (1 - pdc) * (1 - eta_hon),              # test, loss
                      gamma * pdc * 0.25, gamma * pdc * 0.75,         # test, dark ok / err
                      1 - gamma])
    cnt = rng.multinomial(int(L), probs, size=trials)
    ql = cnt[:, 2] / L; qe = (cnt[:, 1] + cnt[:, 4]) / L
    acc = (ql <= gamma * (1 - pdc) * (1 - eta_min)) & (qe + etol * ql <= gamma * (etol * (1 - pdc) + 0.75 * pdc))
    return 1 - acc.mean()


def bound(L, gamma, eta_hon, e_hon, eta_min, etol, pdc):
    k = (1 - pdc) ** 2
    return min(1.0, np.exp(-2 * L * gamma**2 * k * (eta_hon - eta_min) ** 2)
               + np.exp(-2 * L * gamma**2 * k * eta_hon**2 * (etol - e_hon) ** 2))


def bernstein(L, gamma, eta_hon, e_hon, eta_min, etol, pdc):
    """Bernstein + union bound on the same two linear statistics (a sharper replacement, as allowed
    by the sentence after (69)); per-round variables are bounded by 1."""
    pl = gamma * (1 - pdc) * (1 - eta_hon)
    t1 = gamma * (1 - pdc) * (eta_hon - eta_min)
    v1 = pl * (1 - pl)
    pe = gamma * ((1 - pdc) * eta_hon * e_hon + 0.75 * pdc)
    mY = pe + etol * pl
    vY = pe + etol**2 * pl - mY**2                      # Y = 1[err] + etol 1[loss], disjoint events
    t2 = gamma * (1 - pdc) * eta_hon * (etol - e_hon)
    b = lambda t, v: np.exp(-L * t**2 / (2 * v + 2 * t / 3))
    return min(1.0, b(t1, v1) + b(t2, vY))


def run(seed=3, trials=4000):
    rng = np.random.default_rng(seed)
    cases = [dict(name="2*, p=0.11, no dark", gamma=0.055, pdc=0.0),
             dict(name="2*, p=0.11, p_dc=1e-3", gamma=0.055, pdc=1e-3),
             dict(name="2**, p=0.2, no dark", gamma=0.02, pdc=0.0)]
    common = dict(eta_hon=0.8, e_hon=0.01, eta_min=0.7, etol=0.02)
    out = []
    for c in cases:
        for L in np.logspace(4, 8, 17):
            out.append(dict(case=c["name"], L=L,
                            abort_mc=abort_mc(L, c["gamma"], **common, pdc=c["pdc"], trials=trials, rng=rng),
                            abort_bound=bound(L, c["gamma"], **common, pdc=c["pdc"]),
                            abort_bernstein=bernstein(L, c["gamma"], **common, pdc=c["pdc"])))
    return out, common


if __name__ == "__main__":
    import pandas as pd
    print(pd.DataFrame(run(trials=2000)[0]).to_string())
