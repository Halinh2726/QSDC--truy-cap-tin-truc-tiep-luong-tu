"""M6  Finite-size bound, Theorem VI.2, Tables I-II, Prop. VII.1, Prop. VIII.1, tangent-point sensitivity.

All numbers are evaluations of the analytical bound (58)-(61); no new physics is simulated here.
The purpose is (i) reproducibility of Tables I-II, (ii) a numerical test of Prop. VII.1 on the full
bound, (iii) a test of the scaling exponents (88)-(89), (iv) the effect of choosing e_t != e_tol.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from core import hmin_rate, r_rate

EPS = 1e-10


def mu_gamma(proto, p, pB=None):
    if proto == "2*":
        return 1 - p, p / 2
    pB = p if pB is None else pB
    return (1 - p) * (1 - pB), p * pB / 2


def best_alpha(L, mu, gamma, eta, etol, et=None):
    f = lambda a: -hmin_rate(L, mu, gamma, eta, etol, a, et)
    res = minimize_scalar(f, bounds=(1 + 1e-12, 1.4999), method="bounded", options={"xatol": 1e-12})
    # refine on log(alpha - 1)
    g = lambda t: -hmin_rate(L, mu, gamma, eta, etol, 1 + np.exp(t), et)
    t0 = np.log(max(res.x - 1, 1e-12))
    r2 = minimize_scalar(g, bounds=(t0 - 6, min(t0 + 6, np.log(0.4999))), method="bounded")
    return (-r2.fun, 1 + np.exp(r2.x)) if -r2.fun > -res.fun else (-res.fun, res.x)


def optimise(L, proto, eta=1.0, etol=0.02, free_et=False):
    best = (-np.inf, None, None, None)
    for lp in np.linspace(np.log(1e-4), np.log(0.8), 60):
        p = np.exp(lp)
        mu, gam = mu_gamma(proto, p)
        ets = [None] if not free_et else list(np.linspace(etol, 0.3, 30))
        for et in ets:
            v, a = best_alpha(L, mu, gam, eta, etol, et)
            if v > best[0]:
                best = (v, p, a, et)
    # local refinement in p (and e_t)
    v0, p0, a0, et0 = best
    def obj(x):
        p = np.exp(x[0]); et = None if not free_et else x[1]
        if not (0 < p < 0.95) or (free_et and not (etol <= et < 0.74)):
            return 1e3
        mu, gam = mu_gamma(proto, p)
        return -best_alpha(L, mu, gam, eta, etol, et)[0]
    x0 = [np.log(p0)] + ([et0] if free_et else [])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 400})
    if -res.fun > v0:
        p = np.exp(res.x[0]); et = None if not free_et else res.x[1]
        mu, gam = mu_gamma(proto, p); v, a = best_alpha(L, mu, gam, eta, etol, et)
        return dict(rate=v, p=p, alpha=a, et=etol if et is None else et)
    return dict(rate=v0, p=p0, alpha=a0, et=etol if et0 is None else et0)


def table1():
    rows = []
    for L in (1e5, 1e6):
        for etol in (0.01, 0.02, 0.05):
            a, b = optimise(L, "2*", 1.0, etol), optimise(L, "2**", 1.0, etol)
            rows.append(dict(L=L, etol=etol, rate_2s=a["rate"], p_2s=a["p"], rate_2ss=b["rate"], p_2ss=b["p"]))
    return rows


def table2():
    rows = []
    for eta in (0.9, 0.7, 0.5, 0.3):
        a, b = optimise(1e6, "2*", eta, 0.02), optimise(1e6, "2**", eta, 0.02)
        rows.append(dict(eta_min=eta, rate_2s=a["rate"], p_2s=a["p"], rate_2ss=b["rate"], p_2ss=b["p"]))
    return rows


def symmetric_bias(L=1e6, etol=0.02):
    """Prop. VII.1: at fixed mu, scan p_A with p_B = 1 - mu/(1 - p_A); also a free 2-D optimisation."""
    out = []
    for mu in (0.5, 0.7, 0.85):
        pAs = np.linspace(1e-3, 1 - mu - 1e-3, 81)
        vals = []
        for pA in pAs:
            pB = 1 - mu / (1 - pA)
            vals.append(best_alpha(L, mu, pA * pB / 2, 1.0, etol)[0])
        k = int(np.argmax(vals))
        out.append(dict(mu=mu, pA_best=pAs[k], pA_sym=1 - np.sqrt(mu), rate_best=vals[k],
                        rate_asym_edge=vals[5]))
    # free 2-D optimisation over (pA, pB)
    def obj(x):
        pA, pB = x
        if not (1e-4 < pA < 0.9 and 1e-4 < pB < 0.9):
            return 1e3
        mu, gam = (1 - pA) * (1 - pB), pA * pB / 2
        return -best_alpha(L, mu, gam, 1.0, etol)[0]
    starts = [(0.1, 0.5), (0.5, 0.1), (0.3, 0.2)]
    free = [minimize(obj, s, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10}) for s in starts]
    return out, [dict(start=s, pA=f.x[0], pB=f.x[1], rate=-f.fun) for s, f in zip(starts, free)]


def scaling(etol=0.02):
    """Prop. VIII.1: slopes of log p_opt, log(alpha_opt - 1) and log deficit versus log L."""
    Ls = np.logspace(5, 11, 13)
    res = {}
    for proto, s in (("2*", 1), ("2**", 2)):
        rows = [dict(L=L, **optimise(L, proto, 1.0, etol)) for L in Ls]
        h0 = float(r_rate(etol))
        for r in rows:
            r["deficit"] = h0 - r["rate"]
        lL = np.log(Ls)
        fit = lambda y, k=6: np.polyfit(lL[-k:], np.log(y[-k:]), 1)[0]
        P = np.array([r["p"] for r in rows]); A = np.array([r["alpha"] - 1 for r in rows]); D = np.array([r["deficit"] for r in rows])
        # leading-order prediction (85)-(87) with mu1 = 1 (2*) or 2 (2**), gamma0 = 1/2
        import math
        from core import tangent, g_eps
        _, lam, _ = tangent(etol)
        mu1 = 1 if s == 1 else 2; g0 = 0.5
        k1, k2, k3 = mu1 * h0, math.log(2) * lam**2 / (2 * g0), g_eps(1e-10) + math.log2(1e10)
        xi = (s * math.sqrt(k2 * k3) / k1) ** (2 / (s + 2))
        lead = (s + 2) / s * k1 * xi * Ls ** (-1 / (s + 2))
        res[proto] = dict(rows=rows, slope_p=fit(P), slope_alpha=fit(A), slope_deficit=fit(D),
                          pred_p=-1 / (s + 2), pred_alpha=-(s + 1) / (s + 2), pred_deficit=-1 / (s + 2),
                          lead_deficit=lead.tolist(), xi=xi)
    return res


def et_sensitivity():
    rows = []
    for L, etol in ((1e5, 0.01), (1e5, 0.02), (1e6, 0.02), (1e6, 0.05)):
        for proto in ("2*", "2**"):
            a = optimise(L, proto, 1.0, etol, False); b = optimise(L, proto, 1.0, etol, True)
            rows.append(dict(L=L, etol=etol, proto=proto, rate_et_eq_etol=a["rate"], rate_et_free=b["rate"], et_opt=b["et"]))
    return rows


if __name__ == "__main__":
    import pandas as pd
    print(pd.DataFrame(table1()).round(3).to_string())
