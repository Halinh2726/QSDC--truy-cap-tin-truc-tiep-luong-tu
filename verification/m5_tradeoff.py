"""M5  Affine trade-off and GEAT ingredients, Sec. VI and App. B (exact / dense-grid checks).

  (46)-(47)  global tangent r(e) >= beta_t - Lambda_t e on [0, 1];  beta_t - Lambda_t <= 0
  (48), App. B  vertex values mu beta_t, mu(beta_t - Lambda_t), 0  -> range exactly mu Lambda_t
  (51)-(52)  lifted f_t: max Var over q' and Max - Min_Sigma  vs  mu^2 Lambda_t^2 / gamma, mu Lambda_t
  (53)-(55)  worst value over the acceptance polytope at e_t = e_tol equals mu eta_min r(e_tol)  (LP)
  (70)-(74)  dark version: beta_t <= 3 Lambda_t / 4, range unchanged, worst value mu (1-p_dc) eta_min r(e_tol)
  Cor. VI.3  r_q tangent: intercept shift beta_{q,t} = beta_t + (q - 2), global tangent of r_q
"""
import numpy as np
from scipy.optimize import linprog
from core import r_rate, tangent


def lifted_var(mu, gamma, beta, lam, pdc=0.0, n=60):
    """Var and range of the lifted function over C in {ok, err, loss, bot}, test mass gamma."""
    base = mu * (beta * (1 - pdc) + 0.75 * pdc * lam)
    f = {"ok": base, "err": base - mu * lam / gamma, "loss": base - mu * beta / gamma, "bot": base}
    vmax = 0.0
    for a in np.linspace(0, 1, n):                       # q' = (ok, err, loss) on the test branch
        for b in np.linspace(0, 1 - a, max(2, int(n * (1 - a)))):
            q = {"ok": gamma * a, "err": gamma * b, "loss": gamma * (1 - a - b), "bot": 1 - gamma}
            m = sum(q[k] * f[k] for k in q)
            vmax = max(vmax, sum(q[k] * (f[k] - m) ** 2 for k in q))
    # Min_Sigma: minimum over achievable distributions = min over test-branch vertices mixed with bot
    vals_sigma = [gamma * x + (1 - gamma) * f["bot"] for x in (f["ok"], f["err"], f["loss"])]
    rng_ = max(f.values()) - min(vals_sigma)
    return vmax, rng_


def worst_accept(mu, beta, lam, eta_min, etol, pdc=0.0):
    """LP over normalised (q_loss, q_err) >= 0 subject to (53)-(54) or (72)-(73)."""
    c = -np.array([beta, lam])                          # maximise beta ql + lam qe
    A = [[1, 0], [etol, 1]]
    b = [(1 - pdc) * (1 - eta_min), etol * (1 - pdc) + 0.75 * pdc]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])
    return mu * (beta * (1 - pdc) + 0.75 * pdc * lam) + res.fun * mu


def run():
    e = np.linspace(0, 1, 20001)
    ets = np.linspace(0.002, 0.74, 400)
    worst_tan = np.inf; worst_end = -np.inf; worst_dark = -np.inf; worst_q = np.inf
    for et in ets:
        rt, lam, beta = tangent(et)
        worst_tan = min(worst_tan, float(np.min(r_rate(e) - (beta - lam * e))))
        worst_end = max(worst_end, beta - lam)
        worst_dark = max(worst_dark, beta - 0.75 * lam)
        for q in (1.0, 1.5, 1.9):
            if q - 2 + rt > 0:                           # positive branch of (43)
                bq = beta + (q - 2)
                worst_q = min(worst_q, float(np.min(r_rate(e, q) - (bq - lam * e))))
    out = dict(min_r_minus_tangent=worst_tan, max_beta_minus_lambda=worst_end,
               max_beta_minus_3lambda_over_4=worst_dark, min_rq_minus_tangent=worst_q)
    rows = []
    for mu, gamma, eta_min, etol, pdc in [(0.8, 0.1, 1.0, 0.02, 0.0), (0.8, 0.1, 0.7, 0.02, 0.0),
                                          (0.6, 0.02, 0.5, 0.05, 0.0), (0.8, 0.1, 0.7, 0.02, 1e-3),
                                          (0.49, 0.045, 0.9, 0.01, 1e-2)]:
        rt, lam, beta = tangent(etol)
        vmax, rng_ = lifted_var(mu, gamma, beta, lam, pdc)
        rows.append(dict(mu=mu, gamma=gamma, eta_min=eta_min, etol=etol, pdc=pdc,
                         vertex_range=max(mu * beta, mu * (beta - lam), 0) - min(mu * beta, mu * (beta - lam), 0),
                         mu_lambda=mu * lam, var_max=vmax, var_bound=mu**2 * lam**2 / gamma,
                         range_sigma=rng_, worst_accept=worst_accept(mu, beta, lam, eta_min, etol, pdc),
                         claim=mu * (1 - pdc) * eta_min * float(r_rate(etol))))
    out["cases"] = rows
    return out


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1, default=float))
