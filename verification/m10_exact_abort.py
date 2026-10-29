"""M10  Exact abort probability of the acceptance test (53)-(54) / (72)-(73) for an honest i.i.d. channel.

Per physical round the classical record falls in one of three classes:
  loss on the test branch        p_l = gamma (1 - p_dc)(1 - eta_hon)
  recorded error on the test     p_e = gamma [(1 - p_dc) eta_hon e_hon + 3 p_dc / 4]
  anything else                  1 - p_l - p_e
The counts (N_l, N_e) are multinomial(L; p_l, p_e).  Acceptance requires
  N_l <= A := floor(L gamma (1 - p_dc)(1 - eta_min))
  N_e <= floor(L b - e_tol N_l),   b = gamma [e_tol (1 - p_dc) + 3 p_dc / 4].
Hence
  Pr[abort] = Pr[N_l > A] + sum_{n <= A} Pr[N_l = n] Pr[N_e > floor(L b - e_tol n) | N_l = n],
with N_e | N_l = n ~ Binomial(L - n, p_e / (1 - p_l)).  All terms are evaluated in log space, so the
result is exact up to floating-point rounding down to probabilities of order 1e-300.
"""
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp


def log_abort(L, gamma, eta_hon, e_hon, eta_min, etol, pdc=0.0, width=60.0):
    L = int(round(L))
    pl = gamma * (1 - pdc) * (1 - eta_hon)
    pe = gamma * ((1 - pdc) * eta_hon * e_hon + 0.75 * pdc)
    A = int(np.floor(L * gamma * (1 - pdc) * (1 - eta_min) + 1e-9))
    b = gamma * (etol * (1 - pdc) + 0.75 * pdc)
    terms = [binom.logsf(A, L, pl)]                                   # log Pr[N_l > A]
    mu, sd = L * pl, np.sqrt(L * pl * (1 - pl))
    lo, hi = max(0, int(mu - width * sd) - 1), min(A, int(mu + width * sd) + 1)
    if hi >= lo:
        n = np.arange(lo, hi + 1)
        thr = np.floor(L * b - etol * n + 1e-9)
        lpmf = binom.logpmf(n, L, pl)
        ltail = np.where(thr < 0, 0.0, binom.logsf(thr, L - n, pe / (1 - pl)))
        terms.append(logsumexp(lpmf + ltail))
    return float(logsumexp(terms))


def run(trials_mc=None):
    """Exact abort probability on the same grid and cases as M7."""
    from m7_complete import bound, bernstein
    cases = [("2*, p=0.11, no dark", 0.055, 0.0), ("2*, p=0.11, p_dc=1e-3", 0.055, 1e-3), ("2**, p=0.2, no dark", 0.02, 0.0)]
    common = dict(eta_hon=0.8, e_hon=0.01, eta_min=0.7, etol=0.02)
    rows = []
    for name, g, pdc in cases:
        for L in np.logspace(4, 8, 17):
            la = log_abort(L, g, **common, pdc=pdc)
            rows.append(dict(case=name, L=L, log10_abort_exact=la / np.log(10), abort_exact=np.exp(la),
                             abort_hoeffding=bound(L, g, **common, pdc=pdc), abort_bernstein=bernstein(L, g, **common, pdc=pdc)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(run())
    print(df.to_string())
