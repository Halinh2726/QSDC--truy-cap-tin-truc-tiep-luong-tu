"""M8  Scheduling and sampling, Sec. II-B to II-D: adaptive attack (original vs random slots),
early-J_A counterexample, and Monte Carlo check of the sifting rates (11), (12) and Remark II.1."""
import sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from qsdc_slot_sim import simulate
from core import r_rate


def adaptive(L=200_000, seeds=5):
    rows = []
    for sched in ("original", "random"):
        for p in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            rs = [simulate(L, p, sched, "after_ctrl", seed=s) for s in range(seeds)]
            e = np.mean([r["e_mixed"] for r in rs]); k = np.mean([r["eve_known_frac"] for r in rs])
            rows.append(dict(schedule=sched, p=p, e=e, kappa=k, certified=float(r_rate(e)), ceiling=2 * (1 - k),
                             e_exact=0.0 if sched == "original" else 0.75 * p, kappa_exact=p))
    r = simulate(L, 0.2, "random", "early_J", seed=0)
    rows.append(dict(schedule="random, early J_A", p=0.2, e=r["e_mixed"], kappa=r["eve_known_frac"],
                     certified=float(r_rate(r["e_mixed"])), ceiling=2 * (1 - r["eve_known_frac"]), e_exact=0.0, kappa_exact=1.0))
    return rows


def sifting(L=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for p in (0.1, 0.3):                                   # Protocol 2*, sixteen-state control ensemble
        T = rng.random(L) < p; b = rng.integers(4, size=L)       # 0 ZZ, 1 ZX, 2 XZ, 3 XX
        rows.append(dict(protocol="2*", pA=p, pB=None, mu_mc=np.mean(~T), mu=1 - p,
                         gamma_mc=np.mean(T & ((b == 1) | (b == 2))), gamma=p / 2))
        b2 = rng.integers(2, size=L)                              # Remark II.1: only C1, C2
        rows.append(dict(protocol="2* (two-basis)", pA=p, pB=None, mu_mc=np.mean(~T), mu=1 - p,
                         gamma_mc=np.mean(T), gamma=p))
    for pA, pB in ((0.2, 0.2), (0.2, 0.4), (0.4, 0.1)):
        T = rng.choice(3, size=L, p=[1 - pA, pA / 2, pA / 2]); Y = rng.choice(3, size=L, p=[1 - pB, pB / 2, pB / 2])
        rows.append(dict(protocol="2**", pA=pA, pB=pB, mu_mc=np.mean((T == 0) & (Y == 0)), mu=(1 - pA) * (1 - pB),
                         gamma_mc=np.mean((T > 0) & (T == Y)), gamma=pA * pB / 2))
    return rows


if __name__ == "__main__":
    import pandas as pd
    print(pd.DataFrame(adaptive(50_000, 2)).round(4).to_string()); print(pd.DataFrame(sifting()).round(5).to_string())
