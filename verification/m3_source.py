"""M3  Calibrated source-overlap extension: Prop. V.4 (44), definition (42)-(43), Remark V.5 (45).

Alice's virtual measurements are perturbed rank-one bases  D~ = W_D D,  C~_j = W_C C_j  with
W = exp(-i eps G).  Rank-one projective bases always average to I/4, so the common-average-state
assumption of Sec. V-F holds automatically.  Bob keeps the ideal analyzer C_j.
"""
import numpy as np
from core import BELL, C1, C2, r_rate, gue
from m2_quantum import virtual_state, H_D_given_E, mismatch, attack_haar_mix, attack_coherent, attack_intercept


def unitary(eps, rng):
    w, v = np.linalg.eigh(gue(4, rng))
    return (v * np.exp(-1j * eps * w)) @ v.conj().T


def phase_dist(a, b):
    return np.sqrt(max(0.0, 2 - 2 * abs(np.vdot(a, b))))


def run(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        eps_d, eps_c = rng.uniform(0, 0.5, 2)
        WD, WC = unitary(eps_d, rng), unitary(eps_c, rng)
        Dt = (WD @ BELL.T).T
        C1t, C2t = (WC @ C1.T).T, (WC @ C2.T).T
        c_src = max(np.max(np.abs(Dt.conj() @ Cb.T) ** 2) for Cb in (C1t, C2t))
        q = min(2.0, -np.log2(c_src))
        dD = max(phase_dist(Dt[k], BELL[k]) for k in range(4))
        dC = max(phase_dist(Ct[k], C[k]) for Ct, C in ((C1t, C1), (C2t, C2)) for k in range(4))
        bound45 = min(1.0, (0.5 + dD + dC + dD * dC) ** 2)
        fam = rng.integers(4)
        V = (attack_haar_mix(rng.uniform(0, 1) ** 2, 2, rng) if fam == 0 else
             attack_coherent(rng.uniform(0, 2), rng, rng.uniform(0, 1)) if fam == 1 else
             attack_intercept(rng.uniform(0, 1), rng.uniform(0, 1, 4)) if fam == 2 else
             attack_intercept(rng.uniform(0, 1), None, Dt))         # Eve targets the actual source basis
        psi, _ = virtual_state(V)
        H = H_D_given_E(psi, Dt)
        # Alice's virtual control outcome from C~_j, Bob's recorded label from the ideal C_j
        e1, e2 = mismatch(psi, C1t, C1), mismatch(psi, C2t, C2)
        ep = 0.5 * (e1 + e2)
        rows.append(dict(eps_d=eps_d, eps_c=eps_c, c_src=c_src, q_src=q, deltaD=dD, deltaC=dC,
                         bound45=bound45, slack45=bound45 - c_src, H=H, e_pool=ep,
                         gap=H - float(r_rate(ep, q)), gap_ideal_q=H - float(r_rate(ep, 2.0)), family=int(fam)))
    return rows


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(run(300))
    print(df[["c_src", "q_src", "slack45", "gap", "gap_ideal_q"]].describe())
