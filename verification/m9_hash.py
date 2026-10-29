"""M9  Classical toy check of the leftover-hash bound (65) with eps_s = 0.

S uniform on {0,1}^n, Eve's classical side information E = phi(S).  Exact H_min(S|E) = k.
Hash family: all binary l x n matrices (two-universal).  For each sampled seed the trace distance
d(F) = 1/2 sum_{e,y} |P(K=y, E=e | F) - 2^{-l} P(E=e)| is computed exactly by enumeration;
the bound concerns the seed average E_F d(F) <= 1/2 * 2^{-(k - l)/2}.
This exercises only the classical special case of the quantum leftover-hash lemma used in Thm VI.5.
"""
import numpy as np


def run(n=12, seeds=400, seed=9):
    rng = np.random.default_rng(seed)
    X = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    leaks = {
        "linear, 4 parities": lambda: (X @ rng.integers(0, 2, (n, 4)) % 2) @ (1 << np.arange(4)),
        "Hamming weight": lambda: X.sum(1),
        "first 3 bits": lambda: X[:, :3] @ (1 << np.arange(3)),
        "noisy first 6 bits (flip 0.1, fixed seed)": None,
    }
    out = []
    for name, f in leaks.items():
        if f is None:
            # E = first 6 bits passed through a fixed random BSC realisation per S (deterministic function of S)
            flips = (rng.random((2**n, 6)) < 0.1).astype(np.int64)
            E = ((X[:, :6] ^ flips) @ (1 << np.arange(6)))
        else:
            E = f()
        ne = E.max() + 1
        # H_min(S|E) for uniform S and deterministic E: -log2 sum_e max_s P(s, e) = -log2(#distinct e / 2^n)
        k = -np.log2(len(np.unique(E)) / 2**n)
        PE = np.bincount(E, minlength=ne) / 2**n
        for l in range(1, n + 1):
            ds = []
            for _ in range(seeds):
                M = rng.integers(0, 2, (n, l))
                Y = (X @ M % 2) @ (1 << np.arange(l))
                J = np.zeros((ne, 2**l)); np.add.at(J, (E, Y), 1 / 2**n)
                ds.append(0.5 * np.abs(J - PE[:, None] / 2**l).sum())
            out.append(dict(leak=name, k=k, l=l, d_avg=float(np.mean(ds)), bound=min(1.0, 0.5 * 2 ** (-(k - l) / 2))))
    return out


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(run(seeds=100)); df["ok"] = df.d_avg <= df.bound + 1e-12
    print(df.round(4).to_string())
