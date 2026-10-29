#!/usr/bin/env python3
"""
qsdc_slot_sim.py — slot-level Monte Carlo for the random-slot hyperentangled QSDC manuscript.

What is simulated
-----------------
Each slot carries one photon in polarization (P) x spatial mode (S), i.e. a 4-dimensional system.

* Message slot.  Alice keeps photon A and sends photon B.  By Lemma III.1 / Eq. (20), after Alice's
  local Bell measurement with outcome J_A, photon B is the Bell-basis state |d_D>, D = J_A xor S,
  where S = (a, w) is the raw symbol.  Bob measures B in the Bell basis D and, once J_A is announced,
  decodes S = J_A xor J_B.
* Control slot.  Alice sends one of the 16 product states (bases ZZ, ZX, XZ, XX).  Only the mixed
  bases C1 = ZX and C2 = XZ are mutually unbiased with D and enter the test statistics (Eq. (10)).
* Eve.  Intercept-resend in the Bell basis D on a chosen set of slots.  On a message slot this is
  non-disturbing (the photon is a D-eigenstate) and reveals D, hence S once J_A is public.
* Honest channel.  Depolarizing noise (lambda), loss (detection probability eta) and exclusive
  dark clicks (p_dc, uniform label), applied after Eve.

Schedules
---------
original : Protocol 2 of [19] - after a control slot the next slot is always a message slot.
random   : Protocol 2* - i.i.d. slot types, Pr[ctrl] = p.
memoryless (subcommand `memoryless`) : Protocol 2** - Alice T in {msg, C1, C2}, Bob Y in {D, C1, C2}.

Attacks
-------
none | all | fraction (random fraction f of slots) | after_ctrl (slots right after a disclosed control)
| early_J (J_A announced while the slot is still in the channel -> Eve attacks exactly message slots)

Subcommands
-----------
selfcheck   analytic checks (Bell basis, MUB overlaps, Lemma III.1, expected error rates)
table       schedule x attack table
sweep       attack-fraction sweep: observed e vs certified rbar(e) vs Eve-blind bits (CSV + PNG)
honest      honest-channel abort probability vs L, compared with the Hoeffding bound (64) (CSV + PNG)
memoryless  Protocol 2** retained fractions vs mu = (1-pA)(1-pB), gamma = pA pB / 2
all         run everything

Usage:  python3 qsdc_slot_sim.py all --out results
Requires: numpy, matplotlib (matplotlib only for the figures).

This is an illustration of the protocol logic under one attack family; it is not a security proof.
"""
import argparse
import csv
import itertools
import os

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. States and bases (ordering |H0>, |H1>, |V0>, |V1>)
# ----------------------------------------------------------------------------------------------
S2 = 1 / np.sqrt(2)
H0, H1, V0, V1 = np.eye(4)
BELL = np.array([S2 * (H0 + V1), S2 * (H1 + V0), S2 * (H0 - V1), S2 * (H1 - V0)])  # d00 d01 d10 d11, Eqs. (15)-(16)
_Q = {"Z": np.eye(2), "X": np.array([[S2, S2], [S2, -S2]])}


def product_basis(xy):
    """Rows are the four states of basis xy (first letter: polarization, second: spatial mode)."""
    return np.array([np.kron(_Q[xy[0]][i], _Q[xy[1]][j]) for i in (0, 1) for j in (0, 1)])


BASES = {"D": BELL, "ZZ": product_basis("ZZ"), "ZX": product_basis("ZX"),
         "XZ": product_basis("XZ"), "XX": product_basis("XX")}
CTRL_BASES = ("ZZ", "ZX", "XZ", "XX")
MIXED = ("ZX", "XZ")                       # C1, C2
BIDX = {b: i for i, b in enumerate(BASES)}  # basis name -> integer code


def transition(meas_basis, state_basis):
    """T[out, k] = |<out| k>|^2 : probability of outcome `out` when measuring `meas_basis` on state k."""
    return np.abs(BASES[meas_basis].conj() @ BASES[state_basis].T) ** 2


TRANS = {(m, s): transition(m, s) for m in BASES for s in BASES}

# ----------------------------------------------------------------------------------------------
# 2. Helpers
# ----------------------------------------------------------------------------------------------


def h2(e):
    e = np.clip(e, 1e-15, 1 - 1e-15)
    return -e * np.log2(e) - (1 - e) * np.log2(1 - e)


def rbar(e):
    """Four-symbol uncertainty rate, Eq. (22): 2 - h2(e) - e log2 3 on [0, 3/4], 0 beyond."""
    e = np.asarray(e, dtype=float)
    out = 2 - h2(e) - e * np.log2(3)
    return np.where(e <= 0.75, np.maximum(out, 0.0), 0.0)


def sample_rows(prob, rng):
    """prob: (N, 4) row-stochastic -> one categorical sample per row."""
    c = np.cumsum(prob, axis=1)
    u = rng.random((prob.shape[0], 1)) * c[:, -1:]
    return (u > c).sum(axis=1)


def measure(state_basis, state_idx, meas_basis, rng, lam=0.0):
    """Vectorized projective measurement with optional depolarizing noise lambda.
    state_basis, meas_basis: integer basis codes per slot; state_idx: label per slot."""
    names = list(BASES)
    out = np.empty(state_idx.shape[0], dtype=int)
    for sb, mb in itertools.product(range(len(names)), repeat=2):
        sel = (state_basis == sb) & (meas_basis == mb)
        if not sel.any():
            continue
        T = TRANS[(names[mb], names[sb])]               # (out, k)
        prob = (1 - lam) * T[:, state_idx[sel]].T + lam / 4
        out[sel] = sample_rows(prob, rng)
    return out

# ----------------------------------------------------------------------------------------------
# 3. Protocol 2 / 2* slot simulation
# ----------------------------------------------------------------------------------------------


def slot_types(L, p, schedule, rng):
    """Boolean array ctrl[i]."""
    if schedule == "random":
        return rng.random(L) < p
    if schedule == "original":                          # [19], Protocol 2, Sec. 3.2: S1 -> S2
        u = rng.random(L)
        ctrl = np.zeros(L, dtype=bool)
        prev = False
        for i in range(L):
            ctrl[i] = (not prev) and u[i] < p
            prev = ctrl[i]
        return ctrl
    raise ValueError(schedule)


def attack_mask(ctrl, attack, f, rng):
    prev_ctrl = np.concatenate([[False], ctrl[:-1]])
    if attack == "none":
        return np.zeros_like(ctrl)
    if attack == "all":
        return np.ones_like(ctrl)
    if attack == "fraction":
        return rng.random(ctrl.shape[0]) < f
    if attack == "after_ctrl":
        return prev_ctrl.copy()
    if attack == "early_J":
        return ~ctrl
    raise ValueError(attack)


def simulate(L, p, schedule="random", attack="none", f=0.0, lam=0.0, eta=1.0, p_dc=0.0, seed=0):
    """Returns a dict of test statistics and Eve's knowledge for one execution of L slots."""
    rng = np.random.default_rng(seed)
    ctrl = slot_types(L, p, schedule, rng)
    n = L
    # Alice's preparation
    cb = rng.integers(len(CTRL_BASES), size=n)                       # control basis
    k = rng.integers(4, size=n)                                      # control label
    S = rng.integers(4, size=n)                                      # raw symbol (a, w)
    J = rng.integers(4, size=n)                                      # Alice's Bell outcome J_A
    Dlab = J ^ S                                                     # Eq. (20)
    st_basis = np.where(ctrl, np.array([BIDX[b] for b in CTRL_BASES])[cb], BIDX["D"])
    st_idx = np.where(ctrl, k, Dlab)
    # Eve: intercept-resend in the Bell basis
    att = attack_mask(ctrl, attack, f, rng)
    eve_out = np.full(n, -1)
    if att.any():
        eve_out[att] = measure(st_basis[att], st_idx[att], np.full(att.sum(), BIDX["D"]), rng)
        st_basis = np.where(att, BIDX["D"], st_basis)
        st_idx = np.where(att, eve_out, st_idx)
    # Bob (Protocol 2*: control basis disclosed after the slot has left the channel)
    bob_basis = np.where(ctrl, np.array([BIDX[b] for b in CTRL_BASES])[cb], BIDX["D"])
    dark = rng.random(n) < p_dc
    det = dark | (rng.random(n) < eta)
    out = measure(st_basis, st_idx, bob_basis, rng, lam)
    out = np.where(dark, rng.integers(4, size=n), out)
    # Statistics on the pooled mixed-basis test (the GEAT test register)
    mixed = ctrl & np.isin(cb, [CTRL_BASES.index(b) for b in MIXED])
    unmixed = ctrl & ~mixed
    err = det & (out != k)
    q_loss = np.sum(mixed & ~det) / L
    q_err = np.sum(mixed & err) / L
    n_mixed_det = np.sum(mixed & det)
    msg = ~ctrl
    known = msg & att & ((eve_out ^ J) == S)                         # Eve's S once J_A is public
    return {
        "gamma_hat": mixed.mean(),
        "e_mixed": np.sum(mixed & err) / max(1, n_mixed_det),          # conditional mismatch on C1, C2
        "e_unmixed": np.sum(unmixed & err) / max(1, np.sum(unmixed & det)),
        "q_loss": q_loss, "q_err": q_err,
        "eve_known_frac": known.sum() / max(1, msg.sum()),
        "bob_msg_err": np.sum(msg & det & (out != Dlab)) / max(1, np.sum(msg & det)),
    }

# ----------------------------------------------------------------------------------------------
# 4. Protocol 2** (memoryless)
# ----------------------------------------------------------------------------------------------


def simulate_memoryless(L, pA, pB, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.choice(3, size=L, p=[1 - pA, pA / 2, pA / 2])          # 0 msg, 1 C1, 2 C2
    Y = rng.choice(3, size=L, p=[1 - pB, pB / 2, pB / 2])          # 0 D,   1 C1, 2 C2
    return {"mu_hat": np.mean((T == 0) & (Y == 0)), "mu": (1 - pA) * (1 - pB),
            "gamma_hat": np.mean((T > 0) & (T == Y)), "gamma": pA * pB / 2}

# ----------------------------------------------------------------------------------------------
# 5. Subcommands
# ----------------------------------------------------------------------------------------------


def cmd_selfcheck(_):
    ok = np.allclose(BELL @ BELL.T, np.eye(4))
    print(f"Bell basis orthonormal ................ {ok}")
    for b in CTRL_BASES:
        ov = np.unique(np.round(TRANS[("D", b)], 6))
        print(f"overlaps |<d|c>|^2, D vs {b} ........... {ov}")
    # Lemma III.1, Eq. (18)
    X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
    phi = S2 * (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1]))
    good = True
    for a, w in itertools.product((0, 1), repeat=2):
        beta = np.kron(np.linalg.matrix_power(Z, a) @ np.linalg.matrix_power(X, w), np.eye(2)) @ phi
        st = np.kron(beta, phi).reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(16)
        rhs = sum(0.5 * (-1) ** (w * (u ^ a)) * np.kron(BELL[2 * u + v], BELL[2 * (u ^ a) + (v ^ w)])
                  for u in (0, 1) for v in (0, 1))
        good &= np.allclose(st, rhs)
    print(f"Lemma III.1 (Eq. 18), all (a, w) ...... {good}")
    # expected mismatch under full Bell-basis intercept-resend
    # P(correct | state c) = sum_d P(Bob gets c | resent d) P(Eve gets d | c) = diag(T_bD @ T_Db)
    e_mix = np.mean([1 - np.mean(np.diag(TRANS[(b, "D")] @ TRANS[("D", b)])) for b in MIXED])
    e_unm = np.mean([1 - np.mean(np.diag(TRANS[(b, "D")] @ TRANS[("D", b)])) for b in ("ZZ", "XX")])
    print(f"expected e under full D-attack: mixed {e_mix:.3f} (theory 0.75), unmixed {e_unm:.3f} (theory 0.50)")


def cmd_table(a):
    cases = [("original", "none"), ("original", "after_ctrl"), ("random", "none"),
             ("random", "all"), ("random", "early_J")]
    rows = []
    print(f"L = {a.L}, p = {a.p}")
    print(f"{'schedule':9s} {'attack':11s} {'e(C1,C2)':>9s} {'e(ZZ,XX)':>9s} {'Eve knows S':>11s} {'rbar(e)':>8s}")
    for sch, att in cases:
        r = simulate(a.L, a.p, sch, att, seed=a.seed)
        rb = float(rbar(r["e_mixed"]))
        print(f"{sch:9s} {att:11s} {r['e_mixed']:9.3f} {r['e_unmixed']:9.3f} {r['eve_known_frac']:11.3f} {rb:8.3f}")
        rows.append({"schedule": sch, "attack": att, **r, "rbar": rb})
    _write_csv(os.path.join(a.out, "table.csv"), rows)


def cmd_sweep(a):
    fs = np.linspace(0, 1, 11)
    rows = []
    for i, f in enumerate(fs):
        r = simulate(a.L, a.p, "random", "fraction", f=f, seed=a.seed + i)
        rows.append({"f": f, "e": r["e_mixed"], "rbar": float(rbar(r["e_mixed"])),
                     "eve_blind_bits": 2 * (1 - r["eve_known_frac"])})
    rj = simulate(a.L, a.p, "random", "early_J", seed=a.seed + 99)
    _write_csv(os.path.join(a.out, "sweep.csv"), rows)
    print(f"{'f':>5s} {'e':>7s} {'rbar(e)':>8s} {'Eve-blind bits':>15s}")
    for r in rows:
        print(f"{r['f']:5.2f} {r['e']:7.3f} {r['rbar']:8.3f} {r['eve_blind_bits']:15.3f}")
    print(f"early_J: e = {rj['e_mixed']:.3f}, rbar = {float(rbar(rj['e_mixed'])):.3f}, "
          f"Eve-blind bits = {2 * (1 - rj['eve_known_frac']):.3f}")
    _plot_sweep(rows, rj, os.path.join(a.out, "sweep.png"))


def cmd_honest(a):
    """Abort probability of the acceptance event (48)-(49) for an honest i.i.d. channel."""
    rng = np.random.default_rng(a.seed)
    gamma = a.p / 2
    Ls = np.unique(np.logspace(3, 6, 13).astype(int))
    rows = []
    for L in Ls:
        # per-slot categories: test-ok, test-err, test-loss, non-test (i.i.d. honest model)
        probs = [gamma * a.eta_hon * (1 - a.e_hon), gamma * a.eta_hon * a.e_hon,
                 gamma * (1 - a.eta_hon), 1 - gamma]
        cnt = rng.multinomial(L, probs, size=a.trials)
        q_err, q_loss = cnt[:, 1] / L, cnt[:, 2] / L
        accept = (q_loss <= gamma * (1 - a.eta_min)) & (q_err + a.e_tol * q_loss <= gamma * a.e_tol)
        bound = (np.exp(-2 * L * gamma**2 * (a.eta_hon - a.eta_min) ** 2)
                 + np.exp(-2 * L * gamma**2 * a.eta_hon**2 * (a.e_tol - a.e_hon) ** 2))   # Eq. (64)
        rows.append({"L": int(L), "abort_empirical": 1 - accept.mean(), "abort_bound_64": min(1.0, bound)})
    _write_csv(os.path.join(a.out, "honest.csv"), rows)
    print(f"gamma = {gamma}, eta_hon = {a.eta_hon}, eta_min = {a.eta_min}, e_hon = {a.e_hon}, e_tol = {a.e_tol}, trials = {a.trials}")
    print(f"{'L':>8s} {'abort (MC)':>11s} {'bound (64)':>11s}")
    for r in rows:
        print(f"{r['L']:8d} {r['abort_empirical']:11.4f} {r['abort_bound_64']:11.4f}")
    _plot_honest(rows, a.trials, os.path.join(a.out, "honest.png"))


def cmd_memoryless(a):
    rows = []
    print(f"{'pA':>5s} {'pB':>5s} {'mu_hat':>8s} {'mu':>8s} {'gamma_hat':>10s} {'gamma':>8s}")
    for pA, pB in [(0.2, 0.2), (0.2, 0.4), (0.4, 0.4)]:
        r = simulate_memoryless(a.L, pA, pB, seed=a.seed)
        rows.append({"pA": pA, "pB": pB, **r})
        print(f"{pA:5.2f} {pB:5.2f} {r['mu_hat']:8.4f} {r['mu']:8.4f} {r['gamma_hat']:10.4f} {r['gamma']:8.4f}")
    _write_csv(os.path.join(a.out, "memoryless.csv"), rows)

# ----------------------------------------------------------------------------------------------
# 6. Output helpers (figures: light surface, fixed categorical order, thin marks, one axis)
# ----------------------------------------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def _axes(title, xlabel, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5); ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    return plt, fig, ax


def _plot_sweep(rows, rj, path):
    plt, fig, ax = _axes("Certified rate vs. what Eve actually misses (Bell-basis intercept-resend)",
                         "observed mismatch e on C1, C2", "bits per message symbol")
    e = np.linspace(0, 0.75, 200)
    ax.plot(e, rbar(e), color=SERIES[0], lw=1.6, label="certified  r̄(e)  (Eq. 22)")
    ax.plot(e, np.maximum(0, 2 * (1 - 4 * e / 3)), color=SERIES[1], lw=1.6, ls="--",
            label="Eve-blind bits, theory 2(1 − 4e/3)")
    ax.plot([r["e"] for r in rows], [r["eve_blind_bits"] for r in rows], "o", ms=5.5, color=SERIES[1],
            mec=SURFACE, mew=1.2, label="Eve-blind bits, simulated")
    ax.plot([rj["e_mixed"]], [2 * (1 - rj["eve_known_frac"])], "s", ms=6.5, color=SERIES[2], mec=SURFACE, mew=1.2,
            label="J_A announced early (simulated)")
    ax.annotate("e = 0 yet Eve knows S:\nbound needs the timing assumption",
                xy=(rj["e_mixed"], 2 * (1 - rj["eve_known_frac"])), xytext=(0.12, 0.25),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(-0.02, 0.8); ax.set_ylim(-0.08, 2.1)
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)
    print(f"saved {path}")


def _plot_honest(rows, trials, path):
    plt, fig, ax = _axes("Honest-channel abort probability vs. block length",
                         "block length L (slots)", "abort probability")
    L = np.array([r["L"] for r in rows]); emp = np.array([r["abort_empirical"] for r in rows])
    floor = 1.0 / trials
    ax.plot(L, [r["abort_bound_64"] for r in rows], color=SERIES[0], lw=1.6, label="Hoeffding bound (64)")
    nz = emp > 0
    ax.plot(L[nz], emp[nz], "o-", color=SERIES[1], lw=1.4, ms=5.5, mec=SURFACE, mew=1.2,
            label=f"Monte Carlo abort rate ({trials} runs per L)")
    if (~nz).any():
        ax.plot(L[~nz], np.full((~nz).sum(), floor), "v", ms=6, mfc="none", mec=SERIES[1], mew=1.2,
                label=f"no abort observed (rate < 1/{trials})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(floor / 3, 1.6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE); plt.close(fig)
    print(f"saved {path}")

# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selfcheck", "table", "sweep", "honest", "memoryless", "all"])
    ap.add_argument("--L", type=int, default=100_000, help="slots per execution (table, sweep, memoryless)")
    ap.add_argument("--p", type=float, default=0.2, help="control probability (Protocol 2 / 2*)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--trials", type=int, default=4000, help="Monte Carlo runs per L (honest)")
    ap.add_argument("--eta_hon", type=float, default=0.95)
    ap.add_argument("--eta_min", type=float, default=0.90)
    ap.add_argument("--e_hon", type=float, default=0.01)
    ap.add_argument("--e_tol", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cmds = ["selfcheck", "table", "sweep", "honest", "memoryless"] if a.cmd == "all" else [a.cmd]
    for c in cmds:
        print(f"\n=== {c} ===")
        globals()[f"cmd_{c}"](a)


if __name__ == "__main__":
    main()
