"""Figures for Appendix D (Numerical checks). Reads the CSV/JSON outputs of the verification
package (M2, M8, M10) and writes figD1_schedule.pdf, figD2_uncertainty.pdf, figD3_completeness.pdf."""
import json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# usage: python make_figs.py [results_dir] [output_dir]
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "verification", "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6, "savefig.bbox": "tight"})


def rbar(e):
    e = np.asarray(e, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -e * np.log2(e) - (1 - e) * np.log2(1 - e)
    h = np.nan_to_num(h)
    r = 2 - h - e * np.log2(3)
    return np.where(e <= 0.75, np.maximum(r, 0), 0.0)


# ---------------------------------------------------------------- Fig. D1: adaptive schedule attack
d = pd.read_csv(f"{RES}/m8_adaptive.csv")
o, r = d[d.schedule == "original"], d[d.schedule == "random"]
early = d[d.schedule.str.contains("early")].iloc[0]
pp = np.linspace(0.04, 0.51, 100)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
a = ax[0]
a.plot(pp, 0 * pp, color=ORANGE, lw=1.0, ls="--")
a.plot(pp, 0.75 * pp, color=BLUE, lw=1.0, ls="--")
a.plot(o.p, o.e, "o", color=ORANGE, ms=5, mec="white", mew=0.8, label="original schedule (Protocol 2)")
a.plot(r.p, r.e, "s", color=BLUE, ms=5, mec="white", mew=0.8, label=r"random slots (Protocol $2^\ast$)")
a.set_xlabel(r"control probability $p$"); a.set_ylabel(r"observed pooled mismatch $e$")
a.set_title("(a) test statistic", fontsize=9, loc="left")
a.annotate(r"$e=3p/4$", (0.42, 0.315), color=MUTED, fontsize=8)
a.annotate(r"$e=0$", (0.42, 0.015), color=MUTED, fontsize=8)
a.legend(loc="upper left", frameon=False)
b = ax[1]
b.plot(pp, 2 * (1 - pp), color=INK, lw=1.2, ls="--", label=r"Eve-blind entropy $2(1-\kappa)$, $\kappa=p$")
b.plot(o.p, o.certified, "o-", color=ORANGE, ms=5, mec="white", mew=0.8, label=r"certified $\bar r(e)$, original")
b.plot(r.p, r.certified, "s-", color=BLUE, ms=5, mec="white", mew=0.8, label=r"certified $\bar r(e)$, random slots")
b.fill_between(o.p, 2 * (1 - o.p), o.certified, color=ORANGE, alpha=0.12, lw=0)
b.set_xlabel(r"control probability $p$"); b.set_ylabel("bits per message round")
b.set_ylim(0, 2.15); b.set_title("(b) certified rate vs. ceiling", fontsize=9, loc="left")
b.legend(loc="lower left", frameon=False)
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/figD1_schedule.pdf"); fig.savefig(f"{OUT}/figD1_schedule.png", dpi=160)

# ---------------------------------------------------------------- Fig. D2: single-round bound, mixed vs unmixed test
ra = pd.read_csv(f"{RES}/m2_random_attacks.csv")
sw = pd.read_csv(f"{RES}/m2_sweeps.csv")
par = json.load(open(f"{RES}/m2_parity.json"))["ZZ"]
ee = np.linspace(0, 1, 400)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
a = ax[0]
a.plot(ra.e_pool, ra.H, ".", color=BLUE, ms=2.2, alpha=0.45, label="3000 random attacks")
bd = sw[sw.family == "bell_dephasing"]; ic = sw[sw.family == "intercept"]
a.plot(ic.e_pool, ic.H, "^", color=AQUA, ms=4.5, mec="white", mew=0.6, label="Bell intercept-resend")
a.plot(bd.e_pool, bd.H, "o", color=ORANGE, ms=4.5, mec="white", mew=0.6, label="Bell dephasing")
a.plot(ee, rbar(ee), color=INK, lw=1.3, label=r"bound $\bar r(e)$")
a.set_xlabel(r"pooled mixed-basis mismatch $e$"); a.set_ylabel(r"$H(D|E',\mathrm{det})$ [bits]")
a.set_title(r"(a) mixed bases $\mathcal{C}_1,\mathcal{C}_2$", fontsize=9, loc="left")
a.legend(loc="center right", bbox_to_anchor=(1.0, 0.36), frameon=False)
b = ax[1]
viol = ra.gap_ZZ < -1e-9
b.plot(ra.eZZ[~viol], ra.H[~viol], ".", color=BLUE, ms=2.2, alpha=0.45, label="bound holds")
b.plot(ra.eZZ[viol], ra.H[viol], ".", color=ORANGE, ms=2.2, alpha=0.6, label=f"bound violated ({viol.sum()} of {len(ra)})")
b.plot([par["eZZ"]], [par["H"]], "D", color=ORANGE, ms=6, mec=INK, mew=0.8, label=r"parity attack")
b.plot(ee, rbar(ee), color=INK, lw=1.3, label=r"$\bar r(e_{ZZ})$")
b.set_xlabel(r"unmixed-basis mismatch $e_{ZZ}$")
b.set_title(r"(b) unmixed basis $\mathcal{B}_{ZZ}$ used as a test", fontsize=9, loc="left")
b.legend(loc="lower right", bbox_to_anchor=(1.03, 0.10), frameon=False)
for x in ax: x.set_xlim(-0.02, 1.0); x.set_ylim(-0.05, 2.1)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/figD2_uncertainty.pdf"); fig.savefig(f"{OUT}/figD2_uncertainty.png", dpi=160)

# ---------------------------------------------------------------- Fig. D3: completeness
c = pd.read_csv(f"{RES}/m10_exact_abort.csv")
cases = [("2*, p=0.11, no dark", r"(a) Protocol $2^\ast$, $\gamma=0.055$"),
         ("2**, p=0.2, no dark", r"(b) Protocol $2^{\ast\ast}$, $\gamma=0.02$")]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for x, (name, title) in zip(ax, cases):
    s = c[c.case == name]
    x.loglog(s.L, s.abort_hoeffding, "^-", color=AQUA, ms=4, mec="white", mew=0.6, label=r"Hoeffding bound (69)")
    x.loglog(s.L, s.abort_bernstein, "s-", color=ORANGE, ms=4, mec="white", mew=0.6, label="Bernstein bound")
    x.loglog(s.L, np.maximum(s.abort_exact, 1e-300), "o-", color=BLUE, ms=4, mec="white", mew=0.6, label="exact multinomial tail")
    x.axhline(1e-3, color=MUTED, lw=0.9, ls=":")
    x.set_ylim(1e-12, 2); x.set_xlim(1e4, 1e8)
    x.set_xlabel(r"block length $L$"); x.set_title(title, fontsize=9, loc="left")
ax[0].set_ylabel(r"honest abort probability")
ax[0].annotate(r"$10^{-3}$", (1.2e4, 1.6e-3), color=MUTED, fontsize=8)
ax[0].legend(loc="lower left", frameon=False)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/figD3_completeness.pdf"); fig.savefig(f"{OUT}/figD3_completeness.png", dpi=160)

# ---------------------------------------------------------------- numbers quoted in the text
m11 = pd.read_csv(f"{RES}/m11_adversarial.csv")
Lj = json.load(open(f"{RES}/m10_L_at_1e-3.json"))
num = {
    "n_random": int(len(ra)), "min_gap_pool": float(ra.gap_pool.min()),
    "n_viol_ZZ": int(viol.sum()), "n_viol_XX": int((ra.gap_XX < -1e-9).sum()),
    "max_dev_actual_virtual": float(ra[["dev_H_actual_virtual", "dev_e_actual_virtual", "dev_pdet"]].abs().max().max()),
    "bd_max_abs_gap": float(bd.gap_pool.abs().max()), "ic_max_abs_gap": float(ic.gap_pool.abs().max()),
    "m11_min_gap_mixed": m11[m11.test == "mixed"].groupby("dE").gap.min().to_dict(),
    "m11_min_gap_ZZ": m11[m11.test == "ZZ"].groupby("dE").gap.min().to_dict(),
    "L_at_1e-3": Lj,
    "adaptive_original_e_max": float(o.e.max()), "adaptive_random_e": r[["p", "e"]].values.tolist(),
    "early_J": {"e": float(early.e), "kappa": float(early.kappa)},
}
json.dump(num, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print(json.dumps(num, indent=1, default=float))
