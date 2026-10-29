"""Run every verification module, write results/*.json|csv and figures/*.pdf|png, print a pass/fail summary."""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
import m1_optics, m2_quantum, m3_source, m4_receiver, m5_tradeoff, m6_finite, m7_complete, m8_schedule, m9_hash, m10_exact_abort, m11_adversarial
QUICK = '--quick' in sys.argv   # reuse results/m11_adversarial.csv if present (the search takes ~5 min)
from core import r_rate

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5, "axes.linewidth": 0.6,
                     "legend.frameon": False, "legend.fontsize": 7})


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(width=0.6, length=3, colors=INK); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def dump(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


S = {}   # summary: claim -> (pass, evidence string)
t0 = time.time()

# ---------------- M1
o1 = m1_optics.run(); dump("m1_optics", o1)
S["M1 App. A / (17) unitary and adjoint"] = (o1["A_unitary"] < 1e-12 and o1["eq17_equals_adjoint"] < 1e-12, f"max dev {max(o1['A_unitary'], o1['eq17_equals_adjoint']):.1e}")
S["M1 (19) mixed bases MUB with D; ZZ, XX not"] = (o1["overlaps"]["ZX"] == [0.25] and o1["overlaps"]["XZ"] == [0.25] and o1["overlaps"]["ZZ"] == [0.0, 0.5], str(o1["overlaps"]))
S["M1 Lemma III.1 (18) and relation (3)"] = (o1["eq18_maxerr"] < 1e-12 and o1["eq3_relation_holds"], f"max dev {o1['eq18_maxerr']:.1e}")
S["M1 (20) D = J_A xor S"] = (o1["eq20_D_equals_JA_xor_S"], "all 16 (S, J_A)")
S["M1 Sec. II-C / Rem. II.1 average states I/4"] = (max(o1["msg_reduced_state_dev"], o1["ctrl16_avg_dev"], o1["ctrl8_avg_dev"]) < 1e-12, "message, 16-state, 8-state ensembles")

# ---------------- M2
rows, sweep, parity = m2_quantum.run(3000)
df2 = pd.DataFrame(rows); df2.to_csv(os.path.join(RES, "m2_random_attacks.csv"), index=False)
dsw = pd.DataFrame(sweep); dsw.to_csv(os.path.join(RES, "m2_sweeps.csv"), index=False); dump("m2_parity", parity)
act = df2.dropna(subset=["dev_H_actual_virtual"])
S["M2 Prop. III.2 actual = virtual statistics"] = (act[["dev_e_actual_virtual", "dev_pdet"]].values.max() < 1e-10, f"{len(act)} attacks, max dev {act[['dev_e_actual_virtual','dev_pdet']].values.max():.1e}")
S["M2 Lemma III.3 H(S|J_A E,det) = H(D|E,det)"] = (act.dev_H_actual_virtual.max() < 1e-10, f"max dev {act.dev_H_actual_virtual.max():.1e}")
S["M2 H(S|E,det) = 2 without J_A (no-signalling)"] = (np.abs(act.H_S_given_E - 2).max() < 1e-10, f"max dev {np.abs(act.H_S_given_E - 2).max():.1e}")
S["M2 (25)/(27)/Prop. V.1 H(D|E,det) >= r(e_pool)"] = (df2.gap_pool.min() > -1e-10, f"{len(df2)} attacks, min gap {df2.gap_pool.min():.2e}")
rx = df2[df2.family.str.contains("rx")]
S["M2 Prop. V.3 receiver misalignment"] = (rx.gap_pool.min() > -1e-10, f"{len(rx)} attacks, min gap {rx.gap_pool.min():.2e}")
bd = dsw[dsw.family == "bell_dephasing"]
S["M2 tightness: Bell dephasing attains r(e)"] = (np.abs(bd.gap_pool).max() < 1e-10, f"max |gap| {np.abs(bd.gap_pool).max():.1e} on 16 points")
S["M2 counterexample: ZZ/XX test invalid"] = (parity["ZZ"]["gap_ZZ"] < -0.99 and parity["XX"]["gap_XX"] < -0.99,
                                              f"parity attack e_ZZ=0, H=1 < r(0)=2; random attacks min gap_ZZ {df2.gap_ZZ.min():.3f}")

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
ee = np.linspace(0, 0.8, 400)
for ax, col, lab in ((axs[0], "e_pool", r"pooled mismatch $e$ on $\mathcal{C}_1,\mathcal{C}_2$"), (axs[1], "eZZ", r"mismatch on $Z_P\otimes Z_S$")):
    ax.scatter(df2[col], df2.H, s=4, color=BLUE, alpha=0.35, lw=0, label="random single-round attacks")
    ax.plot(ee, r_rate(ee), color=INK, lw=1.4, label=r"$r(e)$, Eq. (22)")
    style(ax); ax.set_xlabel(lab); ax.set_xlim(0, 0.8); ax.set_ylim(-0.05, 2.08)
axs[0].plot(bd.e_pool, bd.H, "s", ms=3.6, mfc=ORANGE, mec="white", mew=0.5, label="Bell-dephasing attack")
axs[1].plot([0], [1.0], "D", ms=5, mfc=ORANGE, mec="white", mew=0.6, label="parity attack")
axs[0].set_ylabel(r"$H(D|E',\mathrm{det})$  [bits]")
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
fig.legend(h0 + h1[-1:], l0 + l1[-1:], loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(bottom=0.27)
axs[0].set_title("(a) secrecy test used in the paper", fontsize=8, loc="left"); axs[1].set_title("(b) unmixed basis as test (invalid)", fontsize=8, loc="left")
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_m2_uncertainty.{ext}"), dpi=300)
plt.close(fig)

# ---------------- M3
df3 = pd.DataFrame(m3_source.run(1500)); df3.to_csv(os.path.join(RES, "m3_source.csv"), index=False)
S["M3 Prop. V.4 H(D~|E) >= r_q(e)"] = (df3.gap.min() > -1e-10, f"{len(df3)} cases, min gap {df3.gap.min():.2e}")
S["M3 Remark V.5 bound (45)"] = (df3.slack45.min() > -1e-12, f"min slack {df3.slack45.min():.3f}")
neg = (df3.gap_ideal_q < 0).sum()
S["M3 q_src correction is necessary"] = (neg > 0, f"{neg} cases violate the uncorrected r(e) (min {df3.gap_ideal_q.min():.3f})")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
b_q = df3.H - df3.gap; b_2 = df3.H - df3.gap_ideal_q
ax.scatter(b_2, df3.H, s=4, color=ORANGE, alpha=0.35, lw=0, label=r"uncorrected bound $r(e)$")
v = df3.gap_ideal_q < 0
ax.scatter(b_2[v], df3.H[v], s=9, marker="x", color=INK, lw=0.7, label=r"violations of $r(e)$")
ax.scatter(b_q, df3.H, s=4, color=BLUE, alpha=0.4, lw=0, label=r"calibrated bound $r_{q_{\rm src}}(e)$")
ax.plot([0, 2], [0, 2], color=INK, lw=1, label="bound = entropy")
style(ax); ax.set_xlabel("lower bound [bits]"); ax.set_ylabel(r"$H(\tilde D|E')$ [bits]"); ax.legend(loc="lower right")
save(fig, "fig_m3_source")

# ---------------- M4
o4 = m4_receiver.run(); dump("m4_receiver", o4)
th = o4["thinning"]["mc"]
spread_raw = np.ptp([v["p_click_raw"] for v in th.values()]); spread_th = np.ptp([v["p_click_thinned"] for v in th.values()])
S["M4 Prop. V.2 thinning (33)"] = (o4["thinning"]["povm_dev"] < 1e-12 and spread_th < 5e-3,
                                   f"POVM dev {o4['thinning']['povm_dev']:.1e}; click-prob spread {spread_raw:.3f} -> {spread_th:.4f} (eta0={o4['thinning']['eta0']:.3f})")
dk = pd.DataFrame(o4["dark"])
S["M4 (35)-(36) dark estimators"] = (np.all(np.abs(dk.e_hat - dk.e_sig) < 4 * dk.e_sd + 1e-4), "unbiased within 4 s.d.; " + ", ".join(f"pdc={r.pdc:g}: e_sig {r.e_sig:.3f} vs {r.e_hat:.4f}" for r in dk.itertuples()))
ct = pd.DataFrame(o4["crosstalk"])
S["M4 (40) cross-talk relation"] = (np.abs(ct.e_obs - ct.e_eq40).max() < 1e-12 and np.abs(ct.e_mc - ct.e_eq40).max() < 3e-3, f"exact dev {np.abs(ct.e_obs - ct.e_eq40).max():.1e}, MC dev {np.abs(ct.e_mc - ct.e_eq40).max():.1e}")
S["M4 LP (39) exact = inversion; box gives upward margin"] = (np.abs(ct.lp_exact - ct.e_inverted).max() < 1e-6 and (ct.lp_box_1e2 >= ct.lp_exact - 1e-9).all(), f"max |LP-inv| {np.abs(ct.lp_exact - ct.e_inverted).max():.1e}")

# ---------------- M5
o5 = m5_tradeoff.run(); dump("m5_tradeoff", o5)
cs = pd.DataFrame(o5["cases"])
S["M5 (47) global tangent; beta-Lambda<=0; beta<=3Lambda/4"] = (o5["min_r_minus_tangent"] > -1e-12 and o5["max_beta_minus_lambda"] < 0 and o5["max_beta_minus_3lambda_over_4"] < 0,
                                                              f"min r-tangent {o5['min_r_minus_tangent']:.1e}, max(beta-Lambda) {o5['max_beta_minus_lambda']:.3f}, max(beta-3Lambda/4) {o5['max_beta_minus_3lambda_over_4']:.1e}")
S["M5 Cor. VI.3 r_q tangent"] = (o5["min_rq_minus_tangent"] > -1e-12, f"min {o5['min_rq_minus_tangent']:.1e}")
S["M5 App. B range = mu Lambda; (52) variance bound"] = (np.allclose(cs.vertex_range, cs.mu_lambda) and np.allclose(cs.range_sigma, cs.mu_lambda) and (cs.var_max <= cs.var_bound).all(),
                                                        "max Var / bound = " + ", ".join(f"{v:.3f}" for v in cs.var_max / cs.var_bound))
S["M5 (55)/(74) worst accepted value"] = (np.allclose(cs.worst_accept, cs["claim"], atol=1e-10), f"max dev {np.abs(cs.worst_accept - cs['claim']).max():.1e}")

# ---------------- M6
T1 = pd.DataFrame(m6_finite.table1()); T2 = pd.DataFrame(m6_finite.table2())
T1.to_csv(os.path.join(RES, "m6_table1.csv"), index=False); T2.to_csv(os.path.join(RES, "m6_table2.csv"), index=False)
paper1 = [(0.564, 0.077, 0.397), (0.584, 0.100, 0.377), (0.547, 0.101, 0.355), (1.259, 0.713, 0.209), (1.236, 0.717, 0.199), (1.126, 0.668, 0.188)]
paper2 = [(1.074, 0.119, 0.601, 0.210), (0.756, 0.142, 0.382, 0.241), (0.449, 0.182, 0.184, 0.291), (0.163, 0.268, 0.021, 0.388)]
ok1 = all(abs(round(r.rate_2s, 3) - a) < 1.5e-3 and abs(round(r.rate_2ss, 3) - b) < 1.5e-3 and abs(round(r.p_2ss, 3) - c) < 1.5e-3 for r, (a, b, c) in zip(T1.itertuples(), paper1))
ok2 = all(abs(round(r.rate_2s, 3) - a) < 1.5e-3 and abs(round(r.p_2s, 3) - b) < 1.5e-3 and abs(round(r.rate_2ss, 3) - c) < 1.5e-3 and abs(round(r.p_2ss, 3) - d) < 1.5e-3 for r, (a, b, c, d) in zip(T2.itertuples(), paper2))
S["M6 Table I reproduced"] = (ok1, "18 entries"); S["M6 Table II reproduced"] = (ok2, "16 entries")
sb, free = m6_finite.symmetric_bias(); dump("m6_symmetric_bias", dict(scan=sb, free=free))
S["M6 Prop. VII.1 symmetric optimum"] = (all(abs(r["pA_best"] - r["pA_sym"]) < 0.01 for r in sb) and all(abs(f["pA"] - f["pB"]) < 1e-4 for f in free),
                                         "free 2-D optimum (pA, pB) = " + ", ".join(f"({f['pA']:.4f}, {f['pB']:.4f})" for f in free))
sc = m6_finite.scaling(); dump("m6_scaling", sc)
S["M6 Prop. VIII.1 exponents (88)-(89)"] = (all(abs(sc[k]["slope_p"] - sc[k]["pred_p"]) < 0.02 and abs(sc[k]["slope_deficit"] - sc[k]["pred_deficit"]) < 0.02 and abs(sc[k]["slope_alpha"] - sc[k]["pred_alpha"]) < 0.02 for k in sc),
                                            "; ".join(f"{k}: p {sc[k]['slope_p']:.3f}/{sc[k]['pred_p']:.3f}, alpha-1 {sc[k]['slope_alpha']:.3f}/{sc[k]['pred_alpha']:.3f}, deficit {sc[k]['slope_deficit']:.3f}/{sc[k]['pred_deficit']:.3f}" for k in sc))
et = pd.DataFrame(m6_finite.et_sensitivity()); et.to_csv(os.path.join(RES, "m6_et_sensitivity.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
for k, c, mk in (("2*", BLUE, "o"), ("2**", ORANGE, "s")):
    Ls = np.array([r["L"] for r in sc[k]["rows"]])
    axs[0].loglog(Ls, [r["p"] for r in sc[k]["rows"]], mk, ms=3.5, color=c, label=f"Protocol ${k[0]}^{{{k[1:].replace('*', chr(92) + 'ast ')}}}$, slope {sc[k]['slope_p']:.3f}")
    axs[1].loglog(Ls, [r["deficit"] for r in sc[k]["rows"]], mk, ms=3.5, color=c, label=f"full bound, slope {sc[k]['slope_deficit']:.3f}")
    axs[1].loglog(Ls, sc[k]["lead_deficit"], color=c, lw=1, label=f"leading order (87), ${k[0]}^{{{k[1:].replace('*', chr(92) + 'ast ')}}}$")
axs[0].set_ylabel(r"optimal control probability $p_{\rm opt}$"); axs[1].set_ylabel(r"deficit $r(e_{\rm tol}) - H_{\min}^{\varepsilon}/L$")
for ax in axs: style(ax); ax.set_xlabel("block length $L$"); ax.legend(loc="lower left")
axs[0].set_title(r"(a) predicted slopes $-1/3$, $-1/4$", fontsize=8, loc="left"); axs[1].set_title("(b) rate deficit", fontsize=8, loc="left")
save(fig, "fig_m6_scaling")

# ---------------- M7
c7, common = m7_complete.run(); d7 = pd.DataFrame(c7); d7.to_csv(os.path.join(RES, "m7_completeness.csv"), index=False)
S["M7 (69)/(75)-(76) valid upper bounds"] = (((d7.abort_mc <= d7.abort_bound + 0.01) & (d7.abort_mc <= d7.abort_bernstein + 0.01)).all(), "MC abort <= Hoeffding and <= Bernstein at all L")
# ---------------- M10 exact abort probability
d10 = pd.DataFrame(m10_exact_abort.run()); d10.to_csv(os.path.join(RES, "m10_exact_abort.csv"), index=False)
x = d7.merge(d10, on=["case", "L"])
sig = np.sqrt(np.maximum(x.abort_exact * (1 - x.abort_exact), 1 / 4000**2) / 4000)
S["M10 exact abort agrees with MC"] = ((np.abs(x.abort_mc - x.abort_exact) <= 4 * sig + 1.0 / 4000).all(), f"max |MC-exact|/sigma {(np.abs(x.abort_mc - x.abort_exact) / sig).max():.2f}")
valid = d10.log10_abort_exact > -300
S["M10 Hoeffding (69) and Bernstein >= exact"] = (((d10.abort_hoeffding >= d10.abort_exact) & (d10.abort_bernstein >= d10.abort_exact)).all(),
                                                  f"{valid.sum()} points with exact value above 1e-300")
from scipy.optimize import brentq
from m7_complete import bound as hoef, bernstein as bern
L3 = {}
for name, g, pdc in (("2*", 0.055, 0.0), ("2**", 0.02, 0.0)):
    f_ex = lambda L: m10_exact_abort.log_abort(L, g, **common, pdc=pdc) - np.log(1e-3)
    f_ho = lambda L: np.log(hoef(L, g, **common, pdc=pdc)) - np.log(1e-3)
    f_be = lambda L: np.log(bern(L, g, **common, pdc=pdc)) - np.log(1e-3)
    L3[name] = dict(exact=brentq(f_ex, 1e3, 1e9, xtol=10), hoeffding=brentq(f_ho, 1e3, 1e11), bernstein=brentq(f_be, 1e3, 1e11))
dump("m10_L_at_1e-3", L3)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
sub = d7[d7.case == "2*, p=0.11, no dark"]; ex = d10[(d10.case == "2*, p=0.11, no dark") & (d10.log10_abort_exact > -300)]
nz = (sub.abort_mc > 0).values
ax.loglog(ex.L, ex.abort_exact, color=INK, lw=1.4, label="exact (binomial tails)")
ax.loglog(sub.L[nz], sub.abort_mc[nz], "o", ms=3.8, color=INK, label="Monte Carlo (4000 runs)")
ax.loglog(sub.L, sub.abort_bound, color=BLUE, lw=1.6, label="Hoeffding bound (69)")
ax.loglog(sub.L, np.maximum(sub.abort_bernstein, 1e-300), color=ORANGE, lw=1.6, ls="--", label="Bernstein bound")
ax.set_ylim(1e-30, 2); ax.set_xlim(8e3, 1.2e8); style(ax); ax.set_xlabel("block length $L$"); ax.set_ylabel("abort probability (honest channel)")
ax.legend(loc="center", bbox_to_anchor=(0.66, 0.5), fontsize=6.3)
save(fig, "fig_m7_completeness")

# ---------------- M8
d8 = pd.DataFrame(m8_schedule.adaptive()); d8.to_csv(os.path.join(RES, "m8_adaptive.csv"), index=False)
d8s = pd.DataFrame(m8_schedule.sifting()); d8s.to_csv(os.path.join(RES, "m8_sifting.csv"), index=False)
orig = d8[d8.schedule == "original"]; rnd = d8[d8.schedule == "random"]
S["M8 original schedule over-certifies"] = ((orig.certified > orig.ceiling).all(), f"e=0 while kappa=p, {len(orig)} values of p")
S["M8 Protocol 2* detects the same attack"] = ((rnd.certified <= rnd.ceiling).all() and np.abs(rnd.e - rnd.e_exact).max() < 0.01, f"max |e - 3p/4| {np.abs(rnd.e - rnd.e_exact).max():.4f}")
S["M8 early J_A counterexample"] = (d8.iloc[-1].kappa > 0.99 and d8.iloc[-1].e < 1e-9, "e = 0, kappa = 1")
S["M8 sifting rates (11), (12), Rem. II.1"] = ((np.abs(d8s.mu_mc - d8s.mu) < 2e-3).all() and (np.abs(d8s.gamma_mc - d8s.gamma) < 2e-3).all(), f"max dev {max(np.abs(d8s.mu_mc - d8s.mu).max(), np.abs(d8s.gamma_mc - d8s.gamma).max()):.1e}")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
pp = np.linspace(0.02, 0.5, 200)
ax.fill_between(pp, 2 * (1 - pp), 2.0, color=BLUE, alpha=0.10, lw=0)
ax.plot(pp, 2 * (1 - pp), color=INK, lw=1.4, label=r"ceiling $2(1-\kappa)$, both schedules")
ax.plot(pp, np.full_like(pp, 2.0), color=BLUE, lw=2, ls="--", label=r"$r(e)$, original schedule")
ax.plot(pp, r_rate(0.75 * pp), color=ORANGE, lw=2, label=r"$r(e)$, Protocol $2^\ast$")
ax.plot(orig.p, orig.certified, "o", ms=4, mfc=BLUE, mec="white"); ax.plot(rnd.p, rnd.certified, "o", ms=4, mfc=ORANGE, mec="white")
ax.plot(d8.p[:16], d8.ceiling[:16], "s", ms=3.2, mfc=INK, mec="white")
ax.text(0.31, 1.8, "over-certification", color=BLUE, fontsize=7.5, ha="center")
style(ax); ax.set_xlim(0, 0.52); ax.set_ylim(0, 2.1); ax.set_xlabel("control probability $p$"); ax.set_ylabel("bits per message symbol"); ax.legend(loc="lower left")
save(fig, "fig_m8_adaptive")

# ---------------- M9
d9 = pd.DataFrame(m9_hash.run()); d9.to_csv(os.path.join(RES, "m9_hash.csv"), index=False)
S["M9 leftover hash (65), classical toy"] = ((d9.d_avg <= d9.bound + 1e-12).all(), f"{len(d9)} (leakage, l) pairs, max d/bound {(d9.d_avg / d9.bound).max():.3f}")

# ---------------- M11 adversarial search
p11 = os.path.join(RES, "m11_adversarial.csv")
d11 = pd.read_csv(p11) if (QUICK and os.path.exists(p11)) else pd.DataFrame(m11_adversarial.run())
d11.to_csv(p11, index=False)
mx = d11[d11.test == "mixed"]; zz = d11[d11.test == "ZZ"]
S["M11 adversarial search: no violation of (27)"] = (mx.gap.min() > -1e-6, f"{len(mx)} optimised attacks, min gap {mx.gap.min():.1e}")
S["M11 optimum reaches r(e) for dim E' >= 4"] = (mx[mx.dE >= 4].gap.max() < 1e-4, f"max optimised gap {mx[mx.dE >= 4].gap.max():.1e} (dE=4, 8); dE=2 stays >= {mx[mx.dE == 2].gap.min():.3f}")
S["M11 control: same search breaks the ZZ test"] = (zz.gap.max() < -0.5, f"optimised gaps {zz.gap.min():.3f} to {zz.gap.max():.3f}")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for dE, c, mk in ((2, AQUA, "^"), (4, BLUE, "o"), (8, ORANGE, "s")):
    t = mx[mx.dE == dE]
    ax.plot(t.e, t.gap, mk, ms=4, mfc=c, mec="white", mew=0.5, ls="none", label=rf"pooled $\mathcal{{C}}_1,\mathcal{{C}}_2$ test, $\dim E'={dE}$")
ax.plot(zz.e, zz.gap, "D", ms=4, mfc="none", mec=INK, mew=0.8, ls="none", label=r"control: $Z_P\otimes Z_S$ test")
ax.axhline(0, color=INK, lw=0.8)
style(ax); ax.set_xlabel("mismatch $e$ at the optimum"); ax.set_ylabel(r"minimised gap $H(D|E',\mathrm{det})-r(e)$")
ax.legend(loc="lower right", fontsize=6.3)
save(fig, "fig_m11_adversarial")

# ---------------- summary
summ = [dict(check=k, passed=bool(v[0]), evidence=v[1]) for k, v in S.items()]
pd.DataFrame(summ).to_csv(os.path.join(RES, "summary.csv"), index=False); dump("summary", summ)
for s in summ:
    print(("PASS " if s["passed"] else "FAIL ") + s["check"] + "  |  " + s["evidence"])
print(f"{sum(s['passed'] for s in summ)}/{len(summ)} passed in {time.time() - t0:.0f}s")
