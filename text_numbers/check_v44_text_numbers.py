"""Numbers added to the main text of v4_4 that are not produced by run_all.py.

1. Table I, new column p_opt^(2*): read from verification/results/m6_table1.csv (column p_2s).
2. Sec. VI-C, sentence after (69): for an honest loss-free device with e_hon = e_tol/2 and the
   parameters of Table I (Protocol 2*, gamma = p_opt/2), compare
     - the second term of (69):  exp(-2 L gamma^2 (e_tol - e_hon)^2)   (eta_hon = 1)
     - the exact probability that (54) is violated:  Pr[Bin(L, gamma e_hon) > floor(L gamma e_tol)]
   (with eta_min = 1 and no loss, (53) holds with probability one).
Usage: python check_v44_text_numbers.py [path/to/m6_table1.csv]
"""
import math, os, sys
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "verification", "results", "m6_table1.csv")
t = pd.read_csv(csv)

print("Table I, column p_opt^(2*) (rounded to 3 decimals as printed in the manuscript)")
for _, r in t.iterrows():
    print(f"  L = {int(r.L):>8d}  e_tol = {r.etol:.2f}  p_opt(2*) = {r.p_2s:.4f} -> {r.p_2s:.3f}")

print("\nSec. VI-C: Hoeffding second term of (69) vs exact binomial tail of (54); e_hon = e_tol/2")
hoef, exact = [], []
for _, r in t.iterrows():
    L, et = int(r.L), float(r.etol)
    p = round(float(r.p_2s), 3)          # value printed in Table I
    g, eh = p / 2, et / 2
    k = math.floor(L * g * et)
    ex = binom.sf(k, L, g * eh)
    hf = math.exp(-2 * L * g * g * (et - eh) ** 2)
    hoef.append(hf); exact.append(ex)
    print(f"  L = {L:>8d}  e_tol = {et:.2f}  gamma = {g:.4f}  Hoeffding term = {hf:.3f}  exact tail = {ex:.2e}")
print(f"\n  Hoeffding range: {min(hoef):.2f} .. {max(hoef):.2f}   (text: 0.04 .. 0.91)")
print(f"  largest exact tail: {max(exact):.1e}   (text: below 1e-13)")
ok = abs(min(hoef) - 0.04) < 0.005 and abs(max(hoef) - 0.91) < 0.005 and max(exact) < 1e-13
print("  PASS" if ok else "  CHECK: values differ from the text")
