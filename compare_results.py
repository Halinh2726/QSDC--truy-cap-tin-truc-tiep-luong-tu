"""Compare a fresh run (verification/results) with the reference results shipped in reference_results/.

All random draws use fixed seeds, so with the same library versions the numbers agree to rounding.
With other numpy/scipy versions the random streams or optimizer paths can change; the script then
reports the largest deviation per file so that the user can judge it against the tolerances below.
Usage: python compare_results.py [new_results_dir] [reference_dir]
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "verification", "results")
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "reference_results", "verification_results")
RTOL, ATOL = 1e-6, 1e-9


def flat(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{pre}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, o


bad = 0
for f in sorted(os.listdir(REF)):
    a, b = os.path.join(NEW, f), os.path.join(REF, f)
    if not os.path.exists(a):
        print(f"MISSING  {f}"); bad += 1; continue
    if f.endswith(".csv"):
        x, y = pd.read_csv(a), pd.read_csv(b)
        if x.shape != y.shape:
            print(f"SHAPE    {f}: {x.shape} vs {y.shape}"); bad += 1; continue
        num = y.select_dtypes("number").columns
        xa, ya = x[num].to_numpy(float), y[num].to_numpy(float)
        same = (xa == ya) | (np.isnan(xa) & np.isnan(ya))          # also covers equal +-inf entries
        with np.errstate(invalid="ignore"):
            d = np.where(same, 0.0, np.abs(xa - ya))
        s = ATOL + RTOL * np.abs(ya)
        dev = float(np.nanmax(d)) if d.size else 0.0
        ok = bool(np.all(same | (d <= s)))
    else:
        fx, fy = dict(flat(json.load(open(a)))), dict(flat(json.load(open(b))))
        dev, ok = 0.0, set(fx) == set(fy)
        for k, v in fy.items():
            w = fx.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(w, (int, float)):
                dd = abs(w - v); dev = max(dev, dd); ok &= dd <= ATOL + RTOL * abs(v)
            else:
                ok &= (w == v)
    bad += not ok
    print(f"{'OK      ' if ok else 'DIFFERS '} {f:28s} max abs deviation {dev:.2e}")
print("\nAll files agree with the reference." if bad == 0 else f"\n{bad} file(s) differ; see the README, section 'Tolerances'.")
