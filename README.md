# Numerical verification code for QSDC v4_4

Code accompanying the manuscript *Finite-size entropic security for random-slot hyperentangled direct
communication* (version v4_4). The code checks numerically the analytic statements of the manuscript and
produces every number and figure that is computed rather than proved: Tables I and II, the Table I
column p_opt^(2*), the completeness numbers in Sec. VI-C, and the three figures of Appendix D
(Numerical checks). No step of a proof depends on this code.

## Contents

| Folder / file | Role |
|---|---|
| `verification/` | Modules M1-M11 (`m1_optics.py` ... `m11_adversarial.py`), shared routines `core.py`, slot-level simulator `qsdc_slot_sim.py`, and the driver `run_all.py` |
| `appendixD/make_figs.py` | Builds Figs. D1-D3 of Appendix D from `verification/results` |
| `text_numbers/check_v44_text_numbers.py` | Recomputes the numbers added to the v4_4 text (Table I column p_opt^(2*); Sec. VI-C, sentence after (69)) |
| `compare_results.py` | Compares a new run with `reference_results/` |
| `reference_results/` | Results of the reference run: `verification_results/` (all CSV/JSON of M1-M11), `appendixD_numbers.json`, `figD*.pdf` |
| `docs/QSDC_numerical_verification_companion_v3.pdf` | Companion document: purpose, scenario, method and flowchart of every module |
| `run_everything.sh` | Runs all steps in order |
| `requirements.txt` | Library versions of the reference run |

## Installation

```bash
python3 -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```
Python 3.10 or newer with numpy, scipy, pandas and matplotlib is sufficient. The pinned versions are those
of the reference run; other versions also work, but random streams and optimizer paths may then differ
slightly (see *Tolerances*).

## Running

```bash
bash run_everything.sh            # full run, about 10-12 min on one CPU core
bash run_everything.sh --quick    # about 1-2 min; reuses the M11 adversarial search of the reference run
```
On Windows, run the four steps of `run_everything.sh` by hand:
```bash
cd verification && python run_all.py && cd ..
python appendixD/make_figs.py verification/results appendixD/figures
python text_numbers/check_v44_text_numbers.py verification/results/m6_table1.csv
python compare_results.py
```

Expected output:
1. `run_all.py` prints a table of 38 claims, each with PASS/FAIL and the numerical evidence, and ends
   with `38/38 passed in ...s`. Results go to `verification/results/`, figures to `verification/figures/`.
2. `appendixD/figures/figD1_schedule.pdf`, `figD2_uncertainty.pdf`, `figD3_completeness.pdf` and
   `appendixD/figures/numbers.json` with all numbers quoted in Appendix D.
3. `check_v44_text_numbers.py` prints the Table I column and the Sec. VI-C values and ends with `PASS`.
4. `compare_results.py` prints one line per result file and ends with
   `All files agree with the reference.`

## Map from modules to the manuscript (v4_4 numbering)

| Module | Statements checked | Appendix D |
|---|---|---|
| M1 `m1_optics` | Analyzer (17) and App. A; MUB relation (19); Lemma III.1 (18) and (3); D = J_A xor S (20); average states I_4/4 (Sec. II-C, Remark II.1) | - |
| M2 `m2_quantum` | Prop. III.2; Lemma III.3; H(S|E',det) = 2 without J_A; (25), (27), Prop. V.1; Prop. V.3; tightness (Bell dephasing); ZZ/XX counterexample | D2, D3 |
| M3 `m3_source` | Prop. V.4, Remark V.5 (45); necessity of the q_src correction | - |
| M4 `m4_receiver` | Prop. V.2 (33); dark-count estimators (35)-(36); cross-talk (40); LP (39) | - |
| M5 `m5_tradeoff` | Tangent (47); Cor. VI.3; App. B range and (52); accepted values (55), (74) | - |
| M6 `m6_finite` | Tables I and II; Prop. VII.1; exponents (88)-(89); sensitivity to the tangent point | - |
| M7 `m7_complete` | (69), (75)-(76) are valid upper bounds (Monte Carlo) | - |
| M8 `m8_schedule` | Adaptive attack on the original schedule; Protocol 2*; early-J_A case; sifting rates (11), (12) | D1 |
| M9 `m9_hash` | Leftover hashing (65), classical example | - |
| M10 `m10_exact_abort` | Exact honest abort probability; Hoeffding (69) and Bernstein bounds versus exact | D4 |
| M11 `m11_adversarial` | Adversarial search: no violation of (27); optimum reaches r(e) for dim E' >= 4; control search breaks the ZZ test | D2, D3 |

## Tolerances

All random draws use fixed seeds. With the library versions of `requirements.txt` a new run reproduces
the reference numbers to rounding. With other versions:
- exact quantities (M1, M3-M6, M9, M10, the entropies of a given attack) agree to about 1e-10;
- Monte Carlo quantities (M7, M8) may differ within their statistical error, e.g. about 1e-3 for the
  pooled mismatch of Fig. D1;
- the M11 optimizer may stop at a different point, but the PASS criteria (no gap below -1e-10; minimum
  gap below 1e-6 for dim E' >= 4; gap near -1 for the ZZ control) must hold.

`compare_results.py` reports the largest deviation per file. A file marked `DIFFERS` is acceptable if the
deviation is within these limits and `run_all.py` still reports 38/38 PASS.

## Reference environment

Python 3.13.16, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2, Linux.
