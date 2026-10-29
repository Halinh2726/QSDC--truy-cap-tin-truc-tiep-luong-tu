#!/usr/bin/env bash
# Full verification for QSDC v4_4: modules M1-M11, Appendix D figures, numbers added to the text,
# and comparison with the reference results. Run from this folder:  bash run_everything.sh [--quick]
set -e
cd "$(dirname "$0")"
echo "== 1/4  verification modules M1-M11 (full ~10 min; --quick reuses the M11 search) =="
if [ "$1" = "--quick" ] && [ ! -f verification/results/m11_adversarial.csv ]; then
  mkdir -p verification/results
  cp reference_results/verification_results/m11_adversarial.csv verification/results/
fi
(cd verification && python3 run_all.py $1)
echo "== 2/4  Appendix D figures =="
python3 appendixD/make_figs.py verification/results appendixD/figures > appendixD/figures_numbers.log
echo "    written: appendixD/figures/figD1_schedule.pdf, figD2_uncertainty.pdf, figD3_completeness.pdf"
echo "== 3/4  numbers added to the v4_4 text (Table I p_opt column, Sec. VI-C) =="
python3 text_numbers/check_v44_text_numbers.py verification/results/m6_table1.csv
echo "== 4/4  comparison with reference_results =="
python3 compare_results.py verification/results reference_results/verification_results
