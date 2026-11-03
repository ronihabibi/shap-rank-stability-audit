#!/usr/bin/env bash
# Full reproduction. Run from the package root (about 30 minutes on 2 cores for step 2).
set -e
pip install -r requirements.txt
python data/download_data.py                                   # 1. five main public datasets -> data/raw/
BREP=40 python code/multi.py adult,telco,diamonds,german,insurance 30 results/multi_rank_matrices.csv 100,300   # 2. audits
python code/analyze_multi.py results/multi_rank_matrices.csv results/multi_estimates_vs_truth.csv              # 3. SRSI, truth, errors
python code/tables_multi.py results                            # 4. tables with sample-level intervals
python code/fig_cal.py results                                 # 5. calibration figure
python code/test_audit.py                                      # 6. calibration / LR+ checks
# Optional: 13 PMLB datasets (data/pmlb/), large-D reference, logistic-regression robustness
#   python code/pmlb/dl.py; python code/pmlb/dl2.py; BREP=40 python code/pmlb/multi_pm.py ...
#   MODEL=lr BREP=40 python code/multi.py adult,telco,diamonds,german,insurance 30 results/multi_rank_matrices_lr.csv 100,300
