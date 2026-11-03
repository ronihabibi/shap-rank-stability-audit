# Data and code: audit of SHAP rank-stability designs on public tabular data

Companion package for the manuscript "How Reliable Are SHAP Rankings on Small Tabular Data? An Empirical Audit of Stability Measures and Audit Designs" (working title).

## What is here
| Path | Content |
|---|---|
| `data/download_data.py` | Downloads the five public datasets into `data/raw/` and prints their SHA-256 prefixes. |
| `code/multi.py` | Draws disjoint samples (n = 100 and 300), runs the full model, a bootstrap audit (B = 40) and a seed-only audit (B = 40), and stores the rank of every domain. |
| `code/analyze_multi.py` | Computes SRSI, rank SD, modal-rank frequency, top-k frequency and interval width for each audit, and the truth from independent samples. |
| `code/tables_multi.py`, `code/fig_cal.py` | Summary tables with sample-level bootstrap intervals (`tables_multi.json`) and Figure 1. |
| `code/srsi_interval.py`, `code/bootstrap_ranks.py` | Stand-alone SRSI recomputation from a rank matrix, and a template that builds a rank matrix from your own data. |
| `code/rank_audit.py`, `code/test_audit.py` | Reusable SRSI/bands implementation and checks of calibration and LR+. |
| `code/pmlb/` | Download and audit of the 13 additional PMLB datasets. |
| `code/sensitivity/` | Large-reference (D = 100/70) analysis and calibration/sensitivity scripts. |
| `case_study/case_study_A.py` | Documented encoding and audit used for the partnership-register case study (needs the register workbook, which is not shared). |
| `results/` | `multi_rank_matrices.csv` (all rank vectors), `multi_estimates_vs_truth.csv`, `tables_multi.json`, `fig_cal.png`. |
| `run_all.sh` | Full reproduction (about 30 minutes on two cores). |

## Reproduce
```
pip install -r requirements.txt
bash run_all.sh
```
To only recompute tables and the figure from the stored rank matrices:
```
python code/analyze_multi.py results/multi_rank_matrices.csv results/multi_estimates_vs_truth.csv
python code/tables_multi.py results
python code/fig_cal.py results
```
Status: the analysis scripts reproduce the stored estimates exactly from the stored rank matrices (checked for `analyze_multi.py`); the sampling scripts were run during the study, and a single sampling unit re-run reproduced the stored rank vectors. Not every script was re-run end to end after the paths were made relative.

Checked earlier: re-running one sampling unit (insurance, n = 100, draw 0) reproduced the stored rank vectors exactly; re-running the analysis from the stored matrices reproduced the reported tables.

## Settings
Python 3.13, scikit-learn 1.9.1, shap 0.54.1. Random Forest: 100 trees, maximum depth 5, other parameters default. Seeds: full model 42; bootstrap resample b uses model seed 42 + b with row indices from generator 777 + draw index; seed-only repeat b uses model seed 1000 + b on the full sample. Disjoint samples come from one fixed permutation of each dataset (generator seed 31). Each original variable is one domain; categorical variables are one-hot encoded (at most 10 categories kept, the rest pooled) and domain scores are sums of the member columns' mean absolute SHAP values.

## Data sources (not redistributed here; each keeps its original terms)
| Dataset | Source |
|---|---|
| adult | UCI Machine Learning Repository, Becker and Kohavi (1996), doi:10.24432/C5XW20; file from github.com/jbrownlee/Datasets |
| German Credit | UCI Machine Learning Repository, Hofmann (1994), doi:10.24432/C5NC77; file from github.com/jbrownlee/Datasets |
| diamonds | ggplot2 data (Wickham, 2016, doi:10.1007/978-3-319-24277-4); file from github.com/mwaskom/seaborn-data |
| telco churn | github.com/IBM/telco-customer-churn-on-icp4d (no DOI) |
| insurance | github.com/stedy/Machine-Learning-with-R-datasets (no DOI) |

## What is not included
The two institutional datasets (partnership records, N = 189; internship survey, N = 396) contain partner and student information and are not shared; the case-study results in the manuscript cannot be reproduced from this package. Only aggregate case-study output (`results/case_study_A_output.json`) is included. `case_study/case_study_A.py` documents the exact domain encoding; it reads a workbook path given as its first argument and an optional `HOME_UNIT_KEYWORD` environment variable, so it can be adapted to another register.

## Licence
Code and derived result files: MIT (see `LICENSE`). Public datasets: original terms.
