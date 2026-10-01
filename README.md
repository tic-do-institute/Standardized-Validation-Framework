# Standardized Validation Framework (v1.0)

## Overview
This repository contains the completely standardized and deterministically frozen computational pipeline (v1.0) utilized by the TIC-DO Institute. It is designed to extract macroscopic neurophysiological dynamics (aperiodic spectral exponent β, Lempel-Ziv Complexity, Sample Entropy) from BIDS-formatted EEG datasets.

## Pipeline Architecture
The pipeline enforces a strict modular sequence to prevent analytical confounding and data leakage:
- **`01_load_data.py` to `04_epoch_trial.py`**: Robust BIDS parsing, automated preprocessing (Bandpass 1-45Hz, Notch 50Hz, 250Hz downsampling), and true trial-based epoching.
- **`05_psd_epochs.py` & `06_fooof_epochs.py`**: Welch's PSD computation and parameterization of the aperiodic component via FOOOF.
- **`07_complexity_eeg.py`**: Mathematically exact, dependency-minimal extraction of Lempel-Ziv Complexity and Sample Entropy.
- **`08_extract_behavior.py` & `09_merge_multimodal.py`**: Static behavioral extraction and strictly deduplicated Master Matrix assembly.
- **`10_analyze_all_trial_dynamics.py`**: Primary hierarchical statistical evaluation (LMM) and robustness checks (OLS, GEE) for trial-based datasets. *(Updated for Physical Review E submission to explicitly output Standard Errors and 95% Wald CIs while strictly maintaining the original random-effects specifications).*
- **`11_analyze_progression_dynamics.py`**: Temporal progression analysis across conditions.
- **`12_progression_dynamics_ds006040.py`**: Specific sensitivity analysis (5s, 10s, 20s windows) for the continuous-performance dataset (ds006040), integrating exact SE and 95% CI extraction.

---

## 🌟 Statistical Auditability
To maximize methodological transparency and facilitate independent statistical verification, the core evaluation scripts (`10_` and `12_`) compute and output precise Standard Errors (SE) and 95% Confidence Intervals directly from the fitted `MixedLMResults` objects. 

We provide the following supplementary audit files:
*   **`generate_lmm_complete_table.py`**: A unified batch script used to compile the `Table 3` directly from the master matrices, ensuring absolute consistency across all evaluated datasets.

---

## Usage Requirements & Directory Structure
Raw datasets must be formatted according to the Brain Imaging Data Structure (BIDS) and placed in the `./data/` directory.

- **Configuration:** Before executing, modify the `TARGET_DATASET` variable in the `__main__` block of each script to match your target OpenNeuro ID (e.g., `TARGET_DATASET = "ds003838"`).
- Outputs (preprocessed EEG, PSDs, extracted features, and master matrices) will be automatically generated in the respective dataset's BIDS derivative folder (`./data/[TARGET_DATASET]/derivatives/`).

```text
Project_Root/
 ├── 01_load_data.py
 ├── 02_preprocess.py
 ├── ...
 ├── 10_analyze_all_trial_dynamics.py         <- Updated with SE & 95% CI outputs
 ├── 11_analyze_progression_dynamics.py
 ├── 12_progression_dynamics_ds006040.py    <- Out-of-sample continuous analysis
 ├── generate_lmm_complete_table.py           <- Table S1 compiler
 ├── requirements.txt
 └── data/
      ├── ds003655/          <- Raw BIDS data
      │    └── derivatives/  <- Automatically generated outputs
      ├── ds003838/
      └── ...

