# Standardized Validation Framework (v1.0)

## Overview
This repository contains the completely standardized and deterministically frozen computational pipeline (v1.0) utilized by the TIC-DO Institute. It is designed to extract macroscopic neurophysiological dynamics (aperiodic spectral exponent $\beta$, Lempel-Ziv Complexity, Sample Entropy) from BIDS-formatted EEG datasets.

## Pipeline Architecture
The pipeline enforces a strict modular sequence to prevent analytical confounding and data leakage:

1. **`01_load_data.py`** to **`04_epoch_trial.py`**: Robust BIDS parsing, automated preprocessing (Bandpass 1-45Hz, Notch 50Hz, 250Hz downsampling), and true trial-based epoching.
2. **`05_psd_epochs.py`** & **`06_fooof_epochs.py`**: Welch's PSD computation and parameterization of the aperiodic component via FOOOF.
3. **`07_complexity_eeg.py`**: Mathematically exact, dependency-minimal extraction of Lempel-Ziv Complexity and Sample Entropy.
4. **`08_extract_behavior.py`** & **`09_merge_multimodal.py`**: Static behavioral extraction and strictly deduplicated Master Matrix assembly.
5. **`10_analyze_all_trial_dynamics.py`** & **`11_analyze_progression_dynamics.py`**: Statistical evaluation using Linear Mixed Models (LMM) and Generalized Estimating Equations (GEE) for sensitivity analysis.

## Usage Requirements & Directory Structure
- Raw datasets must be formatted according to the Brain Imaging Data Structure (BIDS) and placed in the `./data/` directory.
- **Configuration:** Before executing, modify the `TARGET_DATASET` variable in the `__main__` block of each script to match your target OpenNeuro ID (e.g., `TARGET_DATASET = "ds003838"`).
- Outputs (preprocessed EEG, PSDs, extracted features, and master matrices) will be automatically generated in the respective dataset's BIDS derivative folder (`./data/[TARGET_DATASET]/derivatives/`).

```text
Project_Root/
 ├── 01_load_data.py
 ├── 02_preprocess.py
 ├── ...
 ├── 11_analyze_progression_dynamics.py
 ├── requirements.txt
 └── data/
      ├── ds003655/          <- Raw BIDS data
      │    └── derivatives/  <- Automatically generated outputs
      ├── ds003838/
      └── ...
