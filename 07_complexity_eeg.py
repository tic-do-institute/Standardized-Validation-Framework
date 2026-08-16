"""
====================================================================
Title:   07 - EEG Complexity Extraction (Strict Bound Edition)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script extracts nonlinear complexity metrics strictly from 
    trial-synchronized retention epochs. 
    It features a mathematically exact, dependency-free implementation 
    of Lempel-Ziv Complexity (Kaspar & Schuster algorithm) and Sample 
    Entropy, providing a deterministic, dependency-minimal implementation.
====================================================================
"""
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import warnings

warnings.filterwarnings('ignore')

def lzc_pure(sig):
    median = np.median(sig)
    seq = (sig > median).astype(int).astype(str)
    s = "".join(seq)
    n = len(s)
    if n == 0: return 0.0
    
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    
    while True:
        if l + k > n:
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
                
    return (c * np.log2(n)) / n

def sampen_pure(sig, m=2, r_tol=0.2):
    r = r_tol * np.std(sig)
    n = len(sig)
    limit = n - m
    
    if limit <= 0:
        return np.nan
        
    templates_m = np.lib.stride_tricks.sliding_window_view(sig, window_shape=m)
    templates_m1 = np.lib.stride_tricks.sliding_window_view(sig, window_shape=m+1)
    
    A = 0
    B = 0
    
    for i in range(limit):
        dist_m = np.max(np.abs(templates_m[:limit] - templates_m[i]), axis=1)
        A += np.sum(dist_m <= r) - 1  
        
        dist_m1 = np.max(np.abs(templates_m1[:limit] - templates_m1[i]), axis=1)
        B += np.sum(dist_m1 <= r) - 1
        
    if A == 0 or B == 0:
        return np.nan
        
    return -np.log(B / A)

def compute_epoch_complexity(epoch_data):
    n_channels = epoch_data.shape[0]
    lzc_vals = np.zeros(n_channels)
    sampen_vals = np.zeros(n_channels)
    
    for ch in range(n_channels):
        sig = epoch_data[ch, :]
        lzc_vals[ch] = lzc_pure(sig)
        sampen_vals[ch] = sampen_pure(sig)
        
    return np.nanmean(lzc_vals), np.nanmean(sampen_vals)

def extract_complexity(derivatives_dir):
    deriv_path = Path(derivatives_dir)
    epochs_dir = deriv_path / "epochs"
    out_csv = deriv_path / "eeg_complexity_features.csv"
    
    epoch_files = list(epochs_dir.glob("*_desc-epoched_epo.fif"))
    print(f"Found {len(epoch_files)} trial-epoched EEG files.")
    
    if not epoch_files:
        print("[!] ERROR: No epoched files found.")
        return
        
    all_results = []
    
    for file_path in epoch_files:
        subj_id = file_path.name.split('_')[0].replace('sub-', '')
        print(f"Processing Complexity for Subject: {subj_id}...")
        
        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            metadata = epochs.metadata
            data = epochs.get_data()
            
            for idx, trial_idx in enumerate(metadata['trial_index']):
                epoch_sig = data[idx, :, :]
                lzc_mean, sampen_mean = compute_epoch_complexity(epoch_sig)
                
                all_results.append({
                    'subject': subj_id,
                    'trial_index': int(trial_idx),
                    'lzc': float(lzc_mean),
                    'sampen': float(sampen_mean)
                })
                
            print(f" -> Successfully processed {len(metadata)} trials.")
            
        except Exception as e:
            print(f" [!] ERROR processing {subj_id}: {e}")
            
        finally:
            if 'epochs' in locals(): del epochs
            if 'data' in locals(): del data
            gc.collect()
            
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(out_csv, index=False)
        print(f"\nSuccessfully saved global complexity features to: {out_csv}")

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    extract_complexity(DERIVATIVES_DIR)