"""
====================================================================
Title:   05 - Epoched Power Spectral Density (PSD) Computation
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script computes the PSD for each individual trial using Welch's method.
    To avoid HDF5/PyTables dependency conflicts on Windows with NumPy 2.x, 
    the resulting spectra are saved as native compressed NumPy arrays (.npz).
"""
import mne
import numpy as np
from pathlib import Path
import gc

def compute_epoch_psd(derivatives_dir):
    deriv_path = Path(derivatives_dir)
    epochs_dir = deriv_path / "epochs"
    out_dir = deriv_path / "psd_epochs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_files = list(epochs_dir.glob("*_desc-epoched_epo.fif"))
    print(f"Found {len(epoch_files)} epoched files for PSD computation.")
    
    for file_path in epoch_files:
        subj_id = file_path.name.split('_')[0]
        out_file = out_dir / f"{subj_id}_task-memory_desc-psd.npz"
        
        if out_file.exists():
            print(f" -> PSD already exists for {subj_id}. Skipping.")
            continue
            
        print(f"Computing PSD for {subj_id}...")
        try:
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            
            spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=40.0, 
                                          n_fft=250, n_overlap=125, verbose=False)
            
            psd_data = spectrum.get_data()
            freqs = spectrum.freqs
            ch_names = epochs.ch_names
            
            np.savez(out_file, psd=psd_data, freqs=freqs, ch_names=ch_names)
            print(f" -> Saved Epoched PSD to {out_file.name}")
            
        except Exception as e:
            print(f" [!] ERROR computing PSD for {subj_id}: {e}")
            
        finally:
            if 'epochs' in locals(): del epochs
            if 'spectrum' in locals(): del spectrum
            gc.collect()

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    compute_epoch_psd(DERIVATIVES_DIR)