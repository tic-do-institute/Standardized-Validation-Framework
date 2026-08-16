"""
====================================================================
Title:   06 - Epoched Spectral Exponent Extraction via FOOOF
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script applies FOOOF to the single-trial PSD data (.npz format) 
    to extract the aperiodic component (Spectral Exponent) for each trial.
"""
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from fooof import FOOOFGroup
import gc

def extract_epoch_fooof(psd_epochs_dir, epochs_dir, out_csv_path):
    psd_path = Path(psd_epochs_dir)
    psd_files = list(psd_path.glob("*_desc-psd.npz"))
    print(f"Found {len(psd_files)} PSD files for FOOOF extraction.")
    
    fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.1, 
                    max_n_peaks=6, aperiodic_mode='fixed', verbose=False)
    freq_range = [1.0, 40.0]
    results = []
    
    for file_path in psd_files:
        subj_id = file_path.name.split('_')[0].replace('sub-', '')
        print(f"Fitting FOOOF for Subject: {subj_id}...")
        
        try:
            npz_data = np.load(file_path)
            psd_data = npz_data['psd']
            freqs = npz_data['freqs']
            ch_names = npz_data['ch_names']
            
            epoch_file = Path(epochs_dir) / f"sub-{subj_id}_task-memory_desc-epoched_epo.fif"
            epochs = mne.read_epochs(epoch_file, preload=False, verbose=False)
            trial_indices = epochs.metadata['trial_index'].values
            
            n_epochs, n_channels, _ = psd_data.shape
            
            for ep_idx in range(n_epochs):
                trial_idx = trial_indices[ep_idx]
                epoch_psd = psd_data[ep_idx, :, :]
                
                fg.fit(freqs, epoch_psd, freq_range)
                aperiodic_params = fg.get_params('aperiodic_params')
                
                for ch_idx, ch_name in enumerate(ch_names):
                    exponent = aperiodic_params[ch_idx, 1]
                    results.append({
                        'subject': subj_id,
                        'trial_index': trial_idx,
                        'channel': ch_name,
                        'spectral_exponent': exponent
                    })
                    
        except Exception as e:
            print(f" [!] ERROR running FOOOF for {subj_id}: {e}")
            
        finally:
            if 'npz_data' in locals(): npz_data.close()
            if 'epochs' in locals(): del epochs
            gc.collect()
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv_path, index=False)
        print(f"\nSuccessfully saved Epoched FOOOF results to: {out_csv_path}")

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    PSD_DIR = f"{DERIVATIVES_DIR}/psd_epochs"
    EPOCHS_DIR = f"{DERIVATIVES_DIR}/epochs"
    OUT_CSV = f"{DERIVATIVES_DIR}/fooof_spectral_exponents_epochs.csv"
    
    extract_epoch_fooof(PSD_DIR, EPOCHS_DIR, OUT_CSV)