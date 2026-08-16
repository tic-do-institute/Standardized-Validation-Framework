"""
====================================================================
Title:   02 - Robust EEG Preprocessing Batch Pipeline
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script performs robust automated preprocessing (Bandpass and 
    Notch filtering) across all valid subjects.
    It includes strict error handling (fail-safes) and a resume mechanism 
    to skip already processed files, while generating a formal attrition log.
"""

import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
import gc

def batch_preprocess_eeg(bids_root_path, subject_list, task='memory'):
    bids_root = Path(bids_root_path)
    out_dir = bids_root / "derivatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting batch preprocessing for {len(subject_list)} subjects...")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for subj in subject_list:
        print(f"\n" + "="*50)
        print(f" Processing Subject: {subj}")
        print("="*50)
        
        out_file = out_dir / f"sub-{subj}_task-{task}_desc-clean_eeg.fif"
        
        if out_file.exists():
            print(f" -> Output file already exists. Skipping subject {subj}.")
            skip_count += 1
            continue
            
        bids_path = BIDSPath(subject=subj, task=task, datatype='eeg', 
                             suffix='eeg', root=bids_root)
        
        try:
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            raw.load_data(verbose=False)
            
            raw.pick(['eeg'])
            
            print(" -> Resampling to 250 Hz to optimize memory usage...")
            raw.resample(250.0, verbose=False)
            
            print(" -> Applying Bandpass (1.0-45.0 Hz) & Notch (50 Hz) Filters...")
            raw.notch_filter(50.0, verbose=False)
            raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
            
            raw.save(out_file, overwrite=True, verbose=False)
            print(f" -> Successfully saved to {out_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f" [!] ERROR processing subject {subj}: {e}")
            print(f" -> Skipping subject {subj} and continuing...")
            fail_count += 1
            
        finally:
            if 'raw' in locals():
                del raw
            gc.collect()
            
    print("\n" + "="*50)
    print(" [ ATTRITION LOG: PREPROCESSING PHASE ]")
    print("="*50)
    print(f" Initial Subjects : {len(subject_list)}")
    print(f" Skipped (Exists) : {skip_count}")
    print(f" Failed (Errors)  : {fail_count}")
    print(f" Final Successful : {success_count}")
    print("="*50)

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    
    from mne_bids import get_entity_vals
    
    try:
        all_subjs = get_entity_vals(BIDS_ROOT, 'subject')
        valid_subjs = [s for s in all_subjs if BIDSPath(subject=s, task='memory', datatype='eeg', suffix='eeg', root=BIDS_ROOT).fpath.exists()]
        batch_preprocess_eeg(BIDS_ROOT, valid_subjs)
    except FileNotFoundError:
        print(f"BIDS root not found. Please verify the BIDS_ROOT path.")