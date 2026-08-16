"""
====================================================================
Title:   01 - Data Loading and Robust BIDS Structure Validation
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script robustly validates the BIDS directory structure for the 
    OpenNeuro ds003838 dataset. It dynamically scans for missing EEG data 
    (as noted in the dataset README) and generates a strict list of valid 
    subjects for downstream batch processing.
"""

from pathlib import Path
import mne_bids
from mne_bids import BIDSPath, get_entity_vals

def generate_valid_subject_list(bids_root_path, task='memory'):
    bids_root = Path(bids_root_path)
    
    if not bids_root.exists():
        print(f"Error: Directory {bids_root} does not exist.")
        return []

    all_subjects = get_entity_vals(bids_root, 'subject')
    print(f"Total subjects detected in BIDS: {len(all_subjects)}")
    
    valid_subjects = []
    missing_subjects = []
    
    print("\nScanning for EEG data availability...")
    for subj in all_subjects:
        bids_path = BIDSPath(subject=subj, task=task, datatype='eeg', 
                             suffix='eeg', root=bids_root)
        
        if bids_path.fpath.exists():
            valid_subjects.append(subj)
        else:
            missing_subjects.append(subj)
            
    print(f"\n--- Validation Summary ---")
    print(f"Valid subjects (EEG found)  : {len(valid_subjects)}")
    print(f"Missing subjects (Skipped)  : {len(missing_subjects)}")
    if missing_subjects:
        print(f"List of missing subjects  : {missing_subjects}")
        
    return valid_subjects

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # (e.g., 'ds003655', 'ds003838', 'ds005095', 'ds008104', 'ds006040')
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"