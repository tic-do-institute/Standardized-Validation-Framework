"""
====================================================================
Title:   09 - EEG-Behavioral Master Matrix Integration
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script integrates the EEG-derived feature matrices (Complexity 
    and Spectral) with the static behavioral performance matrix.
    It rigorously enforces strict subject-trial alignment to prevent 
    asynchronous mismatches.
====================================================================
"""
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def build_master_matrix(derivatives_dir):
    deriv_path = Path(derivatives_dir)
    
    f_beh = deriv_path / "behavioral_features.csv"
    f_eeg_comp = deriv_path / "eeg_complexity_features.csv"
    f_eeg_spec = deriv_path / "eeg_spectral_features.csv"
    out_csv = deriv_path / "multimodal_master_matrix.csv"
    
    files = [f_beh, f_eeg_comp, f_eeg_spec]
    for f in files:
        if not f.exists():
            print(f"[!] ERROR: Required file missing - {f.name}")
            return
            
    print("Loading feature matrices...")
    df_beh = pd.read_csv(f_beh)
    df_eeg_comp = pd.read_csv(f_eeg_comp)
    df_eeg_spec = pd.read_csv(f_eeg_spec)
    
    for df in [df_beh, df_eeg_comp, df_eeg_spec]:
        df['subject'] = df['subject'].astype(str).str.zfill(3)
        df['trial_index'] = df['trial_index'].astype(int)
    
    print("Applying deduplication...")
    df_beh = df_beh.drop_duplicates(subset=['subject', 'trial_index'])
    df_eeg_comp = df_eeg_comp.drop_duplicates(subset=['subject', 'trial_index'])
    df_eeg_spec = df_eeg_spec.drop_duplicates(subset=['subject', 'trial_index'])
    
    print(f" -> Behavior trials : {len(df_beh)}")
    print(f" -> EEG Complexity trials : {len(df_eeg_comp)}")
    print(f" -> EEG Spectral trials : {len(df_eeg_spec)}")
    
    print("\nExecuting strict Inner Join...")
    
    master_df = df_beh.merge(df_eeg_comp, on=['subject', 'trial_index'], how='inner')
    master_df = master_df.merge(df_eeg_spec, on=['subject', 'trial_index'], how='inner')
    
    if len(master_df) == 0:
        print("[!] CRITICAL ERROR: 0 trials survived the inner join.")
        return
        
    master_df = master_df.sort_values(by=['subject', 'trial_index'])
    master_df.to_csv(out_csv, index=False)
    
    print("\n" + "="*50)
    print(" [ MODULE 3: MASTER MATRIX ASSEMBLED ]")
    print("="*50)
    print(f" Total Surviving Trials : {len(master_df)}")
    print(f" Total Unique Subjects  : {master_df['subject'].nunique()}")
    print(f" Output File            : {out_csv}")
    print("="*50)

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    build_master_matrix(DERIVATIVES_DIR)
