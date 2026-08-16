"""
====================================================================
Title:   08 - Behavioral Feature Extraction (Revised)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script strictly parses the ds003838 behavioral TSV files.
    It extracts the raw condition and computes Normalized Accuracy.
    All temporal dynamics (e.g., rolling variance) are intentionally 
    omitted here to prevent data leakage and arbitrary parameterization 
    in the extraction layer.
"""
import pandas as pd
from pathlib import Path

def extract_behavior_ds003838(bids_root, derivatives_dir):
    bids_path = Path(bids_root)
    out_dir = Path(derivatives_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "behavioral_features.csv"
    
    beh_files = list(bids_path.rglob("*_beh.tsv"))
    print(f"Found {len(beh_files)} behavioral TSV files.")
    
    if not beh_files:
        print("[!] ERROR: No behavioral files found.")
        return
        
    all_results = []
    
    for file_path in beh_files:
        subj_id = file_path.name.split('_')[0].replace('sub-', '')
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            req_cols = ['condition', 'trial', 'partialScore']
            if not all(col in df.columns for col in req_cols):
                print(f" [!] Missing required columns in {file_path.name}. Skipping.")
                continue
                
            df['condition'] = pd.to_numeric(df['condition'], errors='coerce')
            df['partialScore'] = pd.to_numeric(df['partialScore'], errors='coerce')
            df['trial'] = pd.to_numeric(df['trial'], errors='coerce')
            
            df['normalized_accuracy'] = df['partialScore'] / df['condition']
            
            for _, row in df.iterrows():
                if pd.isna(row['trial']):
                    continue
                all_results.append({
                    'subject': subj_id,
                    'trial_index': int(row['trial']),
                    'condition': int(row['condition']),
                    'normalized_accuracy': float(row['normalized_accuracy'])
                })
                
        except Exception as e:
            print(f" [!] ERROR processing {subj_id}: {e}")
            
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(out_csv, index=False)
        print(f"\nSuccessfully saved static behavioral features to: {out_csv}")

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    extract_behavior_ds003838(BIDS_ROOT, DERIVATIVES_DIR)