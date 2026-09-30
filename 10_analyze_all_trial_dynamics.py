"""
====================================================================
Title:   10 - All Trial Dynamics Analysis (LMM & GEE)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script evaluates the macrodynamic trajectories of complexity and
    spectral features across discrete-trial architectures. It performs 
    Linear Mixed Models (LMM) for primary temporal estimation, extracting
    Standard Errors (SE) and 95% Wald Confidence Intervals to ensure
    methodological transparency for the PRE submission.
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def analyze_trial_dynamics(derivatives_dir):
    target_dir = Path(derivatives_dir)
    master_csv = target_dir / "multimodal_master_matrix.csv"
    
    print(f"\n{'='*80}")
    print(f" [ ANALYZING DATASET: {target_dir.parent.name} ]")
    print(f"{'='*80}")

    if not master_csv.exists():
        print(f"[!] ERROR: Master matrix not found at: {master_csv}")
        return
        
    df = pd.read_csv(master_csv)
    
    # Standardization
    df['trial_z'] = (df['trial_index'] - df['trial_index'].mean()) / df['trial_index'].std()
    metrics = ['lzc', 'sampen', 'spectral_exponent']
    
    for m in metrics:
        if m in df.columns:
            df[f'{m}_z'] = (df[m] - df[m].mean()) / df[m].std()
            
    print("\n--- 1. PRIMARY ANALYSIS (LMM) ---")
    for m in metrics:
        if f'{m}_z' not in df.columns: continue
        try:
            # Maintain original model specification (Random Intercept + Random Slope)
            md = smf.mixedlm(f"{m}_z ~ trial_z", df, groups=df["subject"], re_formula="~trial_z")
            mdf = md.fit(method='cg')
            
            coef = mdf.params['trial_z']
            pval = mdf.pvalues['trial_z']
            
            # Extract Standard Error and 95% Confidence Interval
            bse = mdf.bse['trial_z']
            ci = mdf.conf_int().loc['trial_z']
            
            print(
                f" > LMM | {m.ljust(18)} | "
                f"Slope: {coef:>7.4f} | "
                f"SE: {bse:.4f} | "
                f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}] | "
                f"p-val: {pval:.4g}"
            )
        except Exception as e:
            print(f" > LMM | {m.ljust(18)} | [!] Failed: {e}")

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # (e.g., 'ds003655', 'ds003838', 'ds005095', 'ds008104')
    # ==========================================
    TARGET_DATASET = "ds003838"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    analyze_trial_dynamics(DERIVATIVES_DIR)