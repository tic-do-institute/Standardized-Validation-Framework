"""
====================================================================
Title:   12 - Progression Dynamics Analysis (ds006040)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script performs sensitivity analyses on the continuous-
    performance dataset (ds006040) across different pseudo-epoch 
    window sizes (5s, 10s, 20s). It utilizes Random-Intercept-only 
    Linear Mixed Models (LMM) to evaluate macroscopic trajectory 
    generalization.
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def analyze_progression_dynamics(derivatives_dir):
    target_dir = Path(derivatives_dir)
    master_csv = target_dir / "multimodal_master_matrix_progression.csv"
    
    if not master_csv.exists():
        print("[!] ERROR: Master matrix not found.")
        return
        
    df_full = pd.read_csv(master_csv)
    metrics = ['lzc', 'sampen', 'spectral_exponent']
    window_sizes = sorted(df_full['window_sec'].unique())
    
    print("\n" + "="*70)
    print(" [ LMM: NEURAL COMPRESSION GENERALIZATION TEST (ds006040) ]")
    print("="*70)
    
    for win_sec in window_sizes:
        print(f"\n" + "#"*50)
        print(f" ANALYSIS WINDOW: {int(win_sec)} SEC")
        print("#"*50)
        
        df = df_full[df_full['window_sec'] == win_sec].copy()
        if len(df) == 0: continue
            
        df['elapsed_time_z'] = (df['elapsed_time_sec'] - df['elapsed_time_sec'].mean()) / df['elapsed_time_sec'].std()
        
        for m in metrics:
            if m not in df.columns: continue
            df[f'{m}_z'] = (df[m] - df[m].mean()) / df[m].std()
            formula = f"{m}_z ~ elapsed_time_z"
            
            try:
                # Maintain original model specification (Random Intercept only)
                md = smf.mixedlm(formula, df, groups=df["subject"])
                mdf = md.fit(method='cg')
                
                coef = mdf.params['elapsed_time_z']
                pval = mdf.pvalues['elapsed_time_z']
                
                # Extract Standard Error and 95% Confidence Interval
                bse = mdf.bse['elapsed_time_z']
                ci = mdf.conf_int().loc['elapsed_time_z']
                
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
    # (For this continuous analysis, it should be 'ds006040')
    # ==========================================
    TARGET_DATASET = "ds006040"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    DERIVATIVES_DIR = f"{BIDS_ROOT}/derivatives"
    
    analyze_progression_dynamics(DERIVATIVES_DIR)