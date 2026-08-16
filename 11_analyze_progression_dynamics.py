"""
====================================================================
Title:   11 - Generalization Dynamics Evaluation (Continuous Time)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script evaluates macrodynamic trajectories for the continuous-
    performance dataset (ds006040). It specifically handles continuous 
    elapsed time and nested run structures, applying Linear Mixed Models 
    (LMM) and Generalized Estimating Equations (GEE) to test out-of-sample 
    generalization of the dynamical restriction hypothesis.
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def analyze_progression_dynamics(master_csv_path):
    master_csv = Path(master_csv_path)
    if not master_csv.exists():
        print("[!] ERROR: Progression matrix not found.")
        return
        
    df_full = pd.read_csv(master_csv)
    metrics = ['lzc', 'sampen', 'spectral_exponent']
    window_sizes = sorted(df_full['window_sec'].unique())
    
    for win_sec in window_sizes:
        print(f"\n[ ANALYSIS WINDOW: {int(win_sec)} SEC ]")
        df = df_full[df_full['window_sec'] == win_sec].copy()
        if len(df) == 0: continue
            
        df['elapsed_time_z'] = (df['elapsed_time_sec'] - df['elapsed_time_sec'].mean()) / df['elapsed_time_sec'].std()
        for m in metrics:
            if m in df.columns:
                df[f'{m}_z'] = (df[m] - df[m].mean()) / df[m].std()
        
        print("--- 1. PRIMARY ANALYSIS (LMM) ---")
        for m in metrics:
            if f'{m}_z' not in df.columns: continue
            try:
                md = smf.mixedlm(f"{m}_z ~ elapsed_time_z", df, groups=df["subject"])
                mdf = md.fit(method='cg')
                print(f" > LMM | {m.ljust(18)} | Slope: {mdf.params['elapsed_time_z']:>7.4f} | p-val: {mdf.pvalues['elapsed_time_z']:.4f}")
            except Exception as e:
                print(f" > LMM | {m.ljust(18)} | [!] Failed: {e}")

        print("--- 2. SENSITIVITY ANALYSIS (GEE) ---")
        df_sorted = df.sort_values(['subject', 'run', 'epoch_index']).copy()
        df_sorted['group_id'] = df_sorted.groupby(['subject', 'run']).ngroup()
        
        for m in metrics:
            if f'{m}_z' not in df_sorted.columns: continue
            try:
                Y = df_sorted[f'{m}_z'].to_numpy()
                X = sm.add_constant(df_sorted['elapsed_time_z'].to_numpy())
                groups = df_sorted['group_id'].to_numpy()
                
                try:
                    cov_struct = sm.cov_struct.Autoregressive()
                    model = sm.GEE(Y, X, groups=groups, cov_struct=cov_struct, family=sm.families.Gaussian())
                    res = model.fit(cov_type='robust')
                    struct_name = "AR(1)"
                except Exception:
                    cov_struct = sm.cov_struct.Independence()
                    model = sm.GEE(Y, X, groups=groups, cov_struct=cov_struct, family=sm.families.Gaussian())
                    res = model.fit(cov_type='robust')
                    struct_name = "Robust-Indep"
                
                print(f" > GEE | {m.ljust(18)} | Slope: {res.params[1]:>7.4f} | p-val: {res.pvalues[1]:.4f} | ({struct_name})")
            except Exception as e:
                print(f" > GEE | {m.ljust(18)} | [!] Failed: {e}")

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # This script is specifically designed for the continuous-performance dataset (ds006040)
    # ==========================================
    TARGET_DATASET = "ds006040"
    BIDS_ROOT = f"./data/{TARGET_DATASET}"
    MASTER_CSV = f"{BIDS_ROOT}/derivatives/multimodal_master_matrix_progression.csv"
    
    analyze_progression_dynamics(MASTER_CSV)