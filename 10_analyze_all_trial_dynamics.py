"""
====================================================================
Title:   10 - All Trial Dynamics Analysis (LMM & GEE)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script evaluates the macrodynamic trajectories of complexity and
    spectral features across discrete-trial architectures. It performs 
    Linear Mixed Models (LMM) for primary temporal estimation, subject-
    level OLS for robustness checks, and Generalized Estimating Equations 
    (GEE) as a strict sensitivity analysis against autocorrelation.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

def analyze_trial_dynamics(derivatives_dir):
    target_dir = Path(derivatives_dir)
    master_csv = target_dir / "multimodal_master_matrix.csv"
    
    if not master_csv.exists():
        print(f"[!] ERROR: Master matrix not found at: {master_csv}")
        return
        
    df = pd.read_csv(master_csv)
    df['trial_z'] = (df['trial_index'] - df['trial_index'].mean()) / df['trial_index'].std()
    metrics = ['lzc', 'sampen', 'spectral_exponent']
    
    for m in metrics:
        if m in df.columns:
            df[f'{m}_z'] = (df[m] - df[m].mean()) / df[m].std()
            
    print(f"\n[ ANALYZING DATASET: {target_dir.parent.name} ]")
    print("--- 1. PRIMARY ANALYSIS (LMM) ---")
    for m in metrics:
        if f'{m}_z' not in df.columns: continue
        try:
            md = smf.mixedlm(f"{m}_z ~ trial_z", df, groups=df["subject"], re_formula="~trial_z")
            mdf = md.fit(method='cg')
            print(f" > LMM | {m.ljust(18)} | Slope: {mdf.params['trial_z']:>7.4f} | p-val: {mdf.pvalues['trial_z']:.4f}")
        except Exception as e:
            print(f" > LMM | {m.ljust(18)} | [!] Failed: {e}")

    print("--- 2. ROBUSTNESS CHECK: SUBJECT-LEVEL OLS ---")
    subject_metrics = []
    for subj_id, subj_df in df.groupby('subject'):
        subj_df = subj_df.sort_values('trial_index')
        X_z = sm.add_constant(subj_df['trial_z'].values)
        row = {'subject': subj_id}
        for m in metrics:
            if f'{m}_z' in subj_df.columns:
                try:
                    ols = sm.OLS(subj_df[f'{m}_z'].values, X_z).fit()
                    row[f'{m}_slope'] = ols.params[1]
                except Exception as e:
                    print(f" [!] OLS Failed for {subj_id}: {e}")
        subject_metrics.append(row)
        
    df_subj = pd.DataFrame(subject_metrics)
    for m in metrics:
        if f'{m}_slope' in df_subj.columns:
            t_stat, p_val = ttest_1samp(df_subj[f'{m}_slope'].dropna(), 0.0)
            print(f" > OLS | {m.ljust(18)} | Mean Slope: {df_subj[f'{m}_slope'].mean():>7.4f} | p-val: {p_val:.4f}")

    print("--- 3. SENSITIVITY ANALYSIS (GEE) ---")
    df_sorted = df.sort_values(['subject', 'trial_index']).copy()
    df_sorted['group_id'] = df_sorted.groupby('subject').ngroup()
    
    for m in metrics:
        if f'{m}_z' not in df_sorted.columns: continue
        try:
            Y = df_sorted[f'{m}_z'].to_numpy()
            X = sm.add_constant(df_sorted['trial_z'].to_numpy())
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
    # Define the list of target datasets for batch statistical analysis
    # ==========================================
    DATASETS = ["ds003655", "ds003838", "ds005095", "ds008104"]
    
    for ds in DATASETS:
        target_dir = f"./data/{ds}/derivatives"
        analyze_trial_dynamics(target_dir)