"""
====================================================================
Title:   Generate Complete LMM Results Table
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This batch script compiles the primary Linear Mixed Model (LMM) 
    estimates (Slope, SE, 95% CI, p-value) across all evaluated datasets 
    into a unified CSV format (Table S1). It dynamically applies the 
    correct random-effects structure (Random Intercept + Random Slope 
    for trial-based data; Random Intercept only for continuous data).
"""
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def generate_complete_lmm_results():
    datasets = ['ds003655', 'ds003838', 'ds005095', 'ds006040', 'ds008104']
    metrics = ['spectral_exponent', 'lzc', 'sampen']
    results = []
    
    for ds_name in datasets:
        print(f"\nProcessing {ds_name}...")
        
        # Build relative paths based on the standard architecture
        derivatives_dir = Path(f"./data/{ds_name}/derivatives")
        if ds_name == 'ds006040':
            csv_path = derivatives_dir / "multimodal_master_matrix_progression.csv"
        else:
            csv_path = derivatives_dir / "multimodal_master_matrix.csv"
            
        if not csv_path.exists():
            print(f" [!] ERROR: File not found -> {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Special handling for ds006040 (5-second pseudo-epochs, elapsed_time_sec)
        if ds_name == 'ds006040':
            df = df[df['window_sec'] == 5.0].copy()
            time_col = 'elapsed_time_sec'
            re_form = None  # Random Intercept only
        else:
            time_col = 'trial_index'
            re_form = "~time_z"  # Random Intercept + Random Slope
            
        # Standardize temporal variable
        df['time_z'] = (df[time_col] - df[time_col].mean()) / df[time_col].std()
        
        for m in metrics:
            if m not in df.columns:
                print(f" [!] Metric '{m}' not found in {ds_name}. Skipping.")
                continue
                
            # Standardize target metric
            df[f'{m}_z'] = (df[m] - df[m].mean()) / df[m].std()
            
            formula = f"{m}_z ~ time_z"
            
            try:
                # Execute LMM maintaining the original model specification
                md = smf.mixedlm(formula, df, groups=df["subject"], re_formula=re_form)
                mdf = md.fit(method='cg')
                
                if 'time_z' in mdf.pvalues:
                    coef = mdf.params['time_z']
                    pval = mdf.pvalues['time_z']
                    bse = mdf.bse['time_z']
                    
                    # Extract 95% Confidence Interval
                    conf_int = mdf.conf_int().loc['time_z']
                    lower_ci = conf_int[0]
                    upper_ci = conf_int[1]
                    
                    results.append({
                        'Dataset': ds_name,
                        'Metric': 'beta' if m == 'spectral_exponent' else m.upper(),
                        'Slope': coef,
                        'SE': bse,
                        '95% CI Lower': lower_ci,
                        '95% CI Upper': upper_ci,
                        'p-value': pval
                    })
                    print(f" -> {m.upper()}: Slope={coef:.4f}, SE={bse:.4f}, p={pval:.4f}")
                    
            except Exception as e:
                print(f" [!] {ds_name} {m.upper()} LMM failed: {e}")

    # Save results to CSV in the current directory
    if results:
        out_df = pd.DataFrame(results)
        out_csv = Path("./lmm_complete_results_pre.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"\n[ OK ] 15 Complete LMM results saved to: {out_csv.absolute()}")

if __name__ == "__main__":
    generate_complete_lmm_results()