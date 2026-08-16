"""
====================================================================
Title:   04 - Trial-Based Epoching (Retention Period)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This script replaces stimulus-based epoching with true trial-based 
    epoching. It scans all events to maintain a 'Global Trial Index' 
    (incremented at every '(last)' digit marker, including controls). 
    It then extracts strictly the 'memory' trials, epoching the 
    Retention period (e.g., 0.5s to 2.5s post-last-digit) to capture 
    intrinsic precision maintenance (EPRC/gamma).
"""
import mne
import pandas as pd
import numpy as np
from pathlib import Path
import gc

def epoch_retention_trials(derivatives_dir, tmin=0.5, tmax=2.5):
    deriv_path = Path(derivatives_dir)
    epochs_dir = deriv_path / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    
    clean_files = list(deriv_path.glob("*_desc-clean_eeg.fif"))
    print(f"Found {len(clean_files)} clean EEG files for trial epoching.")
    
    for file_path in clean_files:
        subj_id = file_path.name.split('_')[0]
        out_file = epochs_dir / f"{subj_id}_task-memory_desc-epoched_epo.fif"
        
        if out_file.exists():
            print(f" -> Epoched file already exists for {subj_id}. Skipping.")
            continue
            
        print(f"Processing Trial Epochs for: {subj_id}...")
        
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            id_to_marker = {v: k for k, v in event_id.items()}
            
            global_trial_index = 0
            memory_events = []
            metadata_rows = []
            
            for ev in events:
                marker_name = id_to_marker.get(ev[2], "")
                
                if '(last)' in marker_name:
                    global_trial_index += 1
                    
                    if 'memory' in marker_name:
                        memory_events.append(ev)
                        correct = 1 if 'correct' in marker_name else 0
                        metadata_rows.append({
                            'trial_index': global_trial_index,
                            'eeg_correct': correct,
                            'marker': marker_name
                        })
            
            if not memory_events:
                print(f" [!] No memory retention events found for {subj_id}.")
                continue
                
            memory_events = np.array(memory_events)
            metadata_df = pd.DataFrame(metadata_rows)
            
            epochs = mne.Epochs(raw, memory_events, tmin=tmin, tmax=tmax, 
                                baseline=None, preload=True, verbose=False)
            
            epochs.metadata = metadata_df
            
            epochs.save(out_file, overwrite=True)
            print(f" -> Successfully extracted {len(epochs)} retention trials.")
            
        except Exception as e:
            print(f" [!] ERROR processing {subj_id}: {e}")
            
        finally:
            if 'raw' in locals(): del raw
            if 'epochs' in locals(): del epochs
            gc.collect()

if __name__ == "__main__":
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    DERIVATIVES_DIR = f"./data/{TARGET_DATASET}/derivatives"
    
    T_MIN = 0.5 
    T_MAX = 2.5
    epoch_retention_trials(DERIVATIVES_DIR, tmin=T_MIN, tmax=T_MAX)