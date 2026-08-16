"""
====================================================================
Title:   03 - Check EEG Event Structures (Corrected)
Author:  Takafumi Shiga (TIC-DO Institute)
====================================================================
Description:
    This diagnostic script reads a preprocessed EEG file to expose 
    the exact event dictionary and the chronological sequence of triggers.
    Explicit target subject assignment is required to prevent processing
    of excluded datasets.
"""
import mne
from pathlib import Path

def inspect_events(derivatives_dir, target_subject):
    deriv_path = Path(derivatives_dir)
    
    search_pattern = f"{target_subject}_task-memory_desc-clean_eeg.fif"
    target_files = list(deriv_path.glob(search_pattern))
    
    if not target_files:
        print(f"[!] ERROR: Clean EEG file not found for {target_subject}.")
        return
        
    file_path = target_files[0]
    print(f"Inspecting file: {file_path.name}")
    
    try:
        raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        print("\n" + "="*50)
        print(" [ EVENT DICTIONARY (event_id) ]")
        print("="*50)
        for key, val in event_id.items():
            print(f"  Marker: {key:<20} -> ID: {val}")
            
        print("\n" + "="*50)
        print(" [ FIRST 40 EVENTS SEQUENCE ]")
        print("="*50)
        
        id_to_marker = {v: k for k, v in event_id.items()}
        
        for i, event in enumerate(events[:40]):
            sample_time = event[0]
            marker_id = event[2]
            marker_name = id_to_marker.get(marker_id, "UNKNOWN")
            print(f" Event {i+1:>2}: TimeSample={sample_time:<8} | ID={marker_id:<3} | Marker={marker_name}")
            
        print("\nTotal events found in this recording:", len(events))
        
    except Exception as e:
        print(f"[!] ERROR during inspection: {e}")

if __name__ == "__main__":
    import argparse
    # ==========================================
    # USER CONFIGURATION
    # Change 'TARGET_DATASET' to the desired OpenNeuro ID 
    # ==========================================
    TARGET_DATASET = "ds003838"
    DEFAULT_DERIVATIVES = f"./data/{TARGET_DATASET}/derivatives"

    parser = argparse.ArgumentParser(description="Inspect EEG events for a specific subject.")
    parser.add_argument("--subject", type=str, required=True, help="Target subject ID (e.g., sub-032)")
    parser.add_argument("--dir", type=str, default=DEFAULT_DERIVATIVES, help="Path to derivatives directory")
    args = parser.parse_args()

    inspect_events(args.dir, target_subject=args.subject)