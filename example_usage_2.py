from pathlib import Path
import os
import sys
from syncerpy.syncerpy import syncerpy, print_channels

# =============================================================================
# Configuration - Select the headband type to process
# =============================================================================
HEADBAND_TYPE = "sleepprofiler"  # Options: "bitbrain", "sleepprofiler"

# Dataset-specific configurations
DATASETS = {
    "bitbrain": {
        "root_dir": "/project/4180000.46/extra_zmax_data/bitbrain_dataset",
        "output_dir": "/project/4180000.46/extra_zmax_data/aligned_bitbrain",
        "channel_ref": "HB_1",
        "channel_shift": "PSG_F3",
    },
    "sleepprofiler": {
        "root_dir": "/project/4180000.46/extra_zmax_data/sleepprofiler/",
        "output_dir": "/project/4180000.46/extra_zmax_data/aligned_sleepprofiler/",
        "channel_ref": "Fp1",  # Update with correct channel names
        "channel_shift": "EEG",  # Update with correct channel names
    },
}

# Get configuration for selected headband type
config = DATASETS[HEADBAND_TYPE]
ROOT_SEARCH_DIR = config["root_dir"]
OUTPUT_DIR = config["output_dir"]
CHANNEL_REF = config["channel_ref"]
CHANNEL_SHIFT = config["channel_shift"]
PLOT = True


def find_file_pairs(folder, edfs, headband_type):
    """
    Find matching pairs of EDF files based on the headband type.
    
    Returns:
        tuple: (ref_files, shift_files) - lists of matching reference and shift files
    """
    if headband_type == "bitbrain":
        # Bitbrain: files contain "headband" or "psg" in the name
        ref_files = [f for f in edfs if "headband" in f.name.lower()]
        shift_files = [f for f in edfs if "psg" in f.name.lower()]
        
    elif headband_type == "sleepprofiler":
        # SleepProfiler: folder structure is sleepprofiler/6011_N1/ with files:
        #   - 6011_N1.edf (PSG file - same name as folder)
        #   - 6011_N1_SleepProfiler.edf (device file)
        ref_files = [f for f in edfs if "sleepprofiler" in f.name.lower()]
        # PSG file has same name as the folder (without _SleepProfiler suffix)
        shift_files = [f for f in edfs if f.stem == folder.name]
        
    else:
        raise ValueError(f"Unknown headband type: {headband_type}")
    
    return ref_files, shift_files


def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def find_and_process_pairs(root_path):
    """
    Recursively searches for folders containing matching pairs of EDF files
    and processes them based on the selected headband type.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Warning: Path {root} does not exist. Checking current directory.")
        root = Path(".")

    print(f"Searching for data in: {root.resolve()}")
    print(f"Using headband type: {HEADBAND_TYPE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Channel ref: {CHANNEL_REF}, Channel shift: {CHANNEL_SHIFT}")
    print("-" * 60)
    sys.stdout.flush()
    
    # Setup log file
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file_path = out_path / "processing_log.txt"
    
    # Open log file in append mode (or write mode to clear start)
    with open(log_file_path, "w") as log:
        log.write(f"Processing started for headband type: {HEADBAND_TYPE}\n\n")
        
        count = 0
        # Walk through all directories
        for folder in root.rglob("*"):
            if folder.is_dir():
                # Find all EDF files in the current folder
                edfs = list(folder.glob("*.edf"))
                
                # Identify pairs based on headband type
                ref_files, shift_files = find_file_pairs(folder, edfs, HEADBAND_TYPE)

                # If we found exactly one of each in this folder, process them as a pair
                if len(ref_files) == 1 and len(shift_files) == 1:
                    count += 1
                    ref = ref_files[0]
                    shift = shift_files[0]
                    
                    ref_size = get_file_size_mb(ref)
                    shift_size = get_file_size_mb(shift)
                    
                    msg = f"Processing Pair #{count}: {ref.name} ({ref_size:.1f} MB) & {shift.name} ({shift_size:.1f} MB)"
                    print(f"\n{msg}")
                    print(f"  Folder: {folder}")
                    log.write(f"{msg}\n")
                    log.write(f"  Folder: {folder}\n")
                    sys.stdout.flush()
                    log.flush()
                    
                    process_study(ref, shift, log, count)

def process_study(file_ref, file_shift, log_file, count):
    try:
        print(f"  [1/4] Checking channels...")
        sys.stdout.flush()
        # 0. Check Channels (Optional, useful for debugging)
        print_channels(str(file_ref), str(file_shift))
        sys.stdout.flush()

        # Setup output path
        out_path = Path(OUTPUT_DIR)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Generate prefix like sub-001, sub-010, etc.
        prefix = f"sub-{count:03d}"
        
        print(f"  [2/4] Starting syncerpy alignment...")
        print(f"        Reference file: {file_ref}")
        print(f"        Shift file: {file_shift}")
        print(f"        Channel ref: {CHANNEL_REF}, Channel shift: {CHANNEL_SHIFT}")
        sys.stdout.flush()
        log_file.write(f"  Starting alignment with prefix: {prefix}\n")
        log_file.flush()

        # Calculate offset, align and cut files in one call
        print(f"  [3/4] Running syncerpy...")
        sys.stdout.flush()
        new_ref, new_shift = syncerpy(
            file_reference=str(file_ref),
            file_shift=str(file_shift),
            channel_reference=CHANNEL_REF,
            channel_shift=CHANNEL_SHIFT,
            cut_files=True,
            plot=PLOT,
            output_folder=str(out_path),
            prefix=prefix
        )
        print(f"  [4/4] Alignment complete!")
        print(f"  -> Saved aligned files to: {out_path}/")
        sys.stdout.flush()
        log_file.write(f"  -> Success. Files saved to {out_path}/\n")
        log_file.flush()

    except Exception as e:
        import traceback
        error_msg = f"  -> ERROR: {e}\n{traceback.format_exc()}"
        print(error_msg)
        sys.stdout.flush()
        log_file.write(error_msg + "\n")
        log_file.flush()

if __name__ == "__main__":
    find_and_process_pairs(ROOT_SEARCH_DIR)