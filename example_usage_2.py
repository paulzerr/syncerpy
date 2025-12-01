from pathlib import Path
from syncerpy.syncerpy import syncerpy, print_channels
from cutterpy import cut
 
# Configuration
ROOT_SEARCH_DIR = "/mnt/HBS/extra_zmax_data/bitbrain_dataset"
OUTPUT_DIR = "/mnt/HBS/extra_zmax_data/aligned_bitbrain"  # All synced files will be saved here
CHANNEL_REF = 'HB_1'
CHANNEL_SHIFT = 'PSG_F3'
SHOW_PLOT = False
SAVE_PLOT = True
SAVE_CORRELATION_PLOT = True

def find_and_process_pairs(root_path):
    """
    Recursively searches for folders containing matching pairs of EDF files
    (one 'headband' and one 'psg') and processes them immediately.
    """
    root = Path(root_path)
    if not root.exists():
        print(f"Warning: Path {root} does not exist. Checking current directory.")
        root = Path(".")

    print(f"Searching for data in: {root.resolve()}")
    
    # Setup log file
    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file_path = out_path / "processing_log.txt"
    
    # Open log file in append mode (or write mode to clear start)
    with open(log_file_path, "w") as log:
        log.write(f"Processing started...\n\n")
        
        count = 0
        # Walk through all directories
        for folder in root.rglob("*"):
            if folder.is_dir():
                # Find all EDF files in the current folder
                edfs = list(folder.glob("*.edf"))
                
                # Identify pairs based on naming convention
                ref_files = [f for f in edfs if "headband" in f.name.lower()]
                shift_files = [f for f in edfs if "psg" in f.name.lower()]

                # If we found exactly one of each in this folder, process them as a pair
                if len(ref_files) == 1 and len(shift_files) == 1:
                    count += 1
                    ref = ref_files[0]
                    shift = shift_files[0]
                    
                    msg = f"Processing Pair #{count}: {ref.name} & {shift.name}"
                    print(f"\n{msg}")
                    log.write(f"{msg}\n")
                    
                    process_study(ref, shift, log, count)

def process_study(file_ref, file_shift, log_file, count):
    try:
        # 0. Check Channels (Optional, useful for debugging)
        # print_channels(str(file_ref), str(file_shift))

        # 2. Align and cut files
        out_path = Path(OUTPUT_DIR)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Generate prefix like sub-001, sub-010, etc.
        prefix = f"sub-{count:03d}"

        # 1. Calculate Offset
        offset = syncerpy(
            file_reference=str(file_ref),
            file_shift=str(file_shift),
            channel_reference=CHANNEL_REF,
            channel_shift=CHANNEL_SHIFT,
            plot=True,
            show_plot=SHOW_PLOT,
            save_plot=SAVE_PLOT,
            save_correlation_plot=SAVE_CORRELATION_PLOT,
            output_folder=str(out_path),
            prefix=prefix
        )
        print(f"  -> Calculated Offset: {offset} seconds")
        log_file.write(f"  -> Success. Offset: {offset} seconds\n")

        new_ref, new_shift = cut(
            file_reference=str(file_ref),
            file_shift=str(file_shift),
            offset=offset,
            output_folder=str(out_path),
            plot=True,
            show_plot=SHOW_PLOT,
            save_plot=SAVE_PLOT,
            prefix=prefix
        )
        print(f"  -> Saved aligned files to: {out_path}/")

    except Exception as e:
        print(f"  -> ERROR: {e}")
        log_file.write(f"  -> ERROR: {e}\n")

if __name__ == "__main__":
    find_and_process_pairs(ROOT_SEARCH_DIR)