#!/usr/bin/env python3
"""
Script to repair EDF headers for PSG files in the sleepprofiler dataset.
Targets the non-sleepprofiler EDF files (PSG files that have the same name as their folder).

The sleepprofiler dataset structure:
    sleepprofiler/6011_N1/
        - 6011_N1.edf (PSG file - THIS IS WHAT WE REPAIR)
        - 6011_N1_SleepProfiler.edf (device file - skip)
"""

from pathlib import Path
import sys

# =============================================================================
# Configuration
# =============================================================================
ROOT_DIR = "/project/4180000.46/extra_zmax_data/sleepprofiler/"
OUTPUT_DIR = "/project/4180000.46/extra_zmax_data/sleepprofiler_repaired/"

# Set to True to overwrite original files in place, False to save to OUTPUT_DIR
OVERWRITE_ORIGINAL = False


def repair_edf_header(filepath, output_filepath):
    """
    Repair EDF header by replacing invalid non-ASCII characters.
    
    Args:
        filepath: Path to the input EDF file
        output_filepath: Path to save the repaired EDF file
    
    Returns:
        int: Number of replacements made, or -1 if critically corrupt
    """
    with open(filepath, 'rb') as f:
        content = bytearray(f.read())

    # Get number of signals (ns) from the fixed header
    # The number of signals is at bytes 252-256 (4 bytes) as a string
    try:
        ns_str = content[252:256].decode('ascii').strip()
        ns = int(ns_str)
    except Exception as e:
        print(f"  Critically corrupt header (cannot read number of signals): {e}")
        return -1

    # Calculate the size of the total header
    # Fixed header (256 bytes) + (ns * 256 bytes per signal definition)
    header_bytes = 256 + (ns * 256)

    print(f"  Scanning the first {header_bytes} bytes (Header area)...")
    
    replacements = 0
    
    # Iterate ONLY through the header bytes
    for i in range(header_bytes):
        byte_val = content[i]
        
        # Check if byte is outside valid 7-bit ASCII (0-127)
        if byte_val > 127:
            # Common fix: If it looks like 'µ' (Latin-1 0xB5), replace with 'u' (0x75)
            # Otherwise, replace with a space (0x20) or '?' (0x3F)
            if byte_val == 0xB5: 
                content[i] = 0x75  # 'u'
            else:
                content[i] = 0x20  # Space
            replacements += 1

    if replacements > 0:
        print(f"  Found and repaired {replacements} invalid characters.")
        with open(output_filepath, 'wb') as f_out:
            f_out.write(content)
        print(f"  Saved repaired file to: {output_filepath}")
    else:
        print("  No invalid characters found in the header.")
    
    return replacements


def find_psg_files(root_path):
    """
    Find all PSG EDF files (non-sleepprofiler) in the sleepprofiler dataset.
    
    PSG files are identified as EDF files that:
    - Do NOT contain 'sleepprofiler' in their filename (case-insensitive)
    - Have the same stem as their parent folder (e.g., 6011_N1.edf in folder 6011_N1/)
    
    Args:
        root_path: Path to the sleepprofiler dataset root
        
    Yields:
        Path objects for each PSG EDF file found
    """
    root = Path(root_path)
    
    for folder in root.rglob("*"):
        if folder.is_dir():
            # Find all EDF files in this folder
            edfs = list(folder.glob("*.edf"))
            
            for edf_file in edfs:
                # Skip files with 'sleepprofiler' in the name
                if "sleepprofiler" in edf_file.name.lower():
                    continue
                
                # Check if the file stem matches the folder name (PSG file pattern)
                if edf_file.stem == folder.name:
                    yield edf_file


def main():
    root = Path(ROOT_DIR)
    
    if not root.exists():
        print(f"Error: Root directory does not exist: {ROOT_DIR}")
        sys.exit(1)
    
    print(f"Searching for PSG EDF files in: {root.resolve()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Overwrite original: {OVERWRITE_ORIGINAL}")
    print("-" * 60)
    sys.stdout.flush()
    
    # Create output directory if needed
    if not OVERWRITE_ORIGINAL:
        out_path = Path(OUTPUT_DIR)
        out_path.mkdir(parents=True, exist_ok=True)
    
    # Find and process all PSG files
    psg_files = list(find_psg_files(ROOT_DIR))
    total_files = len(psg_files)
    
    print(f"Found {total_files} PSG EDF files to process.\n")
    
    repaired_count = 0
    error_count = 0
    clean_count = 0
    
    for idx, psg_file in enumerate(psg_files, 1):
        print(f"[{idx}/{total_files}] Processing: {psg_file.name}")
        print(f"  Path: {psg_file}")
        sys.stdout.flush()
        
        # Determine output path
        if OVERWRITE_ORIGINAL:
            output_file = psg_file
        else:
            # Preserve folder structure in output directory
            relative_path = psg_file.relative_to(ROOT_DIR)
            output_file = Path(OUTPUT_DIR) / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Repair the header
        result = repair_edf_header(str(psg_file), str(output_file))
        
        if result == -1:
            error_count += 1
        elif result > 0:
            repaired_count += 1
        else:
            clean_count += 1
        
        print()
        sys.stdout.flush()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Files repaired:        {repaired_count}")
    print(f"Files already clean:   {clean_count}")
    print(f"Files with errors:     {error_count}")
    
    if not OVERWRITE_ORIGINAL:
        print(f"\nRepaired files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()