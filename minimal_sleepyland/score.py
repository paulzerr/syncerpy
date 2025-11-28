#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import shutil
from contextlib import contextmanager
import numpy as np
import pandas as pd

# --- Configuration ---
# Base directory for models in the current repo
# Assuming this script is run from the repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the uSLEEPYLAND-main directory (contains modified utime and psg_utils)
USLEEPYLAND_PATH = os.path.join(REPO_ROOT, "custom_libs")
UT_BIN = os.path.join(USLEEPYLAND_PATH, "utime", "bin", "ut.py")
MODEL_BASE_DIR = os.path.join(REPO_ROOT, "models")

# Model directory mappings (from predict.py)
# We map the model name to the subdirectory in usleepyland/model
PROJECT_DIRS = {
    "deepresnet": "deepresnet-nsrr-2024",
    "usleep": "u-sleep-nsrr-2024",
    "transformer": "sleeptransformer-nsrr-2024",
    #"yasa": "yasa",
}

@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull
    """
    # Flush any pending output
    sys.stdout.flush()
    sys.stderr.flush()
    
    with open(os.devnull, 'w') as fnull:
        # Save the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(sys.stdout.fileno())
        saved_stderr_fd = os.dup(sys.stderr.fileno())

        try:
            # Redirect stdout and stderr to devnull
            os.dup2(fnull.fileno(), sys.stdout.fileno())
            os.dup2(fnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            # Flush any pending output to devnull before restoring
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Restore stdout and stderr
            os.dup2(saved_stdout_fd, sys.stdout.fileno())
            os.dup2(saved_stderr_fd, sys.stderr.fileno())
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

def check_environment():
    if not os.path.exists(UT_BIN):
        print(f"Error: ut.py not found at {UT_BIN}")
        print("Please check the USLEEPYLAND_PATH configuration.")
        sys.exit(1)
    
    if not os.path.exists(MODEL_BASE_DIR):
        print(f"Error: Model directory not found at {MODEL_BASE_DIR}")
        print("Please ensure you are running this script from the repo root.")
        sys.exit(1)

def save_prediction_to_csv(npy_path, csv_path):
    try:
        print(f"Converting {npy_path} to CSV...")
        data = np.load(npy_path)
        
        # Expected data shape is (n_epochs, 5) for classes: Wake, N1, N2, N3, REM
        if data.ndim == 2 and data.shape[1] == 5:
            # Create DataFrame with specific columns
            df = pd.DataFrame(data, columns=["Wake", "N1", "N2", "N3", "REM"])
            
            # Add 'Epoch' column at the beginning (0-indexed)
            df.insert(0, "Epoch", range(len(df)))
            
            # Add 'Art' column at the end (filled with 0s as models don't predict artifacts)
            df["Art"] = 0
            
            df.to_csv(csv_path, index=False)
            print(f"Saved probabilities to {csv_path}")
            return True
        else:
            print(f"Warning: Unexpected data shape {data.shape}. Saving without headers.")
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            print(f"Saved probabilities to {csv_path}")
            return True
            
    except Exception as e:
        print(f"Error converting to CSV: {e}")
        return False

def run_model(edf_path, model_name, out_dir, channels=None):
    if model_name not in PROJECT_DIRS:
        print(f"Warning: Unknown model '{model_name}'. Skipping.")
        return

    project_subpath = PROJECT_DIRS[model_name]
    project_dir = os.path.join(MODEL_BASE_DIR, project_subpath)
    
    if not os.path.exists(project_dir):
        print(f"Error: Project directory for {model_name} not found at {project_dir}")
        return

    # Prepare temporary output directory for this model execution
    # We use a temp folder to avoid cluttering the main output folder with intermediate files
    model_out_dir = os.path.join(out_dir, f"temp_{model_name}")
    os.makedirs(model_out_dir, exist_ok=True)

    # Construct the command
    # Based on predict.py run_command_for_prediction_one
    # Use the specific python executable where tensorflow is installed
    python_executable = "/home/rainfern/ALL_PROJECTS/2025 - Coon model/sleepyland/bin/python"
    cmd = [
        python_executable, UT_BIN, "predict_one",
        "--num_gpus", "0",
        "-f", edf_path,
        "--seed", "123",
        "--project_dir", project_dir,
        "--strip_func", "trim_psg_trailing",
        "--majority",
        "--no_argmax",
        "--out_dir", model_out_dir,
        "--overwrite"
    ]

    # Add channels if provided
    if channels:
        cmd.append("--auto_channel_grouping")
        cmd.extend(channels)
        
        ref_channels = [c for c in channels if c != "MASTOID"]
        cmd.append("--auto_reference_types")
        cmd.extend(ref_channels)
    else:
        # If no channels specified, use auto channel grouping with EEG and EOG
        # This is required by predict_one.py when --model is not used (we use --project_dir)
        # All 3 models (usleep, deepresnet, transformer) seem to expect 2 channels (EEG, EOG)
        cmd.extend(["--auto_channel_grouping", "EEG", "EOG"])

    # Model specific flags
    if model_name == "yasa":
        cmd.extend(["--model_external", "yasa"])
    elif model_name == "usleep":
        cmd.append("--one_shot")
    elif model_name == "transformer":
        cmd.append("--is_logits")

    print(f"\n=== Running {model_name} ===")
    print(f"Command: {' '.join(cmd)}")

    # Set PYTHONPATH to include uSLEEPYLAND-main (for utime and psg_utils)
    env = os.environ.copy()
    env["PYTHONPATH"] = USLEEPYLAND_PATH + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Success! Output saved to {model_out_dir}")
        
        # Convert majority vote .npy to .csv
        # The majority vote file is typically in a 'majority' subdirectory or named with 'majority'
        
        edf_basename = os.path.splitext(os.path.basename(edf_path))[0]
        npy_filename = f"{edf_basename}_PRED.npy"
        
        # Check for majority folder first
        majority_npy_path = os.path.join(model_out_dir, "majority", npy_filename)
        
        if os.path.exists(majority_npy_path):
            # Save CSV to the main output directory (out_dir)
            # Format: FILENAME_MODELNAME.csv
            csv_filename = f"{edf_basename}_{model_name}.csv"
            csv_path = os.path.join(out_dir, csv_filename)
            
            success = save_prediction_to_csv(majority_npy_path, csv_path)
            
            if success:
                # Cleanup: Delete the temporary model output directory
                print(f"Cleaning up {model_out_dir}...")
                shutil.rmtree(model_out_dir)
        else:
            print(f"Note: Majority vote file not found at {majority_npy_path}. Skipping CSV conversion and cleanup.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_name}: {e}")

def scorer(input_file, output_folder, model):
    """
    Scores the input EDF file using the specified model(s).
    
    Args:
        input_file (str): Path to the EDF file.
        output_folder (str): Path to the output folder.
        model (str): Name of the model to use ('deepresnet', 'usleep', 'yasa', 'transformer') or 'all'.
    """
    check_environment()
    
    if not os.path.exists(input_file):
        print(f"Error: EDF file not found at {input_file}")
        return

    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

    models_to_run = []
    if model == 'all':
        models_to_run = list(PROJECT_DIRS.keys())
    elif model in PROJECT_DIRS:
        models_to_run = [model]
    else:
        print(f"Error: Unknown model '{model}'. Available models: {list(PROJECT_DIRS.keys())} or 'all'")
        return

    print(f"Processing: {input_file}")
    print(f"Output Directory: {output_folder}")
    print(f"Models: {models_to_run}")

    for m in models_to_run:
        with suppress_stdout_stderr():
            run_model(input_file, m, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score EDF files using sleep models.")
    parser.add_argument("input_file", help="Path to the EDF file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("model", help="Model name or 'all'")
    
    args = parser.parse_args()
    
    scorer(args.input_file, args.output_folder, args.model)
