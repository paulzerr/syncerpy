import numpy as np
import mne
import pandas as pd
from pathlib import Path
from scipy import signal
import time
import pyedflib
from .plotting import plot_complete_alignment

# Configuration Constants
EPSILON = 1e-12
SPECT_MAX_FREQ = 30.0
QUANT_CLIP_MIN = -3
QUANT_CLIP_MAX = 3

COARSE_BANDPASS_ORDER = 4
COARSE_BANDPASS_FREQ = [0.5, 30]
COARSE_ENVELOPE_ORDER = 4
COARSE_ENVELOPE_FREQ = 0.5

# Maximum reasonable offset (hours) - files with larger offsets are likely corrupted
MAX_REASONABLE_OFFSET_HOURS = 2.0

FINE_COMPARE_WIN_SEC = 60.0
FINE_SEARCH_RANGE_SEC = 3.0
FINE_SEARCH_TIMEOUT_SEC = 30.0  # Timeout for fine search in seconds
FINE_SPECT_WIN_SEC = 1.0
FINE_BINS = 64

PLOT_WIN_SEC = 30
PLOT_OVERLAP_DIVISOR = 4
PLOT_MAX_FREQ = 20

def print_channels(file_reference, file_shift):
    """Prints the available channels and their sampling rates in the given EDF files."""
    
    def _print_info(file_path):
        path = Path(file_path)
        print(f"Channels in {path.name}:")
        if path.suffix.lower() == ".edf":
            try:
                raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
                for i, ch_name in enumerate(raw.ch_names):
                    fs = raw.info['chs'][i]['sfreq']
                    print(f"  - {ch_name} ({fs:.1f} Hz)")
            except Exception as e:
                print(f"  Error reading file: {e}")
        else:
            print(f"  Unsupported file extension: {path.suffix}")
        print("")

    _print_info(file_reference)
    _print_info(file_shift)

def load_file(path, fs, chan_name=None):
    path = Path(path)
    print(f"[IO] Loading {path.name}...")
    
    if path.suffix.lower() == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        if chan_name:
            if chan_name not in raw.ch_names:
                raise ValueError(f"Channel '{chan_name}' not found. Available: {raw.ch_names}")
            raw.pick([chan_name])
        if int(raw.info['sfreq']) != int(fs):
            raw.resample(fs, verbose='ERROR')
        sig = raw.get_data()[0]
    else:
        # Fallback or error for other file types if needed
        # For now, assuming EDF as per original code active path
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    # Z-score normalization
    sig = (sig - np.mean(sig)) / (np.std(sig) + EPSILON)
    return sig

def compute_spectrogram_features(sig, fs, win_sec, hop_sec=None, hop_type=None):
    """Generates spectrogram features (0-30Hz)."""
    nperseg = int(win_sec * fs)
    if hop_type == "1_sample":
        noverlap = nperseg - 1
    else:
        # Default overlap if not specified
        if hop_sec is None: hop_sec = win_sec / 2
        noverlap = int(nperseg - (hop_sec * fs))
    
    if noverlap < 0: noverlap = 0
    
    f, t, Sxx = signal.spectrogram(sig, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Keep only 0-30Hz, Log & Normalize
    mask = f <= SPECT_MAX_FREQ
    Sxx = Sxx[mask, :]
    Sxx = 10 * np.log10(Sxx + EPSILON)
    Sxx = (Sxx - Sxx.mean()) / (Sxx.std() + EPSILON)
    
    return Sxx.T # (Time, Freq)

def quantize(arr, bins=32):
    crange = QUANT_CLIP_MAX - QUANT_CLIP_MIN
    return ((np.clip(arr, QUANT_CLIP_MIN, QUANT_CLIP_MAX) - QUANT_CLIP_MIN) / crange * (bins - 1)).astype(np.int32)

def calc_mi(x, y, bins):
    """Fast Mutual Information on two quantized arrays"""
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    p_xy = c_xy / np.sum(c_xy)
    mask = p_xy > 0
    p_xy = p_xy[mask]
    
    p_x = np.sum(c_xy, axis=1) / np.sum(c_xy)
    p_y = np.sum(c_xy, axis=0) / np.sum(c_xy)
    
    denom = (p_x[:, None] * p_y[None, :])[mask]
    return np.sum(p_xy * np.log(p_xy / denom))

def run_coarse_stage(sRef, sShift, fs):
    # bandpass and envelope
    sos = signal.butter(COARSE_BANDPASS_ORDER, COARSE_BANDPASS_FREQ, btype='bandpass', fs=fs, output='sos')
    sos_env = signal.butter(COARSE_ENVELOPE_ORDER, COARSE_ENVELOPE_FREQ, btype='lowpass', fs=fs, output='sos')
    envRef = signal.sosfiltfilt(sos_env, np.abs(signal.sosfiltfilt(sos, sRef)))
    envShift = signal.sosfiltfilt(sos_env, np.abs(signal.sosfiltfilt(sos, sShift)))
    
    # downsample to 1Hz
    step = int(fs)
    envRef_ds = envRef[::step]
    envShift_ds = envShift[::step]
    
    # Normalize
    envRef_ds = (envRef_ds - np.mean(envRef_ds)) / (np.std(envRef_ds) + EPSILON)
    envShift_ds = (envShift_ds - np.mean(envShift_ds)) / (np.std(envShift_ds) + EPSILON)
    
    lags = signal.correlation_lags(len(envRef_ds), len(envShift_ds), mode='full')
    corr = signal.correlate(envRef_ds, envShift_ds, mode='full', method='fft')
    
    # maximum correlation
    best_idx = np.argmax(corr)
    # convert to seconds
    # since we downsampled to 1Hz, the lag value IS the offset in seconds
    offset_sec = lags[best_idx]
    
    print(f"  [Coarse search] Offset: {offset_sec:.4f} s")
    return offset_sec, corr, lags

def run_fine_stage(sRef, sShift, fs, current_offset):
    """
    Refines offset using high-res (1-sample hop) spectrograms.
    
    Returns:
    --------
    tuple : (final_offset, fine_search_data)
        final_offset : float - refined offset in seconds
        fine_search_data : dict or None - contains 'lags_ms', 'scores', 'best_lag_ms' for plotting
    """
    compare_win = FINE_COMPARE_WIN_SEC
    search_range = FINE_SEARCH_RANGE_SEC
    
    # Check if coarse offset is unreasonably large (likely corrupted file)
    max_offset_sec = MAX_REASONABLE_OFFSET_HOURS * 3600
    if abs(current_offset) > max_offset_sec:
        print(f"  [Fine] WARNING: Coarse offset {current_offset:.1f}s exceeds max reasonable offset ({max_offset_sec:.0f}s).")
        print(f"  [Fine] Skipping fine search - files may be corrupted or mismatched.")
        return current_offset, None
    
    # 1. Slice Data from Reference
    mid_idx = len(sRef) // 2
    start_Ref = mid_idx - int(compare_win/2 * fs)
    end_Ref = start_Ref + int(compare_win * fs)
    chunkRef = sRef[max(0, start_Ref):min(len(sRef), end_Ref)]
    
    center_Shift = start_Ref + int(current_offset * fs)
    
    margin_samp = int(search_range * fs)
    start_Shift_wide = center_Shift - margin_samp
    end_Shift_wide = center_Shift + len(chunkRef) + margin_samp
    
    # Validate that the shift window is within bounds
    if start_Shift_wide < 0 or end_Shift_wide > len(sShift):
        print(f"  [Fine] WARNING: Shift window out of bounds (start={start_Shift_wide}, end={end_Shift_wide}, len={len(sShift)}).")
        print(f"  [Fine] Skipping fine search - coarse offset may be incorrect.")
        return current_offset, None
    
    chunkShift_wide = sShift[max(0, start_Shift_wide):min(len(sShift), end_Shift_wide)]
    
    if len(chunkRef) < fs or len(chunkShift_wide) < fs:
        print("  [Fine] Window too short, skipping.")
        return current_offset, None

    # compute features
    print("  [Fine] Computing spectrogram features...")
    featRef = compute_spectrogram_features(chunkRef, fs, win_sec=FINE_SPECT_WIN_SEC, hop_type="1_sample")
    featShift = compute_spectrogram_features(chunkShift_wide, fs, win_sec=FINE_SPECT_WIN_SEC, hop_type="1_sample")
    
    # quantize & search
    qRef = quantize(featRef, bins=FINE_BINS).ravel()
    qShift = quantize(featShift, bins=FINE_BINS)
    
    # search every sample in the margin
    search_len = len(qShift) - len(featRef)
    
    if search_len <= 0:
        print(f"  [Fine] No search range available (search_len={search_len}), skipping.")
        return current_offset, None
    
    print(f"  [Fine] Searching {search_len} positions...")
    
    best_mi = -1
    best_local_lag = 0
    start_time = time.time()
    
    # Store search results for plotting (subsample for efficiency)
    lags_ms = []
    scores = []
    subsample_step = max(1, search_len // 500)  # Keep at most ~500 points for plotting
    
    for i in range(search_len):
        # Check timeout periodically
        if i % 100 == 0:
            elapsed = time.time() - start_time
            if elapsed > FINE_SEARCH_TIMEOUT_SEC:
                print(f"  [Fine] WARNING: Search timeout after {elapsed:.1f}s at position {i}/{search_len}.")
                print(f"  [Fine] Using best match found so far.")
                break
        
        sliceShift = qShift[i : i + len(featRef)]
        score = calc_mi(qRef, sliceShift.ravel(), FINE_BINS)
        
        # Store for plotting (subsampled)
        if i % subsample_step == 0:
            # Convert local lag to ms relative to coarse offset
            # Each position is 1 sample = 1/fs seconds
            lag_offset_sec = (i - margin_samp) / fs  # Offset from coarse position
            lags_ms.append(lag_offset_sec * 1000)  # Convert to ms
            scores.append(score)
        
        if score > best_mi:
            best_mi = score
            best_local_lag = i
            
    best_start_Shift_global = start_Shift_wide + best_local_lag
    final_offset = (best_start_Shift_global - start_Ref) / fs
    
    # Calculate best lag in ms relative to coarse offset
    best_lag_offset_ms = (best_local_lag - margin_samp) / fs * 1000
    
    fine_search_data = {
        'lags_ms': lags_ms,
        'scores': scores,
        'best_lag_ms': best_lag_offset_ms
    }
    
    print(f"  [Fine search] offset: {final_offset:.5f} s (MI={best_mi:.4f})")
    return final_offset, fine_search_data

def syncerpy(file_reference, file_shift, channel_reference=None, channel_shift=None,
             cut_files=False, plot=False, output_folder=None, fs=64, prefix=None):
    """
    Synchronize two EDF files by calculating the time offset between them.
    
    Parameters:
    -----------
    file_reference : str
        Path to the reference EDF file
    file_shift : str
        Path to the file to be shifted/aligned
    channel_reference : str, optional
        Channel name to use from reference file
    channel_shift : str, optional
        Channel name to use from shift file
    cut_files : bool, default=False
        If True, creates aligned/cut EDF files and generates combined plot
        If False, only returns the offset
    plot : bool, default=False
        If True, saves the combined alignment plot to disk (never displays it)
    output_folder : str, optional
        Directory to save output files (default: same as reference file)
    fs : int, default=64
        Sampling frequency to resample to
    prefix : str, optional
        Prefix for output filenames
        
    Returns:
    --------
    If cut_files=False:
        offset : float
            Time offset in seconds
    If cut_files=True:
        tuple : (out_name_ref, out_name_shift) or (None, None) if alignment failed
            Paths to the aligned output files
    """
    t0 = time.time()
    
    file_reference = Path(file_reference)
    file_shift = Path(file_shift)
    
    if output_folder:
        out_dir = Path(output_folder)
    else:
        out_dir = file_reference.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load signals for alignment calculation
    sRef = load_file(file_reference, fs, channel_reference)
    sShift = load_file(file_shift, fs, channel_shift)

    # 2. Calculate offset
    offset, coarse_corr, coarse_lags = run_coarse_stage(sRef, sShift, fs)
    coarse_offset = offset  # Save for diagnostics
    offset, fine_search_data = run_fine_stage(sRef, sShift, fs, offset)
    
    # Check if offset is reasonable
    max_offset_sec = MAX_REASONABLE_OFFSET_HOURS * 3600
    offset_is_suspicious = abs(offset) > max_offset_sec
    
    if offset_is_suspicious:
        print(f"\n{'='*30}")
        print(f"WARNING: SUSPICIOUS OFFSET DETECTED!")
        print(f"OFFSET: {offset:.2f} s ({offset/3600:.2f} hours)")
        print(f"This exceeds the maximum reasonable offset of {MAX_REASONABLE_OFFSET_HOURS} hours.")
        print(f"Files may be corrupted or from different recordings.")
        print(f"{'='*30}")
    else:
        print(f"\n{'='*30}\nOFFSET: {offset*1000:.2f} ms\n{'='*30}")
    
    print(f"Total execution time: {time.time() - t0:.3f} s")
    
    # 3. If cut_files is False, just return the offset
    if not cut_files:
        return offset
    
    # 4. Open readers to get file info (needed for plots even if cutting fails)
    print(f"[Cutter] Processing {file_reference.name} and {file_shift.name}...")
    
    f_ref = pyedflib.EdfReader(str(file_reference))
    f_shift = pyedflib.EdfReader(str(file_shift))
    
    # Get headers
    ref_signal_headers = f_ref.getSignalHeaders()
    shift_signal_headers = f_shift.getSignalHeaders()
    
    ref_header = f_ref.getHeader()
    shift_header = f_shift.getHeader()
    
    # Get file durations
    ref_dur = f_ref.getFileDuration()
    shift_dur = f_shift.getFileDuration()
    
    # Determine crop times (in seconds)
    if offset > 0:
        start_ref_sec = offset
        start_shift_sec = 0
    else:
        start_ref_sec = 0
        start_shift_sec = abs(offset)
    
    # Calculate common duration
    rem_ref = ref_dur - start_ref_sec
    rem_shift = shift_dur - start_shift_sec
    common_dur = min(rem_ref, rem_shift)
    
    # Check if cutting is viable
    cutting_viable = common_dur > 0 and not offset_is_suspicious
    
    if not cutting_viable:
        if common_dur <= 0:
            print(f"[Cutter] ERROR: No overlapping region (common_dur={common_dur:.1f}s).")
        if offset_is_suspicious:
            print(f"[Cutter] WARNING: Skipping file cutting due to suspicious offset.")
        print(f"[Cutter] Start ref: {start_ref_sec:.1f}s, Start shift: {start_shift_sec:.1f}s")
        print(f"[Cutter] Ref duration: {ref_dur:.1f}s, Shift duration: {shift_dur:.1f}s")
    else:
        print(f"[Cutter] Aligning starts: Cut {start_ref_sec:.3f}s from Ref, {start_shift_sec:.3f}s from Shift")
        print(f"[Cutter] Common duration: {common_dur:.3f}s")
    
    # Helper to process and write file
    def process_file(reader, writer_path, start_sec, duration_sec, signal_headers, main_header):
        n_channels = reader.signals_in_file
        
        # Create writer
        writer = pyedflib.EdfWriter(str(writer_path), n_channels=n_channels)
        
        # Set Main Header
        writer.setHeader(main_header)
        
        # Set Signal Headers
        for header in signal_headers:
            if 'sample_rate' in header:
                if 'sample_frequency' not in header:
                    header['sample_frequency'] = header['sample_rate']
                del header['sample_rate']
        
        writer.setSignalHeaders(signal_headers)
        
        # Read, Crop, Write Signals
        data_list = []
        for i in range(n_channels):
            fs_ch = signal_headers[i]['sample_frequency']
            
            start_idx = int(start_sec * fs_ch)
            end_idx = start_idx + int(duration_sec * fs_ch)
            
            sig = reader.readSignal(i)
            sig_cropped = sig[start_idx:end_idx]
            data_list.append(sig_cropped)
            
        writer.writeSamples(data_list)
        writer.close()
        return data_list
        
    # Output paths
    fname_ref = f"{file_reference.stem}_synced.edf"
    fname_shift = f"{file_shift.stem}_synced.edf"
    
    if prefix:
        fname_ref = f"{prefix}-{fname_ref}"
        fname_shift = f"{prefix}-{fname_shift}"
        
    out_name_ref = out_dir / fname_ref
    out_name_shift = out_dir / fname_shift
    
    ref_data = None
    shift_data = None
    
    if cutting_viable:
        print(f"[Cutter] Saving to {out_name_ref}...")
        ref_data = process_file(f_ref, out_name_ref, start_ref_sec, common_dur, ref_signal_headers, ref_header)
        
        print(f"[Cutter] Saving to {out_name_shift}...")
        shift_data = process_file(f_shift, out_name_shift, start_shift_sec, common_dur, shift_signal_headers, shift_header)
    else:
        out_name_ref = None
        out_name_shift = None
    
    # Close readers
    f_ref._close()
    f_shift._close()
    
    # 5. Generate plot if requested - ALWAYS generate for diagnostics, even if cutting failed
    if plot:
        status_str = "FAILED" if offset_is_suspicious else "OK"
        print(f"[Plotting] Generating alignment plot (status: {status_str})...")
        EPSILON = 1e-12
        
        # For cut signals, use actual cut data if available, otherwise use raw signals
        if ref_data is not None and shift_data is not None:
            sig_ref_cut = ref_data[0]
            sig_shift_cut = shift_data[0]
            sig_ref_cut = (sig_ref_cut - np.mean(sig_ref_cut)) / (np.std(sig_ref_cut) + EPSILON)
            sig_shift_cut = (sig_shift_cut - np.mean(sig_shift_cut)) / (np.std(sig_shift_cut) + EPSILON)
            # Cut signals are at original file sample rates
            fs_ref_cut = ref_signal_headers[0].get('sample_frequency', ref_signal_headers[0].get('sample_rate'))
            fs_shift_cut = shift_signal_headers[0].get('sample_frequency', shift_signal_headers[0].get('sample_rate'))
        else:
            # Use raw signals (already normalized) for diagnostic plot
            sig_ref_cut = sRef
            sig_shift_cut = sShift
            # These are at the resampled rate
            fs_ref_cut = fs
            fs_shift_cut = fs
        
        # Add FAILED to filename if alignment was suspicious
        if offset_is_suspicious:
            fname = f"alignment_FAILED_{file_reference.stem}_vs_{file_shift.stem}.png"
        else:
            fname = f"alignment_complete_{file_reference.stem}_vs_{file_shift.stem}.png"
        if prefix:
            fname = f"{prefix}-{fname}"
        save_path = out_dir / fname
        
        plot_complete_alignment(
            sRef_raw=sRef,
            sShift_raw=sShift,
            sRef_cut=sig_ref_cut,
            sShift_cut=sig_shift_cut,
            fs_raw=fs,  # Raw signals are at the resampled rate (64Hz)
            fs_ref_cut=fs_ref_cut,
            fs_shift_cut=fs_shift_cut,
            offset_sec=offset,
            name_ref=file_reference.stem,  # Use stem (filename without extension)
            name_shift=file_shift.stem,
            coarse_corr=coarse_corr,
            coarse_lags=coarse_lags,
            fine_search_data=fine_search_data,
            plot_win_sec=PLOT_WIN_SEC,
            plot_overlap_divisor=PLOT_OVERLAP_DIVISOR,
            plot_max_freq=PLOT_MAX_FREQ,
            save_path=str(save_path),
            show_plot=False,
            is_failed=offset_is_suspicious
        )
        
    if cutting_viable:
        print("[Cutter] Done.")
    else:
        print("[Cutter] Done (no files written due to alignment issues).")
    
    return out_name_ref, out_name_shift