import numpy as np
import mne
import pandas as pd
from pathlib import Path
from scipy import signal
import time
from .plotting import plot_results, plot_spectrograms

# Configuration Constants
EPSILON = 1e-12
SPECT_MAX_FREQ = 30.0
QUANT_CLIP_MIN = -3
QUANT_CLIP_MAX = 3

COARSE_BANDPASS_ORDER = 4
COARSE_BANDPASS_FREQ = [0.5, 30]
COARSE_ENVELOPE_ORDER = 4
COARSE_ENVELOPE_FREQ = 0.5

FINE_COMPARE_WIN_SEC = 60.0
FINE_SEARCH_RANGE_SEC = 3.0
FINE_SPECT_WIN_SEC = 1.0
FINE_BINS = 64

PLOT_WIN_SEC = 30
PLOT_OVERLAP_DIVISOR = 4
PLOT_MAX_FREQ = 30

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
    """Refines offset using high-res (1-sample hop) spectrograms."""
    compare_win = FINE_COMPARE_WIN_SEC
    search_range = FINE_SEARCH_RANGE_SEC
    
    # 1. Slice Data from Reference
    mid_idx = len(sRef) // 2
    start_Ref = mid_idx - int(compare_win/2 * fs)
    end_Ref = start_Ref + int(compare_win * fs)
    chunkRef = sRef[max(0, start_Ref):min(len(sRef), end_Ref)]
    
    center_Shift = start_Ref + int(current_offset * fs)
    
    margin_samp = int(search_range * fs)
    start_Shift_wide = center_Shift - margin_samp
    end_Shift_wide = center_Shift + len(chunkRef) + margin_samp
    chunkShift_wide = sShift[max(0, start_Shift_wide):min(len(sShift), end_Shift_wide)]
    
    if len(chunkRef) < fs or len(chunkShift_wide) < fs:
        print("  [Fine] Window too short, skipping.")
        return current_offset

    # compute features
    featRef = compute_spectrogram_features(chunkRef, fs, win_sec=FINE_SPECT_WIN_SEC, hop_type="1_sample")
    featShift = compute_spectrogram_features(chunkShift_wide, fs, win_sec=FINE_SPECT_WIN_SEC, hop_type="1_sample")
    
    # quantize & search
    qRef = quantize(featRef, bins=FINE_BINS).ravel()
    qShift = quantize(featShift, bins=FINE_BINS)
    
    # search every sample in the margin
    search_len = len(qShift) - len(featRef)
    
    best_mi = -1
    best_local_lag = 0
    for i in range(search_len):
        sliceShift = qShift[i : i + len(featRef)]
        score = calc_mi(qRef, sliceShift.ravel(), FINE_BINS)
        if score > best_mi:
            best_mi = score
            best_local_lag = i
            
    best_start_Shift_global = start_Shift_wide + best_local_lag
    final_offset = (best_start_Shift_global - start_Ref) / fs
    
    print(f"  [Fine search] offset: {final_offset:.5f} s")
    return final_offset

def syncerpy(file_reference, file_shift, channel_reference=None, channel_shift=None,
             plot=False, show_plot=True, save_plot=False, save_correlation_plot=False,
             output_folder=None, fs=64, prefix=None):
    t0 = time.time()
    
    file_reference = Path(file_reference)
    file_shift = Path(file_shift)
    
    if output_folder:
        out_dir = Path(output_folder)
    else:
        out_dir = file_reference.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    sRef = load_file(file_reference, fs, channel_reference)
    sShift = load_file(file_shift, fs, channel_shift)
    
    # Plot 1: Pre-alignment Spectrograms (Raw files) - DISABLED in favor of combined plot in cutterpy
    # if save_plot or (plot and show_plot):
    #     # If we are saving, or if we are showing (which implies generating)
    #     # Note: The user asked for "save_plot" to control saving. "show_plot" controls showing.
    #
    #     # We only save if save_plot is True.
    #     save_path = None
    #     if save_plot:
    #         save_path = out_dir / f"spectrograms_raw_{file_reference.stem}_vs_{file_shift.stem}.png"
    #
    #     # We only generate/show if save_plot is True OR show_plot is True
    #     if save_plot or show_plot:
    #         plot_spectrograms(sRef, sShift, fs,
    #                           name_a=f"{file_reference.name} (raw)",
    #                           name_b=f"{file_shift.name} (raw)",
    #                           title="Raw Spectrograms (Pre-alignment)",
    #                           plot_win_sec=PLOT_WIN_SEC,
    #                           plot_overlap_divisor=PLOT_OVERLAP_DIVISOR,
    #                           plot_max_freq=PLOT_MAX_FREQ,
    #                           save_path=str(save_path) if save_path else None,
    #                           show_plot=show_plot)

    offset, coarse_corr, coarse_lags = run_coarse_stage(sRef, sShift, fs)
    offset = run_fine_stage(sRef, sShift, fs, offset)
    
    print(f"\n{'='*30}\nOFFSET: {offset*1000:.2f} ms\n{'='*30}")
    print(f"Total execution time: {time.time() - t0:.3f} s")
    
    # Plot 3: Correlation Plot (Existing)
    if save_correlation_plot or (plot and show_plot):
        save_path = None
        if save_correlation_plot:
            fname = f"correlation_{file_reference.stem}_vs_{file_shift.stem}.png"
            if prefix:
                fname = f"{prefix}-{fname}"
            save_path = out_dir / fname
            
        if save_correlation_plot or show_plot:
            plot_results(sRef, sShift, fs, offset, file_reference.name, file_shift.name, coarse_corr, coarse_lags,
                         PLOT_WIN_SEC, PLOT_OVERLAP_DIVISOR, PLOT_MAX_FREQ,
                         save_path=str(save_path) if save_path else None,
                         show_plot=show_plot)
        
    return offset