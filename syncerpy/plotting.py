import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lspopt import spectrogram_lspopt
from pathlib import Path


def _get_filename(name_or_path):
    """Extract just the filename (stem) from a path or name."""
    return Path(name_or_path).stem


def plot_complete_alignment(sRef_raw, sShift_raw, sRef_cut, sShift_cut,
                           fs_raw, fs_ref_cut, fs_shift_cut, offset_sec,
                           name_ref, name_shift,
                           coarse_corr=None, coarse_lags=None,
                           fine_search_data=None,
                           plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=20,
                           save_path=None, show_plot=False, is_failed=False):
    """
    Creates a comprehensive 2x3 plot showing alignment analysis.
    
    Layout:
        Left Column:                    Right Column:
        +-----------------------+       +-----------------------+
        | Unaligned TFRs        |       | Full Aligned TFRs     |
        | (both from t=0)       |       |                       |
        +-----------------------+       +-----------------------+
        | Coarse Cross-Corr     |       | Zoomed View 1         |
        |                       |       | (1h from middle)      |
        +-----------------------+       +-----------------------+
        | Fine Search Results   |       | Zoomed View 2         |
        |                       |       | (1h from 2nd half)    |
        +-----------------------+       +-----------------------+
    """
    
    # Extract just filenames
    name_ref = _get_filename(name_ref)
    name_shift = _get_filename(name_shift)
    
    # Plot configuration - increased font sizes
    TITLE_FONT_SIZE = 20
    LABEL_FONT_SIZE = 14
    TICK_FONT_SIZE = 12
    ANNOTATION_FONT_SIZE = 12
    RATIO_FONT_SIZE = 18

    def get_spec(sig, fs_local):
        win = int(plot_win_sec * fs_local)
        overlap = plot_win_sec / plot_overlap_divisor
        f, t, S = spectrogram_lspopt(sig, fs_local, nperseg=win, noverlap=overlap)
        return f[f<=plot_max_freq], t, 10*np.log10(S[f<=plot_max_freq])

    # Calculate all spectrograms
    # Raw signals use fs_raw (resampled rate, e.g. 64Hz)
    fR_raw, tR_raw, SR_raw = get_spec(sRef_raw, fs_raw)
    fS_raw, tS_raw, SS_raw = get_spec(sShift_raw, fs_raw)
    
    # Cut signals use their respective sample rates
    fR_cut, tR_cut, SR_cut = get_spec(sRef_cut, fs_ref_cut)
    fS_cut, tS_cut, SS_cut = get_spec(sShift_cut, fs_shift_cut)

    # Robust normalization
    def get_robust_lims(S):
        q25, q50, q75 = np.percentile(S, [25, 50, 75])
        iqr = q75 - q25
        vmin = q50 - 1.5 * iqr
        vmax = q50 + 1.5 * iqr
        return vmin, vmax

    vmin_Rr, vmax_Rr = get_robust_lims(SR_raw)
    vmin_Sr, vmax_Sr = get_robust_lims(SS_raw)
    vmin_Rc, vmax_Rc = get_robust_lims(SR_cut)
    vmin_Sc, vmax_Sc = get_robust_lims(SS_cut)

    # Create figure with 2 columns, 3 rows
    fig = plt.figure(figsize=(24, 13.5))  # 16:9 aspect ratio
    
    # Format offset string
    if abs(offset_sec) >= 3600:
        offset_str = f"{offset_sec/3600:.2f} hours"
    elif abs(offset_sec) >= 60:
        offset_str = f"{offset_sec/60:.2f} min"
    else:
        offset_str = f"{offset_sec*1000:.2f} ms"
    
    # Title with filename and offset info
    if is_failed:
        title_lines = f"{name_ref} (REFERENCE)\n{name_shift} (SHIFTED)\nOFFSET = {offset_str} ⚠️ SUSPICIOUS"
        plt.suptitle(title_lines, fontsize=TITLE_FONT_SIZE, color='red', fontweight='bold', y=0.99)
    else:
        title_lines = f"{name_ref} (REFERENCE)\n{name_shift} (SHIFTED)\nOFFSET = {offset_str}"
        plt.suptitle(title_lines, fontsize=TITLE_FONT_SIZE, y=0.99)

    # Create 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1, 1], 
                          hspace=0.35, wspace=0.20, left=0.05, right=0.95, top=0.88, bottom=0.05)
    
    # ----- LEFT COLUMN -----
    
    # Row 0 Left: Unaligned TFRs (mirrored pair)
    gs_raw = gs[0, 0].subgridspec(2, 1, hspace=0.0)
    ax_raw_top = fig.add_subplot(gs_raw[0])
    ax_raw_bot = fig.add_subplot(gs_raw[1])
    
    # Row 1 Left: Coarse correlation
    ax_coarse = fig.add_subplot(gs[1, 0])
    
    # Row 2 Left: Fine search results
    ax_fine = fig.add_subplot(gs[2, 0])
    
    # ----- RIGHT COLUMN -----
    
    # Row 0 Right: Full Aligned TFRs (mirrored pair)
    gs_aligned = gs[0, 1].subgridspec(2, 1, hspace=0.0)
    ax_aligned_top = fig.add_subplot(gs_aligned[0])
    ax_aligned_bot = fig.add_subplot(gs_aligned[1])
    
    # Row 1 Right: Zoomed view 1 (mirrored pair)
    gs_zoom1 = gs[1, 1].subgridspec(2, 1, hspace=0.0)
    ax_zoom1_top = fig.add_subplot(gs_zoom1[0])
    ax_zoom1_bot = fig.add_subplot(gs_zoom1[1])
    
    # Row 2 Right: Zoomed view 2 (mirrored pair)
    gs_zoom2 = gs[2, 1].subgridspec(2, 1, hspace=0.0)
    ax_zoom2_top = fig.add_subplot(gs_zoom2[0])
    ax_zoom2_bot = fig.add_subplot(gs_zoom2[1])
    
    # ===== PLOT LEFT COLUMN =====
    
    # --- Unaligned TFRs ---
    t_max_raw = max(tR_raw.max(), tS_raw.max())
    ax_raw_top.set_facecolor('#ffcccc')  # Light red for empty space
    ax_raw_bot.set_facecolor('#ffcccc')
    
    # Top: Reference raw
    ax_raw_top.pcolormesh(tR_raw/3600, fR_raw, SR_raw, shading="auto", 
                          cmap="Spectral_r", vmin=vmin_Rr, vmax=vmax_Rr)
    ax_raw_top.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_raw_top.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_raw_top.tick_params(labelbottom=False)
    ax_raw_top.set_xlim(0, t_max_raw/3600)
    ax_raw_top.set_ylim(0, plot_max_freq)
    ax_raw_top.text(1.01, 0.5, "REFERENCE", transform=ax_raw_top.transAxes, 
                    rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    ax_raw_top.set_title("Unaligned TFRs", fontsize=LABEL_FONT_SIZE, pad=5)
    
    # Bottom: Shift raw (mirrored)
    ax_raw_bot.pcolormesh(tS_raw/3600, fS_raw, SS_raw, shading="auto", 
                          cmap="Spectral_r", vmin=vmin_Sr, vmax=vmax_Sr)
    ax_raw_bot.invert_yaxis()
    ax_raw_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_raw_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_raw_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_raw_bot.set_xlim(0, t_max_raw/3600)
    ax_raw_bot.set_ylim(plot_max_freq, 0)  # Inverted
    ax_raw_bot.text(1.01, 0.5, "SHIFTED", transform=ax_raw_bot.transAxes, 
                    rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

    # --- Coarse Correlation Plot ---
    if coarse_corr is not None and coarse_lags is not None:
        ax_coarse.plot(coarse_lags, coarse_corr, color='blue', linewidth=0.8)
        
        # Find max (peak) correlation
        max_idx = np.argmax(coarse_corr)
        max_lag = coarse_lags[max_idx]
        max_val = coarse_corr[max_idx]
        
        # Find second peak (next highest local maximum)
        corr_copy = coarse_corr.copy()
        exclude_width = max(10, len(coarse_corr) // 50)
        exclude_start = max(0, max_idx - exclude_width)
        exclude_end = min(len(corr_copy), max_idx + exclude_width)
        corr_copy[exclude_start:exclude_end] = -np.inf
        second_idx = np.argmax(corr_copy)
        second_val = coarse_corr[second_idx]
        
        # Calculate ratio
        if second_val > 0:
            peak_ratio = max_val / second_val
        else:
            peak_ratio = np.inf
        
        # Mark max with small arrow at bottom and red dot at peak
        ylim = ax_coarse.get_ylim()
        if ylim[0] == ylim[1]:
            ylim = (coarse_corr.min(), coarse_corr.max())
        arrow_height = (ylim[1] - ylim[0]) * 0.15  # Small arrow at bottom
        ax_coarse.annotate('', xy=(max_lag, ylim[0] + arrow_height), xytext=(max_lag, ylim[0]),
                          arrowprops=dict(arrowstyle='-|>', color='red', lw=2))
        ax_coarse.plot(max_lag, max_val, marker='o', color='red', markersize=8, zorder=5)
        
        # Show ratio and offset in large font
        ratio_text = f"Peak ratio: {peak_ratio:.2f}" if peak_ratio < 100 else "Peak ratio: >100"
        offset_text = f"Offset: {max_lag:.1f}s"
        ax_coarse.text(0.98, 0.95, f"{ratio_text}\n{offset_text}", transform=ax_coarse.transAxes,
                      fontsize=RATIO_FONT_SIZE, ha='right', va='top', fontweight='bold',
                      bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
        
        ax_coarse.set_xlabel("Lag (seconds)", fontsize=LABEL_FONT_SIZE)
        ax_coarse.set_ylabel("Correlation", fontsize=LABEL_FONT_SIZE)
        ax_coarse.set_title("Coarse Cross-Correlation", fontsize=LABEL_FONT_SIZE, pad=5)
        ax_coarse.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax_coarse.grid(True, alpha=0.3)
    else:
        ax_coarse.text(0.5, 0.5, "No coarse correlation data", ha='center', va='center', 
                       transform=ax_coarse.transAxes, fontsize=LABEL_FONT_SIZE)
        ax_coarse.set_title("Coarse Cross-Correlation", fontsize=LABEL_FONT_SIZE, pad=5)

    # --- Fine Search Results Plot ---
    if fine_search_data is not None:
        lags_ms = fine_search_data.get('lags_ms', [])
        scores = fine_search_data.get('scores', [])
        best_lag_ms = fine_search_data.get('best_lag_ms', None)
        
        if len(lags_ms) > 0 and len(scores) > 0:
            scores_arr = np.array(scores)
            lags_arr = np.array(lags_ms)
            ax_fine.plot(lags_arr, scores_arr, color='green', linewidth=0.8)
            
            if best_lag_ms is not None:
                best_idx = np.argmin(np.abs(lags_arr - best_lag_ms))
                if best_idx < len(scores_arr):
                    best_val = scores_arr[best_idx]
                    
                    # Find second peak
                    scores_copy = scores_arr.copy()
                    exclude_width = max(5, len(scores_arr) // 50)
                    exclude_start = max(0, best_idx - exclude_width)
                    exclude_end = min(len(scores_copy), best_idx + exclude_width)
                    scores_copy[exclude_start:exclude_end] = -np.inf
                    second_idx = np.argmax(scores_copy)
                    second_val = scores_arr[second_idx]
                    
                    # Calculate ratio
                    if second_val > 0:
                        peak_ratio = best_val / second_val
                    else:
                        peak_ratio = np.inf
                    
                    # Mark with small arrow at bottom and red dot at peak
                    ylim_fine = (scores_arr.min(), scores_arr.max())
                    arrow_height_fine = (ylim_fine[1] - ylim_fine[0]) * 0.15
                    ax_fine.annotate('', xy=(best_lag_ms, ylim_fine[0] + arrow_height_fine),
                                    xytext=(best_lag_ms, ylim_fine[0]),
                                    arrowprops=dict(arrowstyle='-|>', color='red', lw=2))
                    ax_fine.plot(best_lag_ms, best_val, marker='o', color='red', markersize=8, zorder=5)
                    
                    # Show ratio and offset in large font
                    ratio_text = f"Peak ratio: {peak_ratio:.2f}" if peak_ratio < 100 else "Peak ratio: >100"
                    offset_text = f"Offset: {best_lag_ms:.1f}ms"
                    ax_fine.text(0.98, 0.95, f"{ratio_text}\n{offset_text}", transform=ax_fine.transAxes,
                                fontsize=RATIO_FONT_SIZE, ha='right', va='top', fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
            
            ax_fine.set_xlabel("Lag (ms)", fontsize=LABEL_FONT_SIZE)
            ax_fine.set_ylabel("MI Score", fontsize=LABEL_FONT_SIZE)
            ax_fine.grid(True, alpha=0.3)
        else:
            ax_fine.text(0.5, 0.5, "No fine search data available", ha='center', va='center',
                        transform=ax_fine.transAxes, fontsize=LABEL_FONT_SIZE)
    else:
        ax_fine.text(0.5, 0.5, "Fine search data not recorded", ha='center', va='center',
                    transform=ax_fine.transAxes, fontsize=LABEL_FONT_SIZE, color='gray')
    ax_fine.set_title("Fine Search Results", fontsize=LABEL_FONT_SIZE, pad=5)
    ax_fine.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    
    # ===== PLOT RIGHT COLUMN =====
    
    # --- Full Aligned TFRs ---
    t_max_cut = max(tR_cut.max(), tS_cut.max())
    
    # Top: Reference cut
    ax_aligned_top.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", 
                              cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax_aligned_top.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_aligned_top.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_aligned_top.tick_params(labelbottom=False)
    ax_aligned_top.set_xlim(0, t_max_cut/3600)
    ax_aligned_top.set_ylim(0, plot_max_freq)
    ax_aligned_top.text(1.01, 0.5, "REFERENCE", transform=ax_aligned_top.transAxes, 
                        rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    if is_failed:
        ax_aligned_top.set_title("Aligned TFRs (suspicious)", fontsize=LABEL_FONT_SIZE, 
                                  pad=5, color='red')
    else:
        ax_aligned_top.set_title("Aligned TFRs", fontsize=LABEL_FONT_SIZE, pad=5)
    
    # Bottom: Shift cut (mirrored)
    ax_aligned_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                              cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_aligned_bot.invert_yaxis()
    ax_aligned_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_aligned_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_aligned_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_aligned_bot.set_xlim(0, t_max_cut/3600)
    ax_aligned_bot.set_ylim(plot_max_freq, 0)
    ax_aligned_bot.text(1.01, 0.5, "SHIFTED", transform=ax_aligned_bot.transAxes, 
                        rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')

    # --- Zoomed Views ---
    # Calculate zoom windows (1 hour each)
    zoom_duration_sec = 60 * 60  # 1 hour in seconds
    total_cut_sec = t_max_cut
    
    # Zoom 1: Middle of recording
    zoom1_center_sec = total_cut_sec / 2
    zoom1_start_hr = max(0, (zoom1_center_sec - zoom_duration_sec/2)) / 3600
    zoom1_end_hr = min(total_cut_sec, (zoom1_center_sec + zoom_duration_sec/2)) / 3600
    
    # Zoom 2: Middle of second half
    zoom2_center_sec = total_cut_sec * 0.75
    zoom2_start_hr = max(0, (zoom2_center_sec - zoom_duration_sec/2)) / 3600
    zoom2_end_hr = min(total_cut_sec, (zoom2_center_sec + zoom_duration_sec/2)) / 3600
    
    # Plot Zoom 1
    ax_zoom1_top.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax_zoom1_top.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom1_top.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom1_top.tick_params(labelbottom=False)
    ax_zoom1_top.set_xlim(zoom1_start_hr, zoom1_end_hr)
    ax_zoom1_top.set_ylim(0, plot_max_freq)
    ax_zoom1_top.text(1.01, 0.5, "REFERENCE", transform=ax_zoom1_top.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    ax_zoom1_top.set_title(f"Zoomed: {zoom1_start_hr:.2f}-{zoom1_end_hr:.2f}h", 
                           fontsize=LABEL_FONT_SIZE, pad=5)
    
    ax_zoom1_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_zoom1_bot.invert_yaxis()
    ax_zoom1_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom1_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_zoom1_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom1_bot.set_xlim(zoom1_start_hr, zoom1_end_hr)
    ax_zoom1_bot.set_ylim(plot_max_freq, 0)
    ax_zoom1_bot.text(1.01, 0.5, "SHIFTED", transform=ax_zoom1_bot.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    
    # Plot Zoom 2
    ax_zoom2_top.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax_zoom2_top.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_top.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom2_top.tick_params(labelbottom=False)
    ax_zoom2_top.set_xlim(zoom2_start_hr, zoom2_end_hr)
    ax_zoom2_top.set_ylim(0, plot_max_freq)
    ax_zoom2_top.text(1.01, 0.5, "REFERENCE", transform=ax_zoom2_top.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    ax_zoom2_top.set_title(f"Zoomed: {zoom2_start_hr:.2f}-{zoom2_end_hr:.2f}h", 
                           fontsize=LABEL_FONT_SIZE, pad=5)
    
    ax_zoom2_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_zoom2_bot.invert_yaxis()
    ax_zoom2_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom2_bot.set_xlim(zoom2_start_hr, zoom2_end_hr)
    ax_zoom2_bot.set_ylim(plot_max_freq, 0)
    ax_zoom2_bot.text(1.01, 0.5, "SHIFTED", transform=ax_zoom2_bot.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plotting] Saved complete alignment plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig