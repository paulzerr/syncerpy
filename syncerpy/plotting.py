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
        |                       |       | (30 min from middle)  |
        +-----------------------+       +-----------------------+
        | Fine Search Results   |       | Zoomed View 2         |
        |                       |       | (30 min from 2nd half)|
        +-----------------------+       +-----------------------+
    
    Parameters:
    -----------
    sRef_raw, sShift_raw : array
        Raw (unaligned) signals at fs_raw
    sRef_cut, sShift_cut : array
        Aligned/cut signals at fs_ref_cut and fs_shift_cut respectively
    fs_raw : float
        Sample rate for raw signals (typically 64Hz after resampling)
    fs_ref_cut, fs_shift_cut : float
        Sample rates for cut signals
    offset_sec : float
        Computed offset in seconds
    name_ref, name_shift : str
        Names/paths of the files (will extract just filename)
    coarse_corr, coarse_lags : array
        Coarse cross-correlation results
    fine_search_data : dict, optional
        Contains 'lags_ms', 'scores', 'best_lag_ms' for fine search plot
    is_failed : bool
        If True, adds a warning banner indicating alignment failed
    """
    
    # Extract just filenames
    name_ref = _get_filename(name_ref)
    name_shift = _get_filename(name_shift)
    
    # Plot configuration
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 10
    TICK_FONT_SIZE = 9
    ANNOTATION_FONT_SIZE = 9
    LEGEND_FONT_SIZE = 9

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
    fig = plt.figure(figsize=(20, 16))
    
    # Title with failure warning if applicable
    if is_failed:
        if abs(offset_sec) >= 3600:
            offset_str = f"{offset_sec/3600:.2f} hours"
        elif abs(offset_sec) >= 60:
            offset_str = f"{offset_sec/60:.2f} min"
        else:
            offset_str = f"{offset_sec:.2f} s"
        title = f"⚠️ ALIGNMENT FAILED - Offset: {offset_str} (suspicious)"
        plt.suptitle(title, fontsize=TITLE_FONT_SIZE, color='red', fontweight='bold', y=0.98)
        
        fig.text(0.5, 0.96,
                "Files may be corrupted, mismatched, or from different recordings.",
                ha='center', va='top', fontsize=10,
                color='darkred', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.8))
    else:
        plt.suptitle(f"Alignment Analysis - Computed offset: {offset_sec*1000:.2f} ms", 
                     fontsize=TITLE_FONT_SIZE, y=0.98)

    # Create 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1, 1], 
                          hspace=0.35, wspace=0.20, left=0.05, right=0.92, top=0.93, bottom=0.05)
    
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
    ax_raw_top.text(1.02, 0.5, f"{name_ref} (REF)", transform=ax_raw_top.transAxes, 
                    rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    ax_raw_top.set_title("Unaligned (Raw) TFRs", fontsize=LABEL_FONT_SIZE, pad=5)
    
    # Bottom: Shift raw (mirrored)
    ax_raw_bot.pcolormesh(tS_raw/3600, fS_raw, SS_raw, shading="auto", 
                          cmap="Spectral_r", vmin=vmin_Sr, vmax=vmax_Sr)
    ax_raw_bot.invert_yaxis()
    ax_raw_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_raw_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_raw_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_raw_bot.set_xlim(0, t_max_raw/3600)
    ax_raw_bot.set_ylim(plot_max_freq, 0)  # Inverted
    ax_raw_bot.text(1.02, 0.5, f"{name_shift} (SHIFTED)", transform=ax_raw_bot.transAxes, 
                    rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # --- Coarse Correlation Plot ---
    if coarse_corr is not None and coarse_lags is not None:
        ax_coarse.plot(coarse_lags, coarse_corr, color='blue', linewidth=0.8)
        
        # Mark max correlation
        max_idx = np.argmax(coarse_corr)
        max_lag = coarse_lags[max_idx]
        ax_coarse.axvline(max_lag, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_coarse.plot(max_lag, coarse_corr[max_idx], marker='o', color='red', markersize=6, 
                       label=f'Max: {max_lag:.1f}s', zorder=5)
        
        ax_coarse.set_xlabel("Lag (seconds)", fontsize=LABEL_FONT_SIZE)
        ax_coarse.set_ylabel("Correlation", fontsize=LABEL_FONT_SIZE)
        ax_coarse.set_title("Coarse Cross-Correlation", fontsize=LABEL_FONT_SIZE, pad=5)
        ax_coarse.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax_coarse.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
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
            ax_fine.plot(lags_ms, scores, color='green', linewidth=0.8)
            if best_lag_ms is not None:
                best_idx = np.argmin(np.abs(np.array(lags_ms) - best_lag_ms))
                if best_idx < len(scores):
                    ax_fine.axvline(best_lag_ms, color='red', linestyle='--', linewidth=1, alpha=0.7)
                    ax_fine.plot(best_lag_ms, scores[best_idx], marker='o', color='red', markersize=6,
                                label=f'Best: {best_lag_ms:.2f}ms', zorder=5)
            ax_fine.set_xlabel("Lag (ms)", fontsize=LABEL_FONT_SIZE)
            ax_fine.set_ylabel("MI Score", fontsize=LABEL_FONT_SIZE)
            ax_fine.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right')
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
    ax_aligned_top.text(1.02, 0.5, f"{name_ref} (REF)", transform=ax_aligned_top.transAxes, 
                        rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    if is_failed:
        ax_aligned_top.set_title("Raw Signals (alignment failed)", fontsize=LABEL_FONT_SIZE, 
                                  pad=5, color='red')
    else:
        ax_aligned_top.set_title("Aligned & Cut TFRs (Full)", fontsize=LABEL_FONT_SIZE, pad=5)
    
    # Bottom: Shift cut (mirrored)
    ax_aligned_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                              cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_aligned_bot.invert_yaxis()
    ax_aligned_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_aligned_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_aligned_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_aligned_bot.set_xlim(0, t_max_cut/3600)
    ax_aligned_bot.set_ylim(plot_max_freq, 0)
    ax_aligned_bot.text(1.02, 0.5, f"{name_shift} (SHIFTED)", transform=ax_aligned_bot.transAxes, 
                        rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # --- Zoomed Views ---
    # Calculate zoom windows (30 minutes each)
    zoom_duration_sec = 30 * 60  # 30 minutes in seconds
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
    ax_zoom1_top.text(1.02, 0.5, f"{name_ref} (REF)", transform=ax_zoom1_top.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    ax_zoom1_top.set_title(f"Zoomed: Middle ({zoom1_start_hr:.2f}-{zoom1_end_hr:.2f}h)", 
                           fontsize=LABEL_FONT_SIZE, pad=5)
    
    ax_zoom1_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_zoom1_bot.invert_yaxis()
    ax_zoom1_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom1_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_zoom1_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom1_bot.set_xlim(zoom1_start_hr, zoom1_end_hr)
    ax_zoom1_bot.set_ylim(plot_max_freq, 0)
    ax_zoom1_bot.text(1.02, 0.5, f"{name_shift} (SHIFTED)", transform=ax_zoom1_bot.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    
    # Plot Zoom 2
    ax_zoom2_top.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax_zoom2_top.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_top.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom2_top.tick_params(labelbottom=False)
    ax_zoom2_top.set_xlim(zoom2_start_hr, zoom2_end_hr)
    ax_zoom2_top.set_ylim(0, plot_max_freq)
    ax_zoom2_top.text(1.02, 0.5, f"{name_ref} (REF)", transform=ax_zoom2_top.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    ax_zoom2_top.set_title(f"Zoomed: 2nd Half ({zoom2_start_hr:.2f}-{zoom2_end_hr:.2f}h)", 
                           fontsize=LABEL_FONT_SIZE, pad=5)
    
    ax_zoom2_bot.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", 
                            cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax_zoom2_bot.invert_yaxis()
    ax_zoom2_bot.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_bot.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax_zoom2_bot.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax_zoom2_bot.set_xlim(zoom2_start_hr, zoom2_end_hr)
    ax_zoom2_bot.set_ylim(plot_max_freq, 0)
    ax_zoom2_bot.text(1.02, 0.5, f"{name_shift} (SHIFTED)", transform=ax_zoom2_bot.transAxes, 
                      rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plotting] Saved complete alignment plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig