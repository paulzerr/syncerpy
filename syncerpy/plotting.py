import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lspopt import spectrogram_lspopt

def plot_results(sA, sB, fs, offset_sec, name_a, name_b, coarse_corr=None, coarse_lags=None,
                 plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=30, save_path=None, fsB=None, show_plot=False):
    
    if fsB is None:
        fsB = fs

    # Plot configuration
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 14
    TICK_FONT_SIZE = 14
    LEGEND_FONT_SIZE = 14
    ANNOTATION_FONT_SIZE = 16

    def get_spec(sig, fs_local):
        win = int(plot_win_sec * fs_local)
        overlap = plot_win_sec / plot_overlap_divisor
        f, t, S = spectrogram_lspopt(sig, fs_local, nperseg=win, noverlap=overlap)
        return f[f<=plot_max_freq], t, 10*np.log10(S[f<=plot_max_freq])

    fA, tA, SA = get_spec(sA, fs)
    fB, tB, SB = get_spec(sB, fsB)
    
    tB_shift = tB + offset_sec

    # Determine global time range
    t_min = min(tA.min(), tB_shift.min())
    t_max = max(tA.max(), tB_shift.max())

    # Robust normalization using Median and IQR
    def get_robust_lims(S):
        q25, q50, q75 = np.percentile(S, [25, 50, 75])
        iqr = q75 - q25
        vmin = q50 - 1.5 * iqr
        vmax = q50 + 1.5 * iqr
        return vmin, vmax

    vmin_A, vmax_A = get_robust_lims(SA)
    vmin_B, vmax_B = get_robust_lims(SB)

    fig = plt.figure(figsize=(16,12))
    plt.suptitle(f"Computed offset: {offset_sec*1000:.2f} ms ", fontsize=16)

    # Create a layout where the first two plots touch
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
    gs_top = gs[0].subgridspec(2, 1, hspace=0.0)
    
    ax1 = fig.add_subplot(gs_top[0])
    ax2 = fig.add_subplot(gs_top[1])
    ax3 = fig.add_subplot(gs[1])

    # Plot 1
    ax1.pcolormesh(tA/3600, fA, SA, shading="auto", cmap="Spectral_r", vmin=vmin_A, vmax=vmax_A)
    ax1.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(labelbottom=False)
    ax1.text(1.02, 0.5, "reference", transform=ax1.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # Plot 2
    ax2.pcolormesh(tB_shift/3600, fB, SB, shading="auto", cmap="Spectral_r", vmin=vmin_B, vmax=vmax_B)
    ax2.invert_yaxis() # Flip so 0Hz is at top (touching 0Hz of ax1)
    ax2.text(1.02, 0.5, "shifted", transform=ax2.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
        
    ax2.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    xmin = t_min / 3600
    xmax = t_max / 3600
    
    # Sync x-axis for spectrograms
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)

    # 3. Coarse Correlation Distribution
    if coarse_corr is not None and coarse_lags is not None:
        plt.sca(ax3)
        plt.plot(coarse_lags, coarse_corr, color='blue', linewidth=1)
        
        # Mark max correlation
        max_idx = np.argmax(coarse_corr)
        max_val = coarse_corr[max_idx]
        max_lag = coarse_lags[max_idx]
        plt.plot(max_lag, np.min(coarse_corr), marker='^', color='red', markersize=10, label=f'Max: {max_lag:.1f}s', linestyle='None')
        
        # Peak-to-next-peak ratio
        peaks, _ = signal.find_peaks(coarse_corr, distance=10)
        
        peak_vals = coarse_corr[peaks]
        # Sort descending
        sorted_indices = np.argsort(peak_vals)[::-1]
        sorted_peaks = peak_vals[sorted_indices]
        
        ratio_text = "Ratio: N/A"
        if len(sorted_peaks) >= 2:
            p1 = sorted_peaks[0]
            p2 = sorted_peaks[1]
            if p2 > 0:
                ratio = p1 / p2
                ratio_text = f"Peak Ratio: {ratio:.2f}"
            else:
                ratio_text = "Peak Ratio: > 2nd peak <= 0"
        elif len(sorted_peaks) == 1:
                ratio_text = "Peak Ratio: Single Peak"
        
        plt.xlabel("Lag (seconds)", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Correlation", fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.grid(True, alpha=0.3)

    plt.subplots_adjust(right=0.9)
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plotting] Saved plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_combined_spectrograms(sRef_raw, sShift_raw, sRef_cut, sShift_cut,
                               fs_ref_raw, fs_shift_raw, fs_ref_cut, fs_shift_cut,
                               name_ref, name_shift, title="Combined Spectrograms",
                               plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=30,
                               save_path=None, show_plot=False):
    
    # Plot configuration
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10
    ANNOTATION_FONT_SIZE = 12

    def get_spec(sig, fs_local):
        win = int(plot_win_sec * fs_local)
        overlap = plot_win_sec / plot_overlap_divisor
        f, t, S = spectrogram_lspopt(sig, fs_local, nperseg=win, noverlap=overlap)
        return f[f<=plot_max_freq], t, 10*np.log10(S[f<=plot_max_freq])

    # Calculate spectrograms
    fR_raw, tR_raw, SR_raw = get_spec(sRef_raw, fs_ref_raw)
    fS_raw, tS_raw, SS_raw = get_spec(sShift_raw, fs_shift_raw)
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

    fig = plt.figure(figsize=(16, 16))
    plt.suptitle(title, fontsize=16)

    # Create outer grid for 2 groups (Raw vs Cut)
    gs_outer = fig.add_gridspec(2, 1, hspace=0.3)
    
    # Group 1: Raw
    gs_raw = gs_outer[0].subgridspec(2, 1, hspace=0.0)
    ax1 = fig.add_subplot(gs_raw[0])
    ax2 = fig.add_subplot(gs_raw[1])
    
    # Group 2: Cut
    gs_cut = gs_outer[1].subgridspec(2, 1, hspace=0.0)
    ax3 = fig.add_subplot(gs_cut[0])
    ax4 = fig.add_subplot(gs_cut[1])

    # --- Raw Plots ---
    t_max_raw = max(tR_raw.max(), tS_raw.max())
    
    # Set background to red for "empty space"
    ax1.set_facecolor('red')
    ax2.set_facecolor('red')

    # Plot 1 (Ref Raw)
    ax1.pcolormesh(tR_raw/3600, fR_raw, SR_raw, shading="auto", cmap="Spectral_r", vmin=vmin_Rr, vmax=vmax_Rr)
    ax1.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(labelbottom=False) # Hide x-labels
    ax1.set_xlim(0, t_max_raw/3600)
    # Annotation
    ax1.text(1.02, 0.5, f"{name_ref} (Raw)", transform=ax1.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # Plot 2 (Shift Raw)
    ax2.pcolormesh(tS_raw/3600, fS_raw, SS_raw, shading="auto", cmap="Spectral_r", vmin=vmin_Sr, vmax=vmax_Sr)
    ax2.invert_yaxis() # Mirror
    ax2.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax2.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax2.set_xlim(0, t_max_raw/3600)
    # Annotation
    ax2.text(1.02, 0.5, f"{name_shift} (Raw)", transform=ax2.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # --- Cut Plots ---
    t_max_cut = max(tR_cut.max(), tS_cut.max())

    # Plot 3 (Ref Cut)
    ax3.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax3.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax3.tick_params(labelbottom=False) # Hide x-labels
    ax3.set_xlim(0, t_max_cut/3600)
    # Annotation
    ax3.text(1.02, 0.5, f"{name_ref} (Aligned)", transform=ax3.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # Plot 4 (Shift Cut)
    ax4.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax4.invert_yaxis() # Mirror
    ax4.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax4.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax4.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax4.set_xlim(0, t_max_cut/3600)
    # Annotation
    ax4.text(1.02, 0.5, f"{name_shift} (Aligned)", transform=ax4.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    plt.subplots_adjust(right=0.9)

    if save_path:
        plt.savefig(save_path)
        print(f"[Plotting] Saved combined spectrogram plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_spectrograms(sA, sB, fs, name_a, name_b, title="Spectrograms",
                      plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=30,
                      save_path=None, fsB=None, show_plot=False):
    
    if fsB is None:
        fsB = fs

    # Plot configuration
    LABEL_FONT_SIZE = 14
    TICK_FONT_SIZE = 14
    ANNOTATION_FONT_SIZE = 16

    def get_spec(sig, fs_local):
        win = int(plot_win_sec * fs_local)
        overlap = plot_win_sec / plot_overlap_divisor
        f, t, S = spectrogram_lspopt(sig, fs_local, nperseg=win, noverlap=overlap)
        return f[f<=plot_max_freq], t, 10*np.log10(S[f<=plot_max_freq])

    fA, tA, SA = get_spec(sA, fs)
    fB, tB, SB = get_spec(sB, fsB)
    
    # Robust normalization
    def get_robust_lims(S):
        q25, q50, q75 = np.percentile(S, [25, 50, 75])
        iqr = q75 - q25
        vmin = q50 - 1.5 * iqr
        vmax = q50 + 1.5 * iqr
        return vmin, vmax

    vmin_A, vmax_A = get_robust_lims(SA)
    vmin_B, vmax_B = get_robust_lims(SB)

    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(title, fontsize=16)

    # Create a layout where the two plots touch
    gs = fig.add_gridspec(2, 1, hspace=0.0)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot 1
    ax1.pcolormesh(tA/3600, fA, SA, shading="auto", cmap="Spectral_r", vmin=vmin_A, vmax=vmax_A)
    ax1.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(labelbottom=False)
    ax1.text(1.02, 0.5, name_a, transform=ax1.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # Plot 2
    ax2.pcolormesh(tB/3600, fB, SB, shading="auto", cmap="Spectral_r", vmin=vmin_B, vmax=vmax_B)
    ax2.invert_yaxis() # Flip so 0Hz is at top (touching 0Hz of ax1)
    ax2.text(1.02, 0.5, name_b, transform=ax2.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
        
    ax2.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

    # Sync x-axis
    t_max = max(tA.max(), tB.max())
    ax1.set_xlim(0, t_max/3600)
    ax2.set_xlim(0, t_max/3600)

    plt.subplots_adjust(right=0.9)
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plotting] Saved spectrogram plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_complete_alignment(sRef_raw, sShift_raw, sRef_cut, sShift_cut,
                           fs_ref, fs_shift, offset_sec,
                           name_ref, name_shift,
                           coarse_corr=None, coarse_lags=None,
                           plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=30,
                           save_path=None, show_plot=False, is_failed=False):
    """
    Creates a single comprehensive plot showing:
    1. Unaligned (raw) TFRs (mirrored)
    2. Aligned (cut) TFRs (mirrored)
    3. Correlation plot
    
    Parameters:
    -----------
    is_failed : bool
        If True, adds a warning banner indicating alignment failed
    """
    
    # Plot configuration
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10
    ANNOTATION_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 12

    def get_spec(sig, fs_local):
        win = int(plot_win_sec * fs_local)
        overlap = plot_win_sec / plot_overlap_divisor
        f, t, S = spectrogram_lspopt(sig, fs_local, nperseg=win, noverlap=overlap)
        return f[f<=plot_max_freq], t, 10*np.log10(S[f<=plot_max_freq])

    # Calculate all spectrograms
    fR_raw, tR_raw, SR_raw = get_spec(sRef_raw, fs_ref)
    fS_raw, tS_raw, SS_raw = get_spec(sShift_raw, fs_shift)
    fR_cut, tR_cut, SR_cut = get_spec(sRef_cut, fs_ref)
    fS_cut, tS_cut, SS_cut = get_spec(sShift_cut, fs_shift)

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

    # Create figure with 3 sections: Raw TFRs, Aligned TFRs, Correlation
    fig = plt.figure(figsize=(16, 18))
    
    # Title with failure warning if applicable
    if is_failed:
        # Format large offset nicely
        if abs(offset_sec) >= 3600:
            offset_str = f"{offset_sec/3600:.2f} hours"
        elif abs(offset_sec) >= 60:
            offset_str = f"{offset_sec/60:.2f} min"
        else:
            offset_str = f"{offset_sec:.2f} s"
        title = f"⚠️ ALIGNMENT FAILED - Offset: {offset_str} (suspicious)"
        plt.suptitle(title, fontsize=TITLE_FONT_SIZE, color='red', fontweight='bold')
        
        # Add warning text box
        fig.text(0.5, 0.94,
                "Files may be corrupted, mismatched, or from different recordings.\n"
                "The computed offset is unreasonably large. Manual inspection recommended.",
                ha='center', va='top', fontsize=11,
                color='darkred', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.8))
    else:
        plt.suptitle(f"Alignment Analysis - Computed offset: {offset_sec*1000:.2f} ms", fontsize=TITLE_FONT_SIZE)

    # Create outer grid: 3 rows (Raw, Aligned, Correlation)
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.35)
    
    # Section 1: Raw (Unaligned) TFRs
    gs_raw = gs_outer[0].subgridspec(2, 1, hspace=0.0)
    ax1 = fig.add_subplot(gs_raw[0])
    ax2 = fig.add_subplot(gs_raw[1])
    
    # Section 2: Aligned (Cut) TFRs
    gs_cut = gs_outer[1].subgridspec(2, 1, hspace=0.0)
    ax3 = fig.add_subplot(gs_cut[0])
    ax4 = fig.add_subplot(gs_cut[1])
    
    # Section 3: Correlation plot
    ax5 = fig.add_subplot(gs_outer[2])

    # --- Raw (Unaligned) Plots ---
    t_max_raw = max(tR_raw.max(), tS_raw.max())
    
    # Set background to red for "empty space"
    ax1.set_facecolor('red')
    ax2.set_facecolor('red')

    # Plot 1 (Ref Raw)
    ax1.pcolormesh(tR_raw/3600, fR_raw, SR_raw, shading="auto", cmap="Spectral_r", vmin=vmin_Rr, vmax=vmax_Rr)
    ax1.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax1.tick_params(labelbottom=False)
    ax1.set_xlim(0, t_max_raw/3600)
    ax1.text(1.02, 0.5, f"{name_ref} (Unaligned)", transform=ax1.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    ax1.set_title("Unaligned TFRs", fontsize=LABEL_FONT_SIZE, pad=10)

    # Plot 2 (Shift Raw)
    ax2.pcolormesh(tS_raw/3600, fS_raw, SS_raw, shading="auto", cmap="Spectral_r", vmin=vmin_Sr, vmax=vmax_Sr)
    ax2.invert_yaxis()
    ax2.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax2.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax2.set_xlim(0, t_max_raw/3600)
    ax2.text(1.02, 0.5, f"{name_shift} (Unaligned)", transform=ax2.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # --- Aligned (Cut) Plots ---
    t_max_cut = max(tR_cut.max(), tS_cut.max())

    # Plot 3 (Ref Cut)
    ax3.pcolormesh(tR_cut/3600, fR_cut, SR_cut, shading="auto", cmap="Spectral_r", vmin=vmin_Rc, vmax=vmax_Rc)
    ax3.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax3.tick_params(labelbottom=False)
    ax3.set_xlim(0, t_max_cut/3600)
    ax3.text(1.02, 0.5, f"{name_ref} (Aligned)", transform=ax3.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)
    if is_failed:
        ax3.set_title("Raw Signals (alignment failed - no cutting performed)", fontsize=LABEL_FONT_SIZE, pad=10, color='red')
    else:
        ax3.set_title("Aligned & Cut TFRs", fontsize=LABEL_FONT_SIZE, pad=10)

    # Plot 4 (Shift Cut)
    ax4.pcolormesh(tS_cut/3600, fS_cut, SS_cut, shading="auto", cmap="Spectral_r", vmin=vmin_Sc, vmax=vmax_Sc)
    ax4.invert_yaxis()
    ax4.set_ylabel("Hz", fontsize=LABEL_FONT_SIZE)
    ax4.set_xlabel("Hours", fontsize=LABEL_FONT_SIZE)
    ax4.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax4.set_xlim(0, t_max_cut/3600)
    ax4.text(1.02, 0.5, f"{name_shift} (Aligned)", transform=ax4.transAxes, rotation=90, va='center', ha='left', fontsize=ANNOTATION_FONT_SIZE)

    # --- Correlation Plot ---
    if coarse_corr is not None and coarse_lags is not None:
        ax5.plot(coarse_lags, coarse_corr, color='blue', linewidth=1)
        
        # Mark max correlation
        max_idx = np.argmax(coarse_corr)
        max_val = coarse_corr[max_idx]
        max_lag = coarse_lags[max_idx]
        ax5.plot(max_lag, np.min(coarse_corr), marker='^', color='red', markersize=10, label=f'Max: {max_lag:.1f}s', linestyle='None')
        
        # Peak-to-next-peak ratio
        peaks, _ = signal.find_peaks(coarse_corr, distance=10)
        
        peak_vals = coarse_corr[peaks]
        sorted_indices = np.argsort(peak_vals)[::-1]
        sorted_peaks = peak_vals[sorted_indices]
        
        ratio_text = "Ratio: N/A"
        if len(sorted_peaks) >= 2:
            p1 = sorted_peaks[0]
            p2 = sorted_peaks[1]
            if p2 > 0:
                ratio = p1 / p2
                ratio_text = f"Peak Ratio: {ratio:.2f}"
            else:
                ratio_text = "Peak Ratio: > 2nd peak <= 0"
        elif len(sorted_peaks) == 1:
            ratio_text = "Peak Ratio: Single Peak"
        
        ax5.set_xlabel("Lag (seconds)", fontsize=LABEL_FONT_SIZE)
        ax5.set_ylabel("Correlation", fontsize=LABEL_FONT_SIZE)
        ax5.set_title("Cross-Correlation", fontsize=LABEL_FONT_SIZE, pad=10)
        ax5.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax5.legend(fontsize=LEGEND_FONT_SIZE)
        ax5.grid(True, alpha=0.3)

    plt.subplots_adjust(right=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plotting] Saved complete alignment plot to {save_path}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig