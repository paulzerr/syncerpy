import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from lspopt import spectrogram_lspopt

def plot_results(sA, sB, fs, offset_sec, name_a, name_b, coarse_corr=None, coarse_lags=None,
                 plot_win_sec=30, plot_overlap_divisor=4, plot_max_freq=30, save_path=None, fsB=None):
    
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
        
    plt.show()
    return fig