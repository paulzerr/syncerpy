import numpy as np
from pathlib import Path
import pyedflib
from syncerpy.plotting import plot_complete_alignment
import datetime
from syncerpy.syncerpy import load_file

def cut(file_reference, file_shift, offset, sRef_raw=None, sShift_raw=None,
        coarse_corr=None, coarse_lags=None, output_folder=None,
        plot=False, show_plot=True, save_plot=False, prefix=None):
    """
    Aligns two EDF files based on the provided offset using pyedflib to preserve exact headers.
    
    offset > 0: file_shift is later than file_reference. 
                We cut 'offset' seconds from the beginning of file_reference.
    offset < 0: file_shift is earlier than file_reference.
                We cut 'abs(offset)' seconds from the beginning of file_shift.
                
    The resulting files will start at the same time point and will be cropped to the common duration.
    """
    
    ref_path = Path(file_reference)
    shift_path = Path(file_shift)
    
    if output_folder:
        out_dir = Path(output_folder)
    else:
        out_dir = ref_path.parent
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Cutter] Processing {ref_path.name} and {shift_path.name}...")
    
    # Open readers
    f_ref = pyedflib.EdfReader(str(ref_path))
    f_shift = pyedflib.EdfReader(str(shift_path))
    
    # Get headers
    ref_signal_headers = f_ref.getSignalHeaders()
    shift_signal_headers = f_shift.getSignalHeaders()
    
    ref_header = f_ref.getHeader()
    shift_header = f_shift.getHeader()
    
    # Get sampling frequencies (assuming constant for all channels or taking first for time calc)
    # Ideally we check all, but for cutting time we need a reference.
    # Assuming all channels in a file have same duration, but might have different fs.
    # Duration in seconds is file_duration.
    ref_dur = f_ref.getFileDuration()
    shift_dur = f_shift.getFileDuration()
    
    # Determine crop times (in seconds)
    if offset > 0:
        # Shift is later. Cut Ref start.
        start_ref_sec = offset
        start_shift_sec = 0
    else:
        # Shift is earlier. Cut Shift start.
        start_ref_sec = 0
        start_shift_sec = abs(offset)
        
    print(f"[Cutter] Aligning starts: Cut {start_ref_sec:.3f}s from Ref, {start_shift_sec:.3f}s from Shift")
    
    # Calculate common duration
    # Remaining duration for Ref: ref_dur - start_ref_sec
    # Remaining duration for Shift: shift_dur - start_shift_sec
    rem_ref = ref_dur - start_ref_sec
    rem_shift = shift_dur - start_shift_sec
    common_dur = min(rem_ref, rem_shift)
    
    print(f"[Cutter] Common duration: {common_dur:.3f}s")
    
    # Helper to process and write file
    def process_file(reader, writer_path, start_sec, duration_sec, signal_headers, main_header):
        n_channels = reader.signals_in_file
        
        # Create writer
        writer = pyedflib.EdfWriter(str(writer_path), n_channels=n_channels)
        
        # Set Main Header
        writer.setHeader(main_header)
        
        # Set Signal Headers
        # Ensure 'sample_rate' key exists if 'sample_frequency' is present
        # pyedflib might expect 'sample_rate' in setSignalHeaders but returns 'sample_frequency' in getSignalHeaders
        for header in signal_headers:
            # Ensure we use sample_frequency as sample_rate is deprecated in pyedflib
            if 'sample_rate' in header:
                if 'sample_frequency' not in header:
                    header['sample_frequency'] = header['sample_rate']
                del header['sample_rate']
        
        writer.setSignalHeaders(signal_headers)
        
        # Read, Crop, Write Signals
        data_list = []
        for i in range(n_channels):
            # Get fs for this channel
            fs = signal_headers[i]['sample_frequency']
            
            # Calculate indices
            start_idx = int(start_sec * fs)
            end_idx = start_idx + int(duration_sec * fs)
            
            sig = reader.readSignal(i)
            
            # Crop
            sig_cropped = sig[start_idx:end_idx]
            data_list.append(sig_cropped)
            
        writer.writeSamples(data_list)
        writer.close()
        return data_list # Return for plotting
        
    # Output paths
    fname_ref = f"{ref_path.stem}_synced.edf"
    fname_shift = f"{shift_path.stem}_synced.edf"
    
    if prefix:
        fname_ref = f"{prefix}-{fname_ref}"
        fname_shift = f"{prefix}-{fname_shift}"
        
    out_name_ref = out_dir / fname_ref
    out_name_shift = out_dir / fname_shift
    
    print(f"[Cutter] Saving to {out_name_ref}...")
    ref_data = process_file(f_ref, out_name_ref, start_ref_sec, common_dur, ref_signal_headers, ref_header)
    
    print(f"[Cutter] Saving to {out_name_shift}...")
    shift_data = process_file(f_shift, out_name_shift, start_shift_sec, common_dur, shift_signal_headers, shift_header)
    
    # Close readers
    f_ref._close()
    f_shift._close()
    
    if save_plot or (plot and show_plot):
        print("[Cutter] Generating complete alignment plot...")
        # Use first channel for plotting
        # Normalize
        EPSILON = 1e-12
        
        # Get cut signals
        sig_ref_cut = ref_data[0]
        sig_shift_cut = shift_data[0]
        
        sig_ref_cut = (sig_ref_cut - np.mean(sig_ref_cut)) / (np.std(sig_ref_cut) + EPSILON)
        sig_shift_cut = (sig_shift_cut - np.mean(sig_shift_cut)) / (np.std(sig_shift_cut) + EPSILON)
        
        fs_ref = ref_signal_headers[0].get('sample_frequency', ref_signal_headers[0].get('sample_rate'))
        fs_shift = shift_signal_headers[0].get('sample_frequency', shift_signal_headers[0].get('sample_rate'))
        
        # If raw signals weren't passed, load them
        if sRef_raw is None or sShift_raw is None:
            def load_raw_channel_0(path):
                with pyedflib.EdfReader(str(path)) as f:
                    sig = f.readSignal(0)
                    sig = (sig - np.mean(sig)) / (np.std(sig) + EPSILON)
                    return sig
                    
            sRef_raw = load_raw_channel_0(ref_path)
            sShift_raw = load_raw_channel_0(shift_path)
        
        save_path = None
        if save_plot:
            fname = f"alignment_complete_{ref_path.stem}_vs_{shift_path.stem}.png"
            if prefix:
                fname = f"{prefix}-{fname}"
            save_path = out_dir / fname
        
        plot_complete_alignment(
            sRef_raw=sRef_raw,
            sShift_raw=sShift_raw,
            sRef_cut=sig_ref_cut,
            sShift_cut=sig_shift_cut,
            fs_ref=fs_ref,
            fs_shift=fs_shift,
            offset_sec=offset,
            name_ref=ref_path.name,
            name_shift=shift_path.name,
            coarse_corr=coarse_corr,
            coarse_lags=coarse_lags,
            save_path=str(save_path) if save_path else None,
            show_plot=show_plot
        )
        
    print("[Cutter] Done.")
    return out_name_ref, out_name_shift

if __name__ == "__main__":
    pass