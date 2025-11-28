from syncerpy.syncerpy import syncerpy, print_channels
from cutterpy import cut

FILE_REF = "/home/rainfern/ALL_PROJECTS/2025 - syncing_two_eeg/real_data/1/sub-88_task-Sleep_acq-headband_eeg.edf"
FILE_SHIFT = "/home/rainfern/ALL_PROJECTS/2025 - syncing_two_eeg/real_data/1/sub-88_task-Sleep_acq-psg_eeg_shifted-623321.edf"

# 0. Check Channels
print_channels(FILE_REF, FILE_SHIFT)

# 1. Calculate Offset
offset = syncerpy(
    file_reference=FILE_REF,
    file_shift=FILE_SHIFT,
    channel_reference='HB_1', 
    channel_shift='PSG_F3',    
    plot=True
)

# 3. Align and cut files
new_ref, new_shift = cut(
    file_reference=FILE_REF,
    file_shift=FILE_SHIFT,
    offset=offset,
    output_folder="./aligned_data",
    plot=True
)

print(f"Calculated Offset: {offset} seconds")
print(f"Synced files created:\n  {new_ref}\n  {new_shift}")