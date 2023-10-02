"""Parameters for `stalta_trigger.py`"""
#
# Specifying input and output file
#
data_directory = ("../sharedfolder/input/DAS_data/")

output_file = "../sharedfolder/output/catalogue_raw_temp.csv"
channel_file = "channels_to_read_new.json"

#
# Number of files to load in memory
#
n_files_load = 10

#
# Number of processes - mp.pool
#
import multiprocessing as mp
n_processes = mp.cpu_count()

#
# Filter parameters
#
max_perc = 0.05
corners = 4
df_orig = 1000  # Hz
freqmin = 30  # Hz
freqmax = 100  # Hz
# Decimation factor, check that df_orig/dec_factor > 2*freqmax (Nyquist)
dec_factor = 4
# Number of points per file
npts_single_file = 30000  # 30 s x 1 kHz

#
# STA/LTA trigger parameters
#
sta_len = 0.5  # seconds
lta_len = 3  # seconds
trigger_thresh = 1.5
detrigger_thresh = 1.2
