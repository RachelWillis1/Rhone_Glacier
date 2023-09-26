"""Script to trigger events on DAS data.
Channels are preprocessed in a parallel fashion and then an STA/LTA is computed
for each channel individually. The characteristic functions are linearly
stacked over a user-specified range and a simple trigger is run on the stacked
characteristic function.
Returns an raw (!) event catalogue. Needs to be post-processed in order to
merch the same event triggered on different cable segments. This can be
achieved with the script `merge_events.py`.
All parameters can be set in the config file `stalta_params.py`.
"""
import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
import json
from scipy.signal import iirfilter, zpk2sos, sosfilt, hann
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import settings
# from pydvs.readers import das_reader as reader
from readers import das_reader as reader

sys.stdout = sys.__stdout__

# Set multiprocessing method to `fork`, if not default
mp.set_start_method("fork")

# Specify data directory
# directory = "/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/DAS_data/"
directory = settings.data_directory

print('---')
print('reading files from:')
print(directory)

# import os
# cwd = os.getcwd()
# print('I am here:')
# print(cwd)

files_list = sorted(glob.glob(directory + "*.hdf5"))

# print(glob.glob(cwd + "/*.py"))
# print(glob.glob("../"+ "/*.csv"))
# print(glob.glob(directory + "*.hdf5"))

n_files_load = settings.n_files_load

# Read in a list of all useful channels (e.g. loops removed)
with open(settings.channel_file, 'r') as f:
    channels = json.load(f)
npts = int(settings.npts_single_file*n_files_load)
nrTr = len(channels)

# Define taper
max_perc = settings.max_perc
wlen = int(max_perc * npts)
taper_sides = hann(2*wlen+1)
taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),
                   taper_sides[len(taper_sides)-wlen:]))

# Define bandpass filter parameters
corners = settings.corners
df_orig = settings.df_orig
freqmin = settings.freqmin
freqmax = settings.freqmax
fe = 0.5 * df_orig
low = freqmin / fe
high = freqmax / fe
z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter',
                    output='zpk')
sos = zpk2sos(z, p, k)
dec_factor = settings.dec_factor

# Columns for event catalogue
cols = ['t_start', 't_end', 'first_station', 'last_station', 'SNR']


# Define preprocessing function for parallel implementation.
# This function can be modified to include more sophisticated preprocessing
# steps (AGC, whitening, etc.)
def preproc(tr_id, dec_factor=dec_factor):
    """Taper, filter and decimate stream"""
    tapered = st[tr_id].data * taper
    filtered = sosfilt(sos, tapered)
    filtered = filtered[::dec_factor]
    return filtered


def stack_sta_lta_catalogue(st_preproc, st_stats, nr_cfts=100, noverlap=50,
                            startCh=0, endCh=nrTr):
    """
    Linear stacking of reverse STA/LTA for DAS data

    Parameters
    ----------
    st_preproc : list
        List containing preprocessed channels
    st_stats : dict
        Dictionary containing metadata
    nr_cfts : int
        Number of characteristic functions that are stacked
    noverlap : int
        Overlap between averaging window. Half of `nr_cfts` should be sensible
    startCh : int
        First channel considered, in case only part of the channels is
        of interest. Defaults to 0
    endCh: int
        Last channel considered

    Returns
    -------
    events_df : pd.DataFrame
        Raw (!) evenet catalog. Needs to be preprocessed to ensure that the
        same event triggered on different segments is merged.
    """

    events_df = pd.DataFrame(columns=cols)
    df_dec = st_stats.sampling_rate / dec_factor
    npts = len(st_preproc[0])
    starttime = st_stats.starttime

    # Skip the first file as part of its data got affected by the taper
    first_point = npts // n_files_load

    for i in range(startCh, endCh, noverlap):
        # If there are less than nr_cfts channels to stack, continue
        if (endCh - i) < nr_cfts:
            continue

        # Initialize characteristic function and sum over `nr_cfts` channels
        cft = np.zeros(npts)
        for j in range(i, i+nr_cfts):
            data = st_preproc[j]
            cft += recursive_sta_lta(data,
                                     int(settings.sta_len*df_dec),
                                     int(settings.lta_len*df_dec))

        cft /= nr_cfts
        cft_max = cft[first_point:-first_point].max()  # Exclude tapered edges
        # Trigger events based on STA/LTA thresholds
        if cft_max > settings.trigger_thresh:
            print("Event found between channel {} to {}".format(i, i+nr_cfts))
            on_off = trigger_onset(cft[first_point:-first_point],
                                   settings.trigger_thresh,
                                   settings.detrigger_thresh)
            print(on_off)

            # Add events to the catalogue
            for event in range(on_off.shape[0]):
                trigger_on = on_off[event, 0]
                trigger_off = on_off[event, 1]
                t_start = starttime + (first_point+trigger_on)/df_dec
                t_end = starttime + (first_point+trigger_off)/df_dec
                event_SNR = cft[
                    trigger_on+first_point:trigger_off+first_point
                    ].max()
                new_event_df = pd.DataFrame([[
                    str(t_start), t_end, i, i+nr_cfts, event_SNR
                    ]], columns=cols)
                events_df = pd.concat([events_df, new_event_df],
                                      ignore_index=True)
    return events_df


if __name__ == '__main__':
    # Create empty dataframe to append picked event
    catalogue_df = pd.DataFrame(columns=cols)

    # Loop over all files in the directory
    for file_id in range(0, len(files_list), n_files_load-2):
        if file_id + n_files_load < len(files_list):
            print('File ID', file_id)
            st = reader(files_list[file_id:file_id+n_files_load],
                        stream=True, channels=channels, h5type='idas2',
                        debug=True)
        else:
            print('End of files probably...')
            continue

        print('Pre-process data')
        pool = mp.Pool(settings.n_processes)
        st_preproc = pool.map(preproc, range(nrTr))
        pool.close()
        print('Finished pre-processing.')
        print('Start picking from:' + str(st[0].stats.starttime))

        events_df = stack_sta_lta_catalogue(st_preproc, st[0].stats)
        catalogue_df = pd.concat([catalogue_df, events_df])

    catalogue_df.to_csv(settings.output_file)
