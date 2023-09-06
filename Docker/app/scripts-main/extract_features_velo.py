"""Script to extract velocity features from DAS data.

Small 2D windows are moved in time and space over the DAS data and velocity
characteristics of the data are extracted with help of the slant stack
transform. The resulting features are stored and can be used for a clustering
analysis.
"""
from scipy.signal import decimate, iirfilter, zpk2sos, sosfilt, hann
import json
import multiprocessing
import numpy as np
import glob
from pydvs import fk_functions
from pydvs.readers import das_reader as reader

# Set multiprocessing method to `fork`, if not default
multiprocessing.set_start_method("fork")

# Load file with all useful channels
with open("channels_to_read_new.json", 'r') as f:
    channels_to_read_new = json.load(f)

#
# Setting parameters
#
nr_files_to_load = 3
nr_files_to_process = nr_files_to_load - 2
dec_factor = 5
fs_orig = 1000
fs = fs_orig // dec_factor
df = fs_orig
npts = int(nr_files_to_load * fs_orig * 30)
import settings
directory = settings.data_directory
# directory = ('/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/scripts-main/DAS_data/')
files_list = sorted(glob.glob(directory + "*.hdf5"))
nr_files = len(files_list)
print(f"{nr_files} .hdf5 files are in list to be processed.")

# Length and overlap for averaging (long) window
len_ave_window = int(15*fs)
noverlap_ave = int(5*fs)
# Start and end sample (to avoid tapered data)
start_sample = int(30*fs)
end_sample = int((nr_files_to_load-1)*30*fs+(len_ave_window-noverlap_ave))


# Number of time windows
n_twin = (((nr_files-nr_files_to_load) // nr_files_to_process)
          * int(nr_files_to_process * 30 * fs / noverlap_ave))


# Define taper
# Note: 'Hann' window function is used. If other window function is preferred,
# it can be implemented by importing the preferred function directly from
# `scipy.signal`
max_perc = 0.05  # 5% at the start and end of trace are tapered
wlen = int(max_perc * npts)
taper_sides = hann(2*wlen+1)
taper = np.hstack((taper_sides[:wlen], np.ones(npts-2*wlen),
                   taper_sides[len(taper_sides)-wlen:]))

# Define bandpass filter parameters
corners = 4
freqmin = 10
freqmax = 90
fe = 0.5 * df
low = freqmin / fe
high = freqmax / fe
z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter',
                    output='zpk')
sos = zpk2sos(z, p, k)

# Design filter for spatial highpass
nrTr = 1000
max_perc = 0.05  # 5% at the start and end of trace are tapered
wlen_x = int(max_perc * nrTr)
taper_sides_x = hann(2*wlen_x+1)
taper_x = np.hstack((taper_sides_x[:wlen_x], np.ones(nrTr-2*wlen_x),
                     taper_sides_x[len(taper_sides_x)-wlen_x:]))

# Define highpass filter parameters
df_x = 1/4
corners_x = 8
flow = 0.002
fe_x = 0.5 * df_x
f = flow / fe_x
zx, px, kx = iirfilter(corners_x, f, btype='highpass', ftype='butter',
                       output='zpk')
sos_x = zpk2sos(zx, px, kx)

#
# Define functions for preprocessing and extracting velocity features
#


def preproc(tr_id, dec_factor=5):
    """Taper, filter and decimate along time-axis."""
    tapered = st_demean[:, tr_id] * taper
    firstpass = sosfilt(sos, tapered)
    filtered = sosfilt(sos, firstpass[::-1])[::-1]
    downsampled = filtered[::dec_factor]
    # filtered_white = single_whiten_taper(downsampled, f_taper=f_taper,
    #                                      f_smooth=10)
    return downsampled


def preproc_x(tr_t_id):
    """Taper, filter and decimate along space-axis."""
    tapered = st_whitened_agc[tr_t_id, :] * taper_x
    firstpass = sosfilt(sos_x, tapered)
    filtered = sosfilt(sos_x, firstpass[::-1])[::-1]
    return filtered


def slant_stack_full(tmatrix, velocities, dx, dt):
    """Perform a slant-stack on the given wavefield data.

    The following function code has been modified from the distpy module.
    https://github.com/Schlumberger/distpy/tree/master/distpy
    Access: April 5, 2021

    Parameters
    ----------
    array : Array1d
        One-dimensional array object.
    velocities : ndarray
        One-dimensional array of trial velocities.
    Returns
    -------
    tuple
        Of the form `(tau, slant_stack)` where `tau` is an ndarray
        of the attempted intercept times and `slant_stack` are the
        slant-stacked waveforms.
    """
    npts = tmatrix.shape[1]
    nchannels = tmatrix.shape[0]
    position = np.linspace(0, (nchannels)*dx, nchannels, endpoint=False)

    diff = position[1:] - position[:-1]
    diff = diff.reshape((len(diff), 1))
    ntaus = npts - int(np.max(position)*np.max(1/velocities)/dt) - 1

    tmatrix = np.concatenate((tmatrix, np.zeros((nchannels, npts-ntaus))),
                             axis=1)
    ntaus = npts

    slant_stack = np.empty((len(velocities), ntaus))
    rows = np.tile(np.arange(nchannels).reshape(nchannels, 1), (1, ntaus))
    cols = np.tile(np.arange(ntaus).reshape(1, ntaus), (nchannels, 1))

    pre_float_indices = position.reshape(nchannels, 1)/dt
    previous_lower_indices = np.zeros((nchannels, 1), dtype=int)
    for i, velocity in enumerate(velocities):
        float_indices = pre_float_indices/velocity
        lower_indices = np.array(float_indices, dtype=int)
        delta = float_indices - lower_indices
        cols += lower_indices - previous_lower_indices
        amplitudes = tmatrix[rows, cols] * \
            (1-delta) + tmatrix[rows, cols+1]*delta
        integral = 0.5*diff*(amplitudes[1:, :] + amplitudes[:-1, :])
        summation = np.sum(integral, axis=0)
        slant_stack[i, :] = summation

        previous_lower_indices[:] = lower_indices
    taus = np.arange(ntaus)*dt
    return (taus, slant_stack)


def extract_velocities(stCh, pmin=1/10000, pmax=1/1000):
    """Extract apparent velocities using tau-p transform."""
    data_in = noise_filt_agc[:, stCh:stCh+200]
    p = np.linspace(pmin, pmax, 100)
    velocities = 1 / (p + 1e-8)
    taus, tau_pi = slant_stack_full(data_in[:, ::1].T, velocities,
                                    dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
#    f_p_real_power = f_p_real**2
    velo_energy = f_p_real[:, 150:450].sum(axis=1)
    feature1 = decimate(velo_energy, 5)
    velo_energy = f_p_real[:, 450:750].sum(axis=1)
    feature2 = decimate(velo_energy, 5)

    # Flip array to get negative velocities
    taus, tau_pi = slant_stack_full(data_in[:, ::-1].T, velocities,
                                    dx=4, dt=1/fs)
    f_p_real = np.abs(np.fft.rfft(tau_pi, axis=-1))
#    f_p_real_power = f_p_real**2
    velo_energy = f_p_real[:, 150:450].sum(axis=1)
    feature3 = decimate(velo_energy, 5)
    velo_energy = f_p_real[:, 450:750].sum(axis=1)
    feature4 = decimate(velo_energy, 5)
    scaler_low = feature1.sum() + feature3.sum()
    scaler_high = feature2.sum() + feature4.sum()

    return (feature1/scaler_low, feature2/scaler_high,
            feature3/scaler_low, feature4/scaler_high)


#
# Start looping over all files. Only look at upper channel segment (data
# quality is very low on lower channels since there is no snow.)
# Fk-filter implemented
#

if __name__ == '__main__':
    startCh = 1004
    endCh = 2004

    nr_space_loops = len(range(startCh+50, endCh-50-100, 100))
    all_features = np.zeros((nr_space_loops, n_twin, 4, 20))
    for ChRange in range(startCh+50, endCh-50-100, 100):
        print(ChRange, ChRange+200)

    time_counter = 0
    for file_id in range(0, nr_files - nr_files_to_load,
                         nr_files_to_process):
        print(f'File ID: {file_id}')
        st = reader(files_list[file_id:file_id+nr_files_to_load], stream=False,
                    channels=channels_to_read_new[startCh:endCh],
                    h5type='idas2', debug=True)
        npts = st[0].shape[0]
        nrTr = st[0].shape[1]
        print('Format of data: ({}, {})'.format(npts, nrTr))
        print('')

        # Remove time-mean from data
        time_mean = st[0].mean(axis=1).reshape(-1, 1)
        st_demean = st[0] - time_mean

        print("Starting parallelized pre-processing (filtering+whitening)...")
        # Create pool with X processors
        pool0 = multiprocessing.Pool(settings.n_processes)
        results = pool0.map(preproc, range(nrTr))
        pool0.close()
        results_array = np.stack(results).T
        st_whitened = results_array[start_sample:end_sample, :]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_whitened.shape[0], st_whitened.shape[1]))
        print('')
    #    st_whitened = st_whitened - st_whitened.mean(axis=1).reshape(-1,1)
        st_whitened_agc, tosum = fk_functions.AGC(st_whitened, 600)

        print("Starting parallelized pre-processing (high-pass space)...")
        # Create pool with X processors
        pool1 = multiprocessing.Pool(settings.n_processes)
        results1 = pool1.map(preproc_x, range(st_whitened_agc.shape[0]))
        pool1.close()
        results_array3 = np.stack(results1)
        st_clean = results_array3[:, 50:-50]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_clean.shape[0], st_clean.shape[1]))

        print('Starting feature extraction...')
        for time_index in range(0, st_whitened.shape[0]-2*noverlap_ave,
                                len_ave_window-noverlap_ave):
            print("Time index", time_index)
            noise_filt_agc = st_clean[time_index:time_index+len_ave_window, :]

            pool2 = multiprocessing.Pool(settings.n_processes)
            results2 = pool2.map(extract_velocities,
                                 range(0, nrTr-100-100, 100))
            pool2.close()
            all_features[:, time_counter, :, :] = np.stack(results2)

            # Advance `time_counter` globally
            time_counter += 1

    # Save all calculated features to disc
    np.save('../sharedfolder/output/extr_features/velo'+'_temp.npy', all_features)
