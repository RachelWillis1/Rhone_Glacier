# Importing necessary modules and functions
from scipy.signal import iirfilter, zpk2sos, sosfilt, hann, sosfreqz
import json
import multiprocessing
import numpy as np
import glob
from pydvs import coherency_functions
from pydvs import fk_functions
from pydvs.readers import das_reader as reader

# Set multiprocessing method to `fork`, if not default
multiprocessing.set_start_method("fork")

# Load file which contains all channels with good data quality
with open("channels_to_read_new.json", 'r') as f:
    channels_to_read_new = json.load(f)

# Specify data location and reading parameters
import settings
directory = settings.data_directory
# directory = ('/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/scripts-main/DAS_data/')
# directory = settings.data_directory
nr_files_to_load = 3
nr_files_to_process = nr_files_to_load - 2
dec_factor = 5
fs_orig = 1000
fs = fs_orig // dec_factor
npts = int(nr_files_to_load * fs_orig * 30)
files_list = sorted(glob.glob(directory+'*.hdf5*'))
nr_files = len(files_list)
print(f"{nr_files} .hdf5 files are in list to be processed.")

# Length and overlap for averaging (long) window
len_ave_window = int(15*fs)
noverlap_ave = int(5*fs)
# Length and overlap for sub (short) window
len_sub_window = int(0.6*fs)
noverlap_sub = len_sub_window // 2
# Start and end sample (to avoid tapered data)
start_sample = int(30*fs)
end_sample = int((nr_files_to_load-1)*30*fs+(len_ave_window-noverlap_ave))
# print(start_sample)
# print(end_sample)
# number of short windows inside the long window
nr_sub_windows = len_ave_window // noverlap_sub - 1

# number of frequencies and covariance matrices for 200 channels, when taking
# every 4th channel only (50 channels in total)
channels_example = list(range(0, 200, 4))
nstations = len(channels_example)
nfreqs = len_sub_window // 2 + 1
# n_cov_matrices = int(np.floor(((nr_files_to_process*30*fs
#                                 + noverlap_ave-len_ave_window)
#                                / (len_ave_window - noverlap_ave))+1)
#                      * ((nr_files-nr_files_to_process)
#                         / nr_files_to_process))
n_cov_matrices = (((nr_files-nr_files_to_load) // nr_files_to_process)
                  * int(nr_files_to_process * 30 * fs / noverlap_ave))
print(f"Number of covariance matrices that will be computed: {n_cov_matrices}")
# n_cov_matrices = 6 * 60 * 24

# Compute taper to be used in frequency domain
f_axis_s = np.fft.rfftfreq(len_sub_window, d=1/fs)
f_axis_l = np.fft.rfftfreq(int(nr_files_to_load*30*fs), d=1/fs)

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
freqmin = 5
freqmax = 90
fe = 0.5 * fs_orig
low = freqmin / fe
high = freqmax / fe
z, p, k = iirfilter(corners, [low, high],
                    btype='band', ftype='butter', output='zpk')
sos = zpk2sos(z, p, k)

fe2 = 0.5 * fs
low2 = freqmin / fe2
high2 = freqmax / fe2
z2, p2, k2 = iirfilter(corners, [low2, high2],
                       btype='band', ftype='butter', output='zpk')
sos2 = zpk2sos(z2, p2, k2)
w, h = sosfreqz(sos2, len(f_axis_l))
f_taper = np.abs(h)  # Filter in frequency domain

# Design filter for spatial high-pass
nrTr = 1000
max_perc = 0.05  # 5% at the start and end of trace are tapered
wlen_t = int(max_perc * nrTr)
taper_sides_t = hann(2*wlen_t+1)
taper_x = np.hstack((taper_sides_t[:wlen_t], np.ones(
    nrTr-2*wlen_t), taper_sides_t[len(taper_sides_t)-wlen_t:]))

dfx = 1/4  # Spatial sampling rate (channels are every 4m)
corners = 8
flow = 0.002
fex = 0.5 * dfx
f = flow / fex
zx, px, kx = iirfilter(corners, f, btype='highpass',
                       ftype='butter', output='zpk')
sosx = zpk2sos(zx, px, kx)


# Define function to filter a single trace
def preproc(tr_id, dec_factor=5):
    """Taper, filter and decimate"""
    tapered = st_demean[:, tr_id] * taper
    firstpass = sosfilt(sos, tapered)
    filtered = sosfilt(sos, firstpass[::-1])[::-1]
    downsampled = filtered[::dec_factor]
    filtered_white = coherency_functions.single_whiten_taper(
        downsampled, f_taper=f_taper, f_smooth=10
        )
    return filtered_white


def preproc_x(tr_t_id):
    """Taper, filter and decimate"""
    tapered = st_whitened_agc[tr_t_id, :] * taper_x
    firstpass = sosfilt(sosx, tapered)
    filtered = sosfilt(sosx, firstpass[::-1])[::-1]
    return filtered


def compute_features_averaging_window_local(startChLoc, t_window_short=120,
                                            fs=200, taper_perc=0.25):
    """
    Function to compute covariance matrix for a single averaging window.

    Parameters
    ----------
    t_window_short : int
        Length of sub-window
    t_window_long : int
        Length of averaging window
    fs : int
        Sampling frequency in Hertz
    taper_perc : float
        Proportion of window to be tapered

    Returns
    -------
    cov_matrix_ave : Covariance matrix for input time window
    """
    channels = list(range(startChLoc, startChLoc+200, 4))
    st_ave_window = noise_filt_agc[:, channels]

    # Compute taper-window for sub-windows
    taper_window = np.ones(t_window_short)
    side_len = int(taper_perc * t_window_short)
    taper_sides = hann(2*side_len)
    taper_window[0:side_len] = taper_sides[0:side_len]
    taper_window[-side_len:] = taper_sides[-side_len:]

    # Pre-allocate covariance matrix for averaging window
    cov_matrix_ave = np.zeros((nfreqs, nstations, nstations),
                              dtype=np.complex128)

    # Pre-allocate complex spectras for all sub windows
    data_vector = np.zeros((nfreqs, nstations), dtype=np.complex128)

    # Loop over each channel and compute STFT
    for subw_ind, subw_start in enumerate(range(0, len_ave_window-noverlap_sub,
                                                noverlap_sub)):
        sub_window_tapered = st_ave_window[
            subw_start:subw_start+len_sub_window
            ] * taper_window.reshape(-1, 1)
        data_vector = np.fft.rfft(sub_window_tapered, axis=0)

        # Compute covariance matrix each time
        for freq in range(nfreqs):
            cov_matrix_sub = np.outer(data_vector[freq, :],
                                      np.conj(data_vector[freq, :]))
            cov_matrix_ave[freq, :, :] += cov_matrix_sub

    cov_matrix_ave /= nr_sub_windows

    # Compute features from eigenvalue distribution
    eigenvalues_fk = np.zeros(nfreqs)
    coherence_fk = np.zeros(nfreqs)
    variance_fk = np.zeros(nfreqs)
    # shannon_fk = np.zeros(nfreqs)

    for m in range(nfreqs):
        w, v = np.linalg.eigh(cov_matrix_ave[m, :, :])
        wlen = len(w)
        w_real = np.abs(w)
        indices = np.flip(np.argsort(w_real))
        w_real = w_real[indices]
        w1 = w_real[0]
        w_sum = sum(w_real)
        mean = w_sum/wlen
        # w_norm = w_real/w_sum

        # Extract features
        eigenvalues_fk[m] = w1
        coherence_fk[m] = w1 / w_sum
        variance_fk[m] = sum((w_real - mean)**2) / wlen
#        shannon_fk[m] = sum(-w_norm * np.log(w_norm))

        features = np.stack([eigenvalues_fk, coherence_fk, variance_fk])

    return features.T


#
# START LOOPING OVER ALL FILES
#
if __name__ == "__main__":
    startCh = 1004  # first channel to read
    endCh = 2004  # last channel to read

    # Pre-allocating space for all output
    nr_space_loops = len(range(startCh+50, endCh-50-100, 100))
    all_features = np.zeros((nr_space_loops, n_cov_matrices, nfreqs, 3))
    # for ChRange in range(startCh+50, endCh-50-100, 100):
    #     print(ChRange, ChRange+200)

    time_counter = 0  # global counter for the current averaging window
    for file_id in range(0, nr_files-nr_files_to_load,
                         nr_files_to_process):
        print(f'File ID: {file_id}')
        st = reader(files_list[file_id:file_id+nr_files_to_load],
                    stream=False, channels=channels_to_read_new[startCh:endCh],
                    h5type='idas2', debug=True)
        npts = st[0].shape[0]
        nrTr = st[0].shape[1]
        print('Format of data: ({}, {})'.format(npts, nrTr))
        print('')

        # Remove time-mean from data
        time_mean = st[0].mean(axis=1).reshape(-1, 1)
        st_demean = st[0] - time_mean

        print("Starting parallelized pre-processing (filtering+whitening)...")
        # Create pool with nr. of processors equal to nr. of cores of instance
        pool0 = multiprocessing.Pool(settings.n_processes)
        results0 = pool0.map(preproc, range(nrTr))
        pool0.close()
        # Format data to desired shape and slice to remove the tapered ends
        results_array = np.stack(results0).T
        st_whitened = results_array[start_sample:end_sample, :]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_whitened.shape[0], st_whitened.shape[1]))
        print('')
        # Remove time-mean againg after filtering and apply AGC
        st_whitened = st_whitened - st_whitened.mean(axis=1).reshape(-1, 1)
        st_whitened_agc, tosum = fk_functions.AGC(st_whitened, 400)

        print("Starting parallelized pre-processing (high-pass space)...")
        # Create pool with nr. of processors equal to nr. of cores of instance
        pool1 = multiprocessing.Pool(settings.n_processes)
        results1 = pool1.map(preproc_x, range(st_whitened_agc.shape[0]))
        pool1.close()
        results_array3 = np.stack(results1)
        # Remove first and last 50 channels because of taper in space, as they
        # were affected by tapering
        st_clean = results_array3[:, 50:-50]
        print('Complete! Quick check... Format of results: ({},{})'.format(
            st_clean.shape[0], st_clean.shape[1]))

        print('Starting feature extraction...')
        for time_index in range(0, st_whitened.shape[0]-2*noverlap_ave,
                                noverlap_ave):
            # print(f"Time index {time_index}")
            noise_filt_agc = st_clean[time_index:time_index+len_ave_window, :]

            # Create pool with 8 processes
            # pool2 = multiprocessing.Pool(nr_space_loops)
            pool2 = multiprocessing.Pool(settings.n_processes)
            results2 = pool2.map(compute_features_averaging_window_local,
                                 range(0, nrTr-100-100, 100))
            pool2.close()
            all_features[:, time_counter, :, :] = np.stack(results2)

            # Advance time_counter globally
            time_counter += 1

    # Save all calculated features to disc
    np.save('../sharedfolder/output/extr_features/covmat'+'_temp.npy', all_features)
