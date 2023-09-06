"""Script to plot events from a catalogue."""
from pydvs.readers import das_reader as reader
import glob
import pandas as pd
import json
from pydvs import fk_functions
from obspy import UTCDateTime

# data_directory = ('/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/scripts-main/DAS_data/')
import settings
data_directory = settings.data_directory

def query_files_from_starttime_local(index=0, directory=data_directory):
    """Function to read starttime from event catalogue and create a list of
    potential data files to query.
    """
    # Split starttime from event to query raw data
    event_start_exact = df_cat.loc[index, 't_start']
    # `starttime` -1 min
    event_start_previous = str(UTCDateTime(df_cat.loc[index, 't_start']) - 60)
    # `starttime` +1 min
    event_start_next = str(UTCDateTime(df_cat.loc[index, 't_start']) + 60)
    time_split_exact = event_start_exact.split(":")
    time_split_previous = event_start_previous.split(":")
    time_split_next = event_start_next.split(":")
    # Format splitted time for using in regex
    file_regex_exact = f"{time_split_exact[0]}{time_split_exact[1]}".replace(
        "-", "").replace("T", "_")
    file_regex_previous = (
        f"{time_split_previous[0]}{time_split_previous[1]}"
        ).replace("-", "").replace("T", "_")
    file_regex_next = f"{time_split_next[0]}{time_split_next[1]}".replace(
        "-", "").replace("T", "_")

    # Query data from disk
    files_event = sorted(glob.glob(directory + "*" + file_regex_previous
                                   + "*.hdf5"))
    files_event += sorted(glob.glob(directory + "*" + file_regex_exact
                                    + "*.hdf5"))
    files_event += sorted(glob.glob(directory + "*" + file_regex_next
                                    + "*.hdf5"))

    return files_event[1:-1]


def plot_event_from_catalogue_local(index, clipPerc):
    # Create a list of files to scan for specific event
    files_event = query_files_from_starttime_local(index)

    # Set parameters for reading in raw data
    starttime = UTCDateTime(df_cat.loc[index, 't_start']) - 1
    endtime = UTCDateTime(df_cat.loc[index, 't_end']) + 1
    startCh = df_cat.loc[index, 'first_station']

    # Determine first and last channel to plot
    if startCh >= 50:
        startCh = startCh - 50
    else:
        startCh = 0

    endCh = df_cat.loc[index, 'last_station']
    if endCh >= 1890:
        endCh = -1
    else:
        endCh = endCh + 50
    channels = channels_to_read[startCh:endCh]

    # Read in data
    st = reader(files_event, stream=True, channels=channels,
                h5type='idas2', debug=True)

    # Pre-process data
    st.detrend('demean')
    st.taper(0.05, type='cosine')
    st.filter('bandpass', freqmin=30, freqmax=100, corners=4)
    st.decimate(4)

    # Trim data to event time +/- 1 second
    st.trim(starttime=starttime, endtime=endtime)

    # Transform data to 2D numpy array
    st_array = fk_functions.stream2array(st)

    # create/checkifexists output/events_pngs/ folder

    # Plot event
    starttime_file = str(starttime).replace(":", "_")
    outfile = f"../sharedfolder/output/event_pngs/ind{index:04d}_{starttime_file}.png"
    fk_functions.plotChRange(st_array, startCh=startCh, endCh=endCh, fs=250,
                             clipPerc=clipPerc, title=f"{starttime}",
                             outfile=outfile)

#
# Load catalogue and plot all events
#


# Loading (processed) event catalog
cat_path = settings.output_file
df_cat = pd.read_csv((cat_path.split('.csv')[0] + '_processed.csv'),
                     index_col=0)
# df_cat = pd.read_csv(('../output/catalogue_FS_raw_processed.csv'),
#                      index_col=0)
# df_cat = pd.read_csv(('/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/scripts-main/output/catalogue_0720_raw_processed.csv'),
#                      index_col=0)

# Read in list of all useful channels
with open('channels_to_read_new.json', 'r') as f:
    channels_to_read = json.load(f)


# Loop over all events in catalogue and plot
for ind in df_cat.index:
    plot_event_from_catalogue_local(ind, clipPerc=99.97)
