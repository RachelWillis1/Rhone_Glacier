"""Script to process raw event catalogue."""
import pandas as pd
from obspy import UTCDateTime

# Define path of raw catalogue
# cat_path = ('/Users/stanekfr/Documents/Work/MINES/DOE_EileenJin/FS_work/Rhonegletscher_FS/scripts-main/output/catalogue_0720_raw.csv')

import settings
cat_path = settings.output_file

# Read catalogue
df_cat = pd.read_csv(cat_path)
# First column is local index
df_cat.rename(columns={'Unnamed: 0': 'local_ind'}, inplace=True)

#
# Loop over all entries in the catalogue
#
drop_me = []  # index of entries to be dropped after merge
for ind in df_cat.index[:-1]:
    # current and current + k element are checked for overlap.
    # If local_ind=0 or end of catalogue are reached, loop ends
    k = 1

    while ind+k < len(df_cat):
        if df_cat.loc[ind+k, 'local_ind']:
            starttime_current = UTCDateTime(df_cat.loc[ind, 't_start'])
            endtime_current = UTCDateTime(df_cat.loc[ind, 't_end'])
            starttime_next = UTCDateTime(df_cat.loc[ind+k, 't_start'])
            endtime_next = UTCDateTime(df_cat.loc[ind+k, 't_end'])

            # Time condition for merge: Starttimes are close OR starttime_next
            # between `starttime_current` and `endtime_current`
            time_overlap = ((abs(starttime_current - starttime_next) < 0.2)
                            or (starttime_next > (starttime_current-0.1)
                                and starttime_next < endtime_current))
            channel_overlap = (abs(df_cat.loc[ind+k, 'first_station']
                                   - df_cat.loc[ind, 'last_station']) <= 50
                               or abs(
                                   df_cat.loc[ind+k, 'last_station']
                                   - df_cat.loc[ind, 'first_station']
                                   )
                               ) <= 50

            # Update entry ind+k and add ind to drop_me
            if time_overlap and channel_overlap:
                df_cat.loc[ind+k, 't_start'] = min(
                    starttime_current, starttime_next)
                df_cat.loc[ind+k, 't_end'] = max(endtime_current, endtime_next)
                df_cat.loc[ind+k, 'first_station'] = min(
                    df_cat.loc[ind, 'first_station'],
                    df_cat.loc[ind+k, 'first_station']
                    )
                df_cat.loc[ind+k, 'last_station'] = max(
                    df_cat.loc[ind, 'last_station'],
                    df_cat.loc[ind+k, 'last_station']
                    )
                df_cat.loc[ind+k, 'SNR'] = max(df_cat.loc[ind, 'SNR'],
                                               df_cat.loc[ind+k, 'SNR'])
                drop_me.append(ind)
                break  # Break if event has been linked to another
        else:
            break  # Leave while loop if local_ind is 0
        k += 1  # Look at next element until condition in while/if are not met


# Drop redundant rows from the catalogue
df_merged = df_cat.drop(labels=drop_me)
# Add attribute `channel_length` to give the spatial extent of the event
df_merged['channel_length'] = (df_merged['last_station']
                               - df_merged['first_station'])
df_merged.sort_values('channel_length')

# Drop local index and sort by `t_start`
unnamed_columns = df_merged.columns[
    df_merged.columns.str.contains('local', case=False)]
df_merged.drop(unnamed_columns, axis=1, inplace=True)
df_merged.sort_values('t_start', inplace=True)
df_merged = df_merged.reset_index(drop=True)

# Save processed catalogue
df_merged.to_csv(cat_path.split('.csv')[0] + '_processed.csv')
