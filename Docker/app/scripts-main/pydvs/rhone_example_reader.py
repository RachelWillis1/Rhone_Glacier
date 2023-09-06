# -*- coding: utf-8 -*-
"""
Created on Friday, Nov 13, 2020

@author: Patrick
"""

# import glob
import os
import pathlib
import warnings
import numpy as np
# import datetime

# import h5py
import obspy
import pyasdf

from pyasdf import ASDFDataSet
# from obspy.io.segy.core import _read_segy as readsgy
from obspy import UTCDateTime


def silixa_attributes(ds):
    return ds.auxiliary_data.SystemInformation.sysinfo.parameters


def convert_to_strainrate(st):
    for tr in st:
        tr.data = 116. * tr.data / 8192. * tr.stats.sampling_rate / 10.
    return st


def read_pyasdf(files,
                starttime=UTCDateTime("2010-01-01T00:00:00Z"),
                endtime=UTCDateTime("2200-01-01T00:00:00Z"),
                network="*",
                station="*",
                location="*",
                channel="*",
                tag='raw_data',
                auxiliary=True,
                merge=True,
                dx=8, segy=False,
                merge_method=1, fill_value=0):
    """
    Read in pyasdf .h5 data from a list of files for the given timewindow and
    specified network, station, location and channel codes and returns them as
    an obspy.Stream object. If merge=True, traces with the same name will be
    merged, utilizing obspy.stream.merge with the given merge_method.
    """

    if not isinstance(files, list):
        files = [files]

    st = obspy.Stream()
    it = 0
    files = sorted(files)

    for f in files:
        print("-Reading in:", it + 1, "of", len(files), end="\r")

        f = pathlib.Path(f)

        with pyasdf.ASDFDataSet(f, mode="r") as ds:
            tmp = ds.get_waveforms(network=network, station=station,
                                   location=location, channel=channel,
                                   starttime=starttime, endtime=endtime,
                                   tag=tag)
            if it == 0:
                if auxiliary:
                    st.data_attrib = silixa_attributes(ds)
                    st.resolution = int(st.data_attrib["SpatialResolution[m]"])
                    st.startdistance = float(
                        st.data_attrib["Start Distance (m)"])
                else:
                    pass
                if segy:
                    st.startdistance = -124
                else:
                    pass

        it += 1
        st.__iadd__(tmp.copy())

    if merge:
        try:
            st.merge(method=merge_method, fill_value=fill_value)
        except:
            warnings.warn('Stream could not be merged')
            pass

    if st.get_gaps():
        warnings.warn(
            "Gaps or overlap in the data. Returned stream object is not merged!", UserWarning)

        # if print_gaps:
        #     st.print_gaps()

    return st.sort()


def write_asdf(st, outfile, compression='gzip-3', auxdata=True,
               data_test=True):
    outfile = pathlib.Path(outfile)
    if os.path.exists(outfile):
        warnings.warn(str(outfile) + '\n existsing. Testing File content')
    else:
        with ASDFDataSet(outfile, mode='w', compression=compression) as ds:
            ds.add_waveforms(st, tag="raw_data")
            if auxdata:
                data = np.zeros(2)
                data_type = "SystemInformation"
                path = "sysinfo"
                ds.add_auxiliary_data(data=data, data_type=data_type,
                                      path=path, parameters=st.data_attrib)

        if data_test:
            test_Stream = read_pyasdf([outfile], auxiliary=auxdata)

            if stream_comparison(st, test_Stream):
                print("\U00002705 " + str(outfile),
                      'Stream integrity verified')

            else:
                print("\U0000274C " + str(outfile) + ": Stream integrity could"
                      "not be verified. The input and output Streams are"
                      "different")


def stream_comparison(st0, st1):
    if st0.data_attrib == st1.data_attrib:
        pass
    else:
        print('Data Attributes different!')
        return False

    if len(st0) == len(st1):
        pass
    else:
        print("Lengths not the same")
        return False

    if st0[0].stats.starttime == st1[0].stats.starttime:
        if st0[0].stats.endtime == st1[0].stats.endtime:
            pass
        else:
            print("end times dont align")
            return False
    else:
        print("Start times do not align")
        return False

    for i in range(len(st0)):
        try:
            np.testing.assert_array_equal(st0[i].data, st1[i].data,
                                          "Data content of sorted array is not equal")
        except AssertionError as err:
            print(err)
            return False

    return True


if __name__ == "__main__":
    warnings.warn("Please call the functions separately. This file is not"
                  "supposed to be run by itself :) ", UserWarning)
