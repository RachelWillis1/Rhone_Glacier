import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# filename = 'UTC_20200715_000220.719.h5'
filename = "/run/media/julius/TOSHIBA EXT/DAS_data/UTC_20200720_000138.575.h5"

f = h5.File(filename, 'r')
aux = f['AuxiliaryData']  # 1 member
pro = f['Provenance']  # 0 members
wf = f['Waveforms']  # 2496 members
samplesPerSec = 1000  # ***NEED TO NOT HARD CODE***

channels = list(wf.keys())  # overall list of channels
nChTotal = len(channels)  # number of channels


def readChRange(startCh, endCh):
    nCh = endCh-startCh

    # read first channel
    ch = channels[startCh]
    thisCh = wf.get(ch)
    keys = list(thisCh.keys())
    firstTrace = thisCh[keys[0]]
    nSamples = firstTrace.size

    # create array to store all data and put first channel in there
    data = np.zeros((nCh, nSamples), dtype=firstTrace.dtype)
    data[0, :] = firstTrace

    for idx in range(startCh+1, endCh):
        ch = channels[idx]
        thisCh = wf.get(ch)
        keys = list(thisCh.keys())  # just has 1 key
        # put the data into a numpy array
        data[idx-startCh, :] = thisCh[keys[0]]

    return data


def plotChRange(data, startCh, endCh, samplesPerSec):
    clip = np.percentile(np.absolute(data), 95)
    nSec = float(data.shape[1]) / float(samplesPerSec)
    plt.imshow(data, aspect='auto', interpolation='nearest', vmin=-clip,
               vmax=clip, cmap='seismic', extent=(0, nSec, endCh-1, startCh))
    plt.xlabel('time (s)')
    plt.ylabel('channels')
    plt.colorbar()
    plt.savefig('initial-data-views/ch-'+str(startCh)+'-'+str(endCh)+'.png')
    plt.clf()
    return


channelRanges = [[x, x+300] for x in range(0, 2000, 200)]
for chRange in channelRanges:
    if chRange[1] > nChTotal:  # for last one
        chRange[1] = nChTotal
    print(chRange)

    data = readChRange(chRange[0], chRange[1])
    plotChRange(data, chRange[0], chRange[1], samplesPerSec)
