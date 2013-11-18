
import os, glob, optparse, shutil, warnings, matplotlib, pickle, math, copy, pickle, time
import numpy as np

import gwpy.seismon.seismon_utils

import gwpy.time, gwpy.timeseries, gwpy.spectrum, gwpy.plotter
import gwpy.segments

import obspy.core, obspy.signal.array_analysis
import matplotlib.pyplot as plt
import matplotlib.colorbar, matplotlib.cm, matplotlib.colors

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def beamforming(params, segment):
    """@calculates beam forming data for given segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    st = obspy.core.Stream()

    tstart = gwpy.seismon.seismon_utils.GPSToUTCDateTime(gpsStart)
    tend = gwpy.seismon.seismon_utils.GPSToUTCDateTime(gpsEnd)

    for channel in params["channels"]:
        # make timeseries
        dataFull = gwpy.seismon.seismon_utils.retrieve_timeseries(params, channel, segment)
        if dataFull == []:
            continue

        dataFull = dataFull / channel.calibration
        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        trace = obspy.core.Trace()
        trace.stats.network = ""
        trace.stats.station = channel.station
        trace.stats.location = ""
        trace.stats.channel = ""
        trace.stats.sampling_rate = channel.samplef
        trace.stats.delta = 1/channel.samplef
        trace.stats.calib = 1
        trace.data = np.array(dataFull.data)

        trace.stats.starttime = tstart
        trace.stats.npts = len(dataFull.data)

        trace.stats.coordinates = obspy.core.AttribDict({
            'latitude': channel.latitude,
            'longitude': channel.longitude,
            'elevation': 0.0})

        st += trace

    print st

    # Execute sonic
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
        # sliding window properties
        win_len=1.0, win_frac=0.05,
        # frequency properties
        frqlow=0.0, frqhigh=1.0, prewhiten=0,
        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',
        #stime=tstart, etime=tstart+300,
        stime=tstart, etime=tend-1,
        coordsys = 'lonlat', method = 0
    )

    out = obspy.signal.array_analysis.array_processing(st, **kwargs)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/beamforming"
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"timeseries.png")

        # Plot
        labels = ['rel.power', 'abs.power', 'baz', 'slow']

        fig = plt.figure(figsize=(16, 16))
        for i, lab in enumerate(labels):
            ax = fig.add_subplot(4, 1, i + 1)
            ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                   edgecolors='none')
            ax.set_ylabel(lab)
            ax.set_xlim(out[0, 0], out[-1, 0])
            ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())

        fig.autofmt_xdate()
        fig.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0)
        plt.show()
        plt.savefig(pngFile,dpi=200)
        plt.close('all')

        cmap = matplotlib.cm.hot_r

        # make output human readable, adjust backazimuth to values between 0 and 360
        t, rel_power, abs_power, baz, slow = out.T
        baz[baz < 0.0] += 360

        # choose number of fractions in plot (desirably 360 degree/N is an integer!)
        N = 30
        abins = np.arange(N + 1) * 360. / N
        sbins = np.linspace(0, 3, N + 1)

        # sum rel power in bins given by abins and sbins
        hist, baz_edges, sl_edges = np.histogram2d(baz, slow,
                bins=[abins, sbins], weights=rel_power)

        max_i,max_j = np.unravel_index(hist.argmax(), hist.shape)
        print hist[max_i,max_j]
        print baz_edges[max_i]
        print sl_edges[max_j]

        # transform to gradient
        baz_edges = baz_edges / 180 * np.pi

        pngFile = os.path.join(plotDirectory,"polar.png")

        # add polar and colorbar axes
        fig = plt.figure(figsize=(16, 16))
        cax = fig.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)

        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])

        # circle through backazimuth
        for i, row in enumerate(hist):
            bars = ax.bar(left=(np.pi / 2 - (i + 1) * dw) * np.ones(N),
                          height=dh * np.ones(N),
                          width=dw, bottom=dh * np.arange(N),
                          color=cmap(row / hist.max()))

        ax.set_xticks([np.pi / 2, 0, 3. / 2 * np.pi, np.pi])
        ax.set_xticklabels(['N', 'E', 'S', 'W'])

        # set slowness limits
        ax.set_ylim(0, 3)
        matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                     norm=matplotlib.colors.Normalize(vmin=hist.min(), vmax=hist.max()))

        plt.show()
        plt.savefig(pngFile,dpi=200)
        plt.close('all')

def strainz(params, segment):
    """@calculates strain for given segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataAll = []

    for channel in params["channels"]:
        # make timeseries
        dataFull = gwpy.seismon.seismon_utils.retrieve_timeseries(params, channel, segment)
        if dataFull == []:
            continue

        dataFull = dataFull / channel.calibration
        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        if np.mean(dataFull.data) == 0.0:
            print "data only zeroes... continuing\n"
            continue
        if len(dataFull.data) < 2*channel.samplef:
            print "timeseries too short for analysis... continuing\n"
            continue

        cutoff = 0.3
        dataFull = dataFull.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')
        dataFull = dataFull.resample(16)

        dataAll.append(dataFull)

    ts1 = []
    ts2 = []
    ts3 = []
    for dataFull in dataAll:
        if "SEISX" in dataFull.channel.name:
            if ts1 == []:
                ts1 = dataFull.data
            else:
                ts1 = np.vstack([ts1,dataFull.data])
        if "SEISY" in dataFull.channel.name:
            if ts2 == []:
                ts2 = dataFull.data
            else:
                ts2 = np.vstack([ts2,dataFull.data])
        if "SEISZ" in dataFull.channel.name:
            if ts3 == []:
                ts3 = dataFull.data
            else:
                ts3 = np.vstack([ts3,dataFull.data])

    tt = np.array(dataFull.times)

    ts1 = ts1.T
    ts2 = ts2.T
    ts3 = ts3.T

    Nsamples,Na = ts1.shape
    subarray = np.arange(Na)
    vp = 8.0
    vs = 4.0
    array_coords = np.array([[0,0,0],[4000,0,0],[0,4000,0]])

    sigmau = np.zeros((Na,3))
    for i in xrange(Na):
        sigmau[i,0] = np.std(ts1[i,:60*channel.samplef])
        sigmau[i,1] = np.std(ts2[i,:60*channel.samplef])
        sigmau[i,2] = np.std(ts3[i,:60*channel.samplef])

    out_dic = obspy.signal.array_analysis.array_rotation_strain(subarray, ts1, ts2, ts3, vp, vs, array_coords, sigmau)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/strainz"
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"strainz.png")

        # Plot
        labels = ['Shear strain [1/s]', 'Dilation [1/s]', 'x1 rotation [rad/s]', 
            'x2 rotation [rad/s]','x3 rotation [rad/s]','Misfit']

        fig = plt.figure(figsize=(32, 24))
        for i, lab in enumerate(labels):
            ax = fig.add_subplot(6, 1, i + 1)
            data = gwpy.timeseries.TimeSeries(data, epoch=tt[0], sample_rate=1.0/(tt[1]-tt[0]))
            if i == 0:
                data = out_dic["ts_s"]
                ax.plot(tt,out_dic["ts_s"])
            elif i == 1:
                data = out_dic["ts_d"]
                ax.plot(tt,out_dic["ts_d"])
            elif i == 2:
                data = out_dic["ts_w1"]
                ax.plot(tt,out_dic["ts_w1"])
            elif i == 3:
                data = out_dic["ts_w2"]
                ax.plot(tt,out_dic["ts_w2"])
            elif i == 4:
                data = out_dic["ts_w3"]
                ax.plot(tt,out_dic["ts_w3"])
            elif i == 5:
                data = out_dic["ts_M"]
                ax.plot(tt,out_dic["ts_M"])
            data = gwpy.timeseries.TimeSeries(data, epoch=tt[0], sample_rate=1.0/(tt[1]-tt[0]))
            ax.set_ylabel(lab)
            #ax.set_xlim(out[0, 0], out[-1, 0])
            #ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())

        fig.autofmt_xdate()
        fig.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0)
        plt.xlabel('Time [s]')
        plt.show()
        plt.savefig(pngFile,dpi=200)
        plt.close('all')

