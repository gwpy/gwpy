import os, glob, optparse, shutil, warnings, pickle, math, copy, pickle, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal, scipy.stats
from lxml import etree
import gwpy.seismon.seismon_NLNM, gwpy.seismon.seismon_html
import gwpy.seismon.seismon_eqmon, gwpy.seismon.seismon_utils

import gwpy.time, gwpy.timeseries
import gwpy.spectrum, gwpy.spectrogram
import gwpy.plotter

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def hilbert(params, segment):
    """@calculates hilbert transform for given segment.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps

    """

    from obspy.core.util.geodetics import gps2DistAzimuth

    ifo = gwpy.seismon.seismon_utils.getIfo(params)
    ifolat,ifolon = gwpy.seismon.seismon_utils.getLatLon(params)

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

        dataAll.append(dataFull)

    if len(dataAll) == 0:
        print "No data... returning"
        return

    if params["doEarthquakes"]:
        earthquakesDirectory = os.path.join(params["path"],"earthquakes")
        earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
        attributeDics = gwpy.seismon.seismon_utils.read_eqmons(earthquakesXMLFile)

    else:
        attributeDics = []

    for attributeDic in attributeDics:

        if params["ifo"] == "IRIS":
            attributeDic = gwpy.seismon.seismon_eqmon.ifotraveltimes(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        for dataFull in dataAll:
            if "X" in dataFull.channel.name:
                tsx = dataFull.data
            if "Y" in dataFull.channel.name:
                tsy = dataFull.data
            if "Z" in dataFull.channel.name:
                tsz = dataFull.data

        tt = np.array(dataAll[0].times)

        Ptime = max(traveltimes["Ptimes"])
        Stime = max(traveltimes["Stimes"])
        Rtwotime = max(traveltimes["Rtwotimes"])
        RthreePointFivetime = max(traveltimes["RthreePointFivetimes"])
        Rfivetime = max(traveltimes["Rfivetimes"])
        distance = max(traveltimes["Distances"])

        indexes = np.intersect1d(np.where(tt >= Rfivetime)[0],np.where(tt <= Rtwotime)[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)
        tt = tt[indexes]
        tsx = tsx[indexes]
        tsy = tsy[indexes]
        tsz = tsz[indexes]

        tszhilbert = scipy.signal.hilbert(tsz).imag

        N = len(tsz)
        n = np.arange(0,N)
        #print N, len(n)
        coefficients = (1/(2*np.pi*(n-(N-1)/2))) * (1-np.cos(np.pi*(n-(N-1)/2)))
        #coefficients = (2.0/N) * (np.sin(np.pi*n/2.0)**2) * 1.0/np.tan(np.pi*n/N)
        print coefficients

        dataHilbert = tszhilbert.view(tsz.__class__)
        dataHilbert = gwpy.timeseries.TimeSeries(dataHilbert)
        dataHilbert.sample_rate =  dataFull.sample_rate
        dataHilbert.epoch = Rfivetime

        distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
        xazimuth,yazimuth = gwpy.seismon.seismon_utils.getAzimuth(params)

        angle1 = fwd
        rot1 = np.array([[np.cos(angle1), -np.sin(angle1)],[np.sin(angle1),np.cos(angle1)]])
        angle2 = xazimuth * (np.pi/180.0)
        rot2 = np.array([[np.cos(angle2), -np.sin(angle2)],[np.sin(angle2),np.cos(angle2)]])

        angleEQ = np.mod(angle1+angle2,2*np.pi)
        rot = np.array([[np.cos(angleEQ), -np.sin(angleEQ)],[np.sin(angleEQ),np.cos(angleEQ)]])

        twodarray = np.vstack([tsx,tsy])
        z = rot.dot(twodarray)
        tsxy = np.sum(z.T,axis=1)
     
        dataXY = tsxy.view(tsz.__class__)
        dataXY = gwpy.timeseries.TimeSeries(tsxy)
        dataXY.sample_rate = dataFull.sample_rate
        dataXY.epoch = Rfivetime

        dataHilbert = dataHilbert.resample(16)
        dataXY = dataXY.resample(16)

        if params["doPlots"]:

            plotDirectory = params["path"] + "/Hilbert"
            gwpy.seismon.seismon_utils.mkdir(plotDirectory)

            dataHilbert *= 1e6
            dataXY *= 1e6

            pngFile = os.path.join(plotDirectory,"%s.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXY,label="XY",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

        dataHilbert = tszhilbert.view(tsz.__class__)
        dataHilbert = gwpy.timeseries.TimeSeries(dataHilbert)
        dataHilbert.sample_rate =  dataFull.sample_rate
        dataHilbert.epoch = Rfivetime
        dataHilbert = dataHilbert.resample(16)

        angles = np.linspace(0,2*np.pi,10)
        xcorrs = []
        for angle in angles:
            rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

            twodarray = np.vstack([tsx,tsy])
            z = rot.dot(twodarray)
            tsxy = np.sum(z.T,axis=1)

            dataXY = tsxy.view(tsz.__class__)
            dataXY = gwpy.timeseries.TimeSeries(tsxy)
            dataXY.sample_rate = dataFull.sample_rate
            dataXY.epoch = Rfivetime
            dataXY = dataXY.resample(16)

            xcorr,lags = gwpy.seismon.seismon_utils.xcorr(dataHilbert.data,dataXY.data,maxlags=1)
            xcorrs.append(xcorr[1])
        xcorrs = np.array(xcorrs)

        angleMax = angles[np.argmax(xcorrs)]
        rot = np.array([[np.cos(angleMax), -np.sin(angleMax)],[np.sin(angleMax),np.cos(angleMax)]])

        tsxy = []
        for x,y in zip(tsx,tsy):
            vector = np.array([x,y])
            #z = rot1.dot(rot2.dot(vector))
            z = rot.dot(vector)
            tsxy.append(np.sum(z))
        tsxy = np.array(tsxy)

        dataXY = tsxy.view(tsz.__class__)
        dataXY = gwpy.timeseries.TimeSeries(tsxy)
        dataXY.sample_rate = dataFull.sample_rate
        dataXY.epoch = Rfivetime
        dataXY = dataXY.resample(16)

        if params["doPlots"]:

            plotDirectory = params["path"] + "/Hilbert"
            gwpy.seismon.seismon_utils.mkdir(plotDirectory)

            dataHilbert *= 1e6
            dataXY *= 1e6

            pngFile = os.path.join(plotDirectory,"%s-max.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_timeseries(dataHilbert,label="Hilbert",**kwargs)
            kwargs = {"linestyle":"-","color":"g"}
            plot.add_timeseries(dataXY,label="XY",**kwargs)
            plot.ylabel = r"Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

            pngFile = os.path.join(plotDirectory,"%s-rot.png"%(attributeDic["eventName"]))
            plot = gwpy.plotter.Plot(figsize=[14,8])

            kwargs = {"linestyle":"-","color":"b"}
            plot.add_line(angles,xcorrs,label="Xcorrs",**kwargs)
            ylim = [plot.ylim[0],plot.ylim[1]]
            kwargs = {"linestyle":"--","color":"r"}
            plot.add_line([angleEQ,angleEQ],ylim,label="EQ",**kwargs)
            plot.ylabel = r"XCorr"
            plot.xlabel = r"Angle [rad]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()
