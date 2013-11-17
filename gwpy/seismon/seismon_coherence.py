#!/usr/bin/python

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

def coherence(params, channel1, channel2, segment):
    """@calculates spectral data for given channel and segment.

    @param params
        seismon params dictionary
    @param channel1
        seismon channel structure
    @param channel2
        seismon channel structure
    @param segment
        [start,end] gps
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    # make timeseries
    dataFull1 = gwpy.seismon.seismon_utils.retrieve_timeseries(params, channel1, segment)
    if dataFull1 == []:
        return

    dataFull1 = dataFull1 / channel1.calibration
    indexes = np.where(np.isnan(dataFull1.data))[0]
    meanSamples = np.mean(np.ma.masked_array(dataFull1.data,np.isnan(dataFull1.data)))
    for index in indexes:
        dataFull1[index] = meanSamples
    dataFull1 -= np.mean(dataFull1.data)

    if np.mean(dataFull1.data) == 0.0:
        print "data only zeroes... continuing\n"
        return
    if len(dataFull1.data) < 2*channel1.samplef:
        print "timeseries too short for analysis... continuing\n"
        return

    dataFull2 = gwpy.seismon.seismon_utils.retrieve_timeseries(params, channel2, segment)
    if dataFull2 == []:
        return

    dataFull2 = dataFull2 / channel2.calibration
    indexes = np.where(np.isnan(dataFull2.data))[0]
    meanSamples = np.mean(np.ma.masked_array(dataFull2.data,np.isnan(dataFull2.data)))
    for index in indexes:
        dataFull2[index] = meanSamples
    dataFull2 -= np.mean(dataFull2.data)

    if np.mean(dataFull2.data) == 0.0:
        print "data only zeroes... continuing\n"
        return
    if len(dataFull2.data) < 2*channel2.samplef:
        print "timeseries too short for analysis... continuing\n"
        return

    gpss = np.arange(gpsStart,gpsEnd,params["fftDuration"])
    fft1 = []
    fft2 = []
    for i in xrange(len(gpss)-1):

        tt1 = np.array(dataFull1.times)
        indexes = np.intersect1d(np.where(tt1 >= gpss[i])[0],np.where(tt1 <= gpss[i+1])[0])
        indexMin = np.min(indexes)
        indexMax = np.max(indexes)
        dataCut1 = dataFull1[indexMin:indexMax]

        dataFFT1 = dataCut1.fft()
        freqFFT1 = np.array(dataFFT1.frequencies)
        dataFFT1 = np.array(dataFFT1.data)
        indexes = np.where((freqFFT1 >= params["fmin"]) & (freqFFT1 <= params["fmax"]))[0]
        freqFFT1 = freqFFT1[indexes]
        dataFFT1 = dataFFT1[indexes]
        dataFFT1 = gwpy.spectrum.Spectrum(dataFFT1, f0=np.min(freqFFT1), df=(freqFFT1[1]-freqFFT1[0]))

        tt2 = np.array(dataFull2.times)
        indexes = np.intersect1d(np.where(tt2 >= gpss[i])[0],np.where(tt2 <= gpss[i+1])[0])
        indexMin = np.min(indexes)
        indexMax = np.max(indexes)
        dataCut2 = dataFull2[indexMin:indexMax]

        dataFFT2 = dataCut2.fft()
        freqFFT2 = np.array(dataFFT2.frequencies)
        dataFFT2 = np.array(dataFFT2.data)
        indexes = np.where((freqFFT2 >= params["fmin"]) & (freqFFT2 <= params["fmax"]))[0]
        freqFFT2 = freqFFT2[indexes]
        dataFFT2 = dataFFT2[indexes]
        dataFFT2 = gwpy.spectrum.Spectrum(dataFFT2, f0=np.min(freqFFT2), df=(freqFFT2[2]-freqFFT2[0]))

        fft1.append(dataFFT1)
        fft2.append(dataFFT2)

    specgram1 = gwpy.spectrogram.Spectrogram.from_spectra(*fft1, dt=params["fftDuration"],epoch=gpsStart)
    specgram2 = gwpy.spectrogram.Spectrogram.from_spectra(*fft2, dt=params["fftDuration"],epoch=gpsStart)

    freq = np.array(specgram1.frequencies)
    coherence = []
    for i in xrange(len(freq)):
        a1 = specgram1.data[:,i]
        psd1 = np.mean(a1 * np.conjugate(a1)).real
        a2 = specgram2.data[:,i]
        psd2 = np.mean(a2 * np.conjugate(a2)).real
        csd12 = np.mean(a1 * np.conjugate(a2))
        coh = np.absolute(csd12) / np.sqrt(psd1 * psd2)
        coherence.append(coh)
    coherence = np.array(coherence)

    coherenceDirectory = params["dirPath"] + "/Text_Files/Coherence/" + channel1.station_underscore + "_" + channel2.station_underscore + "/" + str(params["fftDuration"])
    gwpy.seismon.seismon_utils.mkdir(coherenceDirectory)

    psdFile = os.path.join(coherenceDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))
    f = open(psdFile,"wb")
    for i in xrange(len(freq)):
        f.write("%e %e\n"%(freq[i],coherence[i]))
    f.close()

    if params["doPlots"]:

        plotDirectory = params["path"] + "/" + channel1.station_underscore + "_" + channel2.station_underscore
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"coh.png")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        kwargs = {"linestyle":"-","color":"b"}
        plot.add_line(freq,coherence,**kwargs)
        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [0,1]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Coherence"
        plot.axes[0].set_xscale("log")

        plot.save(pngFile,dpi=200)
        plot.close()

def coherence_summary(params, channel1, segment):
    """@summary of channels of spectral data.

    @param params
        seismon params dictionary
    @param channel1
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    data = {}
    for channel2 in params["channels"]:

        coherenceDirectory = params["dirPath"] + "/Text_Files/Coherence/" + channel1.station_underscore + "_" + channel2.station_underscore + "/" + str(params["fftDuration"])
        file = os.path.join(coherenceDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))

        if not os.path.isfile(file):
            continue

        spectra_out = gwpy.spectrum.Spectrum.read(file)
        spectra_out.unit = 'counts/Hz^(1/2)'

        if np.sum(spectra_out.data) == 0.0:
            continue

        data[channel2.station_underscore] = {}
        data[channel2.station_underscore]["data"] = spectra_out

    if data == {}:
        return

    if params["doPlots"]:

        plotDirectory = params["path"] + "/coherence_summary"
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"%s.png"%(channel1.station_underscore))
        lowBin = np.inf
        highBin = -np.inf
        plot = gwpy.plotter.Plot(figsize=[14,8])
        for key in data.iterkeys():

            label = key.replace("_","\_")

            plot.add_spectrum(data[key]["data"], label=label)
            lowBin = np.min([lowBin,np.min(data[key]["data"])])
            highBin = np.max([highBin,np.max(data[key]["data"])])

        plot.xlim = [params["fmin"],params["fmax"]]
        plot.ylim = [lowBin, highBin]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Coherence"
        plot.add_legend(loc=1,prop={'size':10})
        plot.axes.set_xscale("log")
        plot.axes.set_yscale("log")

        plot.save(pngFile,dpi=200)
        plot.close()

