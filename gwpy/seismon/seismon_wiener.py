#!/usr/bin/python

import sys, os
import numpy as np
import scipy.linalg

import gwpy.seismon.seismon_utils

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

def wiener(params, target_channel, segment):
    """@calculates wiener filter for given channel and segment.

    @param params
        seismon params dictionary
    @param target_channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataAll = []

    samplef = target_channel.samplef
    N = 512
    samplef = 1024

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

        dataFull = dataFull.resample(samplef)

        cutoff = 10.0
        cutoff_low = 300.0
        cutoff_high = 350.0
        #dataFull = dataFull.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')
        #print np.sum(dataFull)
        #dataFull = dataFull.bandpass(cutoff_low,cutoff_high, amplitude=0.9, order=6, method='scipy')
        #print dataFull

        indexes = np.where(np.isnan(dataFull.data))[0]
        meanSamples = np.mean(np.ma.masked_array(dataFull.data,np.isnan(dataFull.data)))
        for index in indexes:
            dataFull[index] = meanSamples
        dataFull -= np.mean(dataFull.data)

        dataAll.append(dataFull)

    X = []
    y = []
    for dataFull in dataAll:
        if dataFull.channel.name == target_channel.station:
            tt = np.array(dataFull.times)
            y = dataFull.data
        else:
            if X == []:
                X = dataFull.data
            else:
                try:
                    X = np.vstack([X,dataFull.data])
                except:
                    continue

    if len(y) == 0:
        print "No data for target channel... continuing"
        return

    originalASD = []
    residualASD = []
    FFASD = []

    gpss = np.arange(gpsStart,gpsEnd,params["fftDuration"])
    create_filter = True
    for i in xrange(len(gpss)-1):
        tt = np.array(dataFull.times)
        indexes = np.intersect1d(np.where(tt >= gpss[i])[0],np.where(tt <= gpss[i+1]+5)[0])

        if len(indexes) == 0:
            continue

        indexMin = np.min(indexes)
        indexMax = np.max(indexes)

        ttCut = tt[indexMin:indexMax] 
        yCut = y[indexMin:indexMax]
        XCut = X[:,indexMin:indexMax]

        XCut = XCut.T
        if create_filter:
            W,R,P = miso_firwiener(N,XCut,yCut)
            create_filter = False
            continue
            
        residual, FF = subtractFF(W,XCut,yCut,samplef)

        gpsStart = tt[indexMin]
        dataOriginal = gwpy.timeseries.TimeSeries(yCut, epoch=gpsStart, sample_rate=samplef,name="Original")
        dataResidual = gwpy.timeseries.TimeSeries(residual, epoch=gpsStart, sample_rate=samplef,name="Residual")
        dataFF = gwpy.timeseries.TimeSeries(FF, epoch=gpsStart, sample_rate=samplef,name="FF")

        #residual, FF = subtractFF(W,XCut,yCut,target_channel.samplef)

        #gpsStart = tt[indexMin]
        #dataOriginal = gwpy.timeseries.TimeSeries(yCut, epoch=gpsStart, sample_rate=target_channel.samplef)
        #dataResidual = gwpy.timeseries.TimeSeries(residual, epoch=gpsStart, sample_rate=target_channel.samplef)
        #dataFF = gwpy.timeseries.TimeSeries(FF, epoch=gpsStart, sample_rate=target_channel.samplef)

        #cutoff = 1.0
        #dataOriginal = dataOriginal.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')
        #dataResidual = dataResidual.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')
        #dataFF = dataFF.lowpass(cutoff, amplitude=0.9, order=3, method='scipy')

        # calculate spectrum
        NFFT = params["fftDuration"]
        #window = None
        dataOriginalASD = dataOriginal.asd(NFFT,NFFT,'welch')
        dataResidualASD = dataResidual.asd(NFFT,NFFT,'welch')
        dataFFASD = dataFF.asd(NFFT,NFFT,'welch')

        freq = np.array(dataOriginalASD.frequencies)
        indexes = np.where((freq >= params["fmin"]) & (freq <= params["fmax"]))[0]
        freq = freq[indexes]

        dataOriginalASD = np.array(dataOriginalASD.data)
        dataOriginalASD = dataOriginalASD[indexes]
        dataOriginalASD = gwpy.spectrum.Spectrum(dataOriginalASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataResidualASD = np.array(dataResidualASD.data)
        dataResidualASD = dataResidualASD[indexes]
        dataResidualASD = gwpy.spectrum.Spectrum(dataResidualASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        dataFFASD = np.array(dataFFASD.data)
        dataFFASD = dataFFASD[indexes]
        dataFFASD = gwpy.spectrum.Spectrum(dataFFASD, f0=np.min(freq), df=(freq[1]-freq[0]))

        originalASD.append(dataOriginalASD)
        residualASD.append(dataResidualASD)
        FFASD.append(dataFFASD)

    dt = gpss[1] - gpss[0]
    epoch = gwpy.time.Time(gpss[0], format='gps')
    originalSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*originalASD, dt=dt,epoch=epoch)
    residualSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*residualASD, dt=dt,epoch=epoch)
    FFSpecgram = gwpy.spectrogram.Spectrogram.from_spectra(*FFASD, dt=dt,epoch=epoch)

    freq = np.array(originalSpecgram.frequencies)
    bins,originalSpecvar = gwpy.seismon.seismon_utils.spectral_histogram(originalSpecgram)
    original_spectral_variation_50per = gwpy.seismon.seismon_utils.spectral_percentiles(originalSpecvar,bins,50)
    bins,residualSpecvar = gwpy.seismon.seismon_utils.spectral_histogram(residualSpecgram)
    residual_spectral_variation_50per = gwpy.seismon.seismon_utils.spectral_percentiles(residualSpecvar,bins,50)
    bins,FFSpecvar = gwpy.seismon.seismon_utils.spectral_histogram(FFSpecgram)
    FF_spectral_variation_50per = gwpy.seismon.seismon_utils.spectral_percentiles(FFSpecvar,bins,50)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/Wiener/" + target_channel.station_underscore
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        fl, low, fh, high = gwpy.seismon.seismon_NLNM.NLNM(2)

        pngFile = os.path.join(plotDirectory,"psd.png")

        plot = gwpy.plotter.Plot(figsize=[14,8])
        kwargs = {"linestyle":"-","color":"b","label":"Original"}
        plot.add_line(freq, original_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-","color":"g","label":"Residual"}
        plot.add_line(freq, residual_spectral_variation_50per, **kwargs)
        kwargs = {"linestyle":"-.","color":"k"}
        plot.add_line(fl, low, **kwargs)
        plot.add_line(fh, high, **kwargs)
        plot.axes[0].set_yscale("log")
        plot.axes[0].set_xscale("log")
        plot.xlim = [params["fmin"],params["fmax"]]
        #plot.ylim = [np.min(bins), np.max(bins)]
        plot.xlabel = "Frequency [Hz]"
        plot.ylabel = "Amplitude Spectrum [(m/s)/rtHz]"
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

def miso_firwiener(N,X,y):

    # MISO_FIRWIENER Optimal FIR Wiener filter for multiple inputs.
    # MISO_FIRWIENER(N,X,Y) computes the optimal FIR Wiener filter of order
    # N, given any number of (stationary) random input signals as the columns
    # of matrix X, and one output signal in column vector Y.
    # Author: Keenan Pepper
    # Last modified: 2007/08/02
    # References:
    # [1] Y. Huang, J. Benesty, and J. Chen, Acoustic MIMO Signal
    # Processing, SpringerVerlag, 2006, page 48

    # Number of input channels.
    try:
        junk, M = X.shape
    except:
        M = 1

    # Input covariance matrix.
    R = np.zeros([M*(N+1),M*(N+1)])
    for m in xrange(M):
        for i in xrange(m,M):
            rmi,lags = gwpy.seismon.seismon_utils.xcorr(X[:,m]-np.mean(X[:,m]),X[:,i]-np.mean(X[:,i]),maxlags=N,normed=False)
            Rmi = scipy.linalg.toeplitz(np.flipud(rmi[range(N+1)]),r=rmi[range(N,2*N+1)])
            top = m*(N+1)
            bottom = (m+1)*(N+1)
            left = i*(N+1)
            right = (i+1)*(N+1)
            #R[range(top,bottom),range(left,right)] = Rmi

            for j in xrange(top,bottom):
                for k in xrange(left,right):
                    R[j,k] = Rmi[j-top,k-left]
         
            if not i == m:
                #R[range(left,right),range(top,bottom)] = Rmi  # Take advantage of hermiticity.

                RmiT = Rmi.T
                for j in xrange(left,right):
                    for k in xrange(top,bottom):
                        R[j,k] = RmiT[j-left,k-top]

    # Crosscorrelation vector.
    P = np.zeros([M*(N+1),])
    for i in xrange(M):
        top = i*(N+1)
        bottom = (i+1)*(N+1)
        p, lags = gwpy.seismon.seismon_utils.xcorr(y-np.mean(y),X[:,i]-np.mean(X[:,i]),maxlags=N,normed=False)

        P[range(top,bottom)] = p[range(N,2*N+1)]

    # The following step is very inefficient because it fails to exploit the
    # block Toeplitz structure of R. Its done the same way in the builtin
    # function "firwiener".
    # P / R

    Z = np.linalg.lstsq(R.T, P.T)[0].T
    W = Z.reshape(M,N+1).T

    return W,R,P

def subtractFF(W,SS,S,samplef):

    # Subtracts the filtered SS from S using FIR filter coefficients W.
    # Routine written by Jan Harms. Routine modified by Michael Coughlin.
    # Modified: August 17, 2012
    # Contact: michael.coughlin@ligo.org

    N = len(W)-1
    ns = len(S)

    FF = np.zeros([ns-N,])

    for k in xrange(N,ns):
        tmp = SS[k-N:k+1,:] * W
        FF[k-N] = np.sum(tmp)

    cutoff = 10.0
    dataFF = gwpy.timeseries.TimeSeries(FF, sample_rate=samplef)
    dataFFLowpass = dataFF.lowpass(cutoff, amplitude=0.9, order=12, method='scipy')
    FF = np.array(dataFFLowpass)
    FF = np.array(dataFF)

    residual = S[range(ns-N)]-FF
    residual = residual - np.mean(residual)

    return residual, FF
