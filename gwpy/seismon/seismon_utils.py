#!/usr/bin/python

import os, sys, code, glob, optparse, shutil, warnings, matplotlib, pickle, math, copy, pickle, time
import numpy as np
import scipy.signal, scipy.stats, scipy.fftpack
from collections import namedtuple
from datetime import datetime
from lxml import etree

import lal.gpstime

import glue.datafind, glue.segments, glue.segmentsUtils, glue.lal
import gwpy.seismon.seismon_NLNM, gwpy.seismon.seismon_html
import gwpy.seismon.seismon_eqmon

import gwpy.time, gwpy.timeseries, gwpy.spectrum, gwpy.plotter
import gwpy.segments

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def mkdir(path):
    """@create path (if it does not already exist).

    @param path
        directory path to create
    """

    pathSplit = path.split("/")
    pathAppend = "/"
    for piece in pathSplit:
        if piece == "":
            continue
        pathAppend = os.path.join(pathAppend,piece)
        if not os.path.isdir(pathAppend):
            os.mkdir(pathAppend)   

def read_frames(start_time,end_time,channel,cache):
    """@read frames and return time series.

    @param start_time
        start gps time
    @param end_time
        end gps time
    @param channel
        seismon channel structure
    @param cache
        seismon cache structure
    """

    time = []
    data = []

    #== loop over frames in cache
    for frame in cache:

        if end_time < frame.segment[0]:
            continue
        if start_time > frame.segment[1]:
            continue

        frame_data,data_start,_,dt,_,_ = Fr.frgetvect1d(frame.path,channel.station)
        frame_length = float(dt)*len(frame_data)
        frame_time = data_start+dt*np.arange(len(frame_data))

        for i in range(len(frame_data)):
            if frame_time[i] <= start_time:  continue
            if frame_time[i] >= end_time:  continue
            time.append(frame_time[i])
            data.append(frame_data[i])
    data = [e/channel.calibration for e in data]

    indexes = np.where(np.isnan(data))[0]
    meanSamples = np.mean(np.ma.masked_array(data,np.isnan(data)))
    for index in indexes:
        data[index] = meanSamples

    return time,data

def read_nds(start_time,end_time,channel,conn):
    """@read nds2 and return time series.

    @param start_time
        start gps time
    @param end_time
        end gps time
    @param channel
        seismon channel structure
    @param conn
        nds2 connection
    """

    try:
        buffers = conn.fetch(start_time, end_time,[channel.station])
    except:
        time = []
        data = []
        return time,data

    data = buffers[0].data
    data_start = buffers[0].gps_seconds + buffers[0].gps_nanoseconds
    dt = 1.0 / buffers[0].channel.sample_rate 
    time = data_start+dt*np.arange(len(data))
    data = [e/channel.calibration for e in data]

    indexes = np.where(np.isnan(data))[0]
    meanSamples = np.mean(np.ma.masked_array(data,np.isnan(data)))
    for index in indexes:
        data[index] = meanSamples

    return time,data

def normalize_timeseries(data):  
    """@normalize timeseries for plotting purposes

    @param data
        timeseries structure
    """

    dataSort = np.sort(data)
    index10 = np.floor(len(data) * 0.1)
    index90 = np.floor(len(data) * 0.9)
    dataMin = dataSort[index10]
    dataMax = dataSort[index90] 

    dataNorm = (1/(dataMax - dataMin)) * (data - dataMin)

    indexes = (np.absolute(dataNorm) >= 2).nonzero()[0]
    dataNorm[indexes] = 0

    dataNorm = dataNorm / 2
 
    return dataNorm

def envelope(data):
    """@calculate data envelope

    @param data
        timeseries structure
    """

    hilb = scipy.fftpack.hilbert(data)
    data = (data ** 2 + hilb ** 2) ** 0.5
    return data

def read_eqmons(file):
    """@read eqmon file, returning earthquakes

    @param file
        eqmon file
    """

    attributeDics = []

    if not os.path.isfile(file):
        print "Missing eqmon file: %s"%file
        return attributeDics

    tree = etree.parse(file)
    baseroot = tree.getroot()       # get the document root
    for root in baseroot.iterchildren():
        attributeDic = {}
        for element in root.iterchildren(): # now iter through it and print the text
            if element.tag == "traveltimes":
                attributeDic[element.tag] = {}
                for subelement in element.iterchildren():
                    attributeDic[element.tag][subelement.tag] = {}
                    for subsubelement in subelement.iterchildren():
                        textlist = subsubelement.text.replace("\n","").split(" ")
                        floatlist = [float(x) for x in textlist]
                        attributeDic[element.tag][subelement.tag][subsubelement.tag] = floatlist
            else:
                try:
                    attributeDic[element.tag] = float(element.text)
                except:
                    attributeDic[element.tag] = element.text

        magThreshold = 0
        if not "Magnitude" in attributeDic or attributeDic["Magnitude"] < magThreshold:
            return attributeDic

        attributeDic["doPlots"] = 0
        for ifoName, traveltimes in attributeDic["traveltimes"].items():
            arrivalMin = min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
            arrivalMax = max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
            attributeDic["traveltimes"][ifoName]["arrivalMin"] = arrivalMin
            attributeDic["traveltimes"][ifoName]["arrivalMax"] = arrivalMax
            #if params["gps"] <= attributeDic["traveltimes"][ifoName]["arrivalMax"]:
            #    attributeDic["doPlots"] = 1

        attributeDics.append(attributeDic)
    return attributeDics

def spectral_histogram(specgram,bins=None,lowBin=None,highBin=None,nbins=None):
    """@calculate spectral histogram from spectrogram

    @param specgram
        spectrogram structure
    @param bins
        spectral bins
    @param lowBin
        low bin
    @param highBin
        high bin
    @param nbins
        number of spectral bins
        
    """

    # Define bins for the spectral variation histogram
    if lowBin == None:
        lowBin = np.log10(np.min(specgram)/2)
    if highBin == None:
        highBin = np.log10(np.max(specgram)*2)
    if nbins == None:
        nbins = 500   
    if bins == None:
        bins = np.logspace(lowBin,highBin,num=nbins)

    # Ensure we work with numpy array data
    data = np.array(specgram)

    spectral_variation_norm = []
    rows, columns = data.shape

    # Loop over frequencies
    for i in xrange(columns):
        # calculate histogram for this frequency bin
        this_spectral_variation, bin_edges = np.histogram(data[:,i],bins)
        this_spectral_variation = np.array(this_spectral_variation)
        # Calculate weights for bins (to normalize)
        weight = (100/float(sum(this_spectral_variation))) + np.zeros(this_spectral_variation.shape)
        # stack output array
        if spectral_variation_norm == []:
            spectral_variation_norm = this_spectral_variation * weight
        else:
            spectral_variation_norm = np.vstack([spectral_variation_norm,this_spectral_variation * weight])
    spectral_variation_norm = np.transpose(spectral_variation_norm)

    return bins,spectral_variation_norm

def spectral_percentiles(specvar,bins,percentile):
    """@calculate spectral percentiles from spectral variation histogram

    @param specvar
        spectral variation histogram
    @param bins
        spectral bins
    @param percentile
        percentile of the bins to compute
    """

    # Ensure we work with numpy array data
    data = np.array(specvar)

    percentiles = []
    rows, columns = specvar.shape

    # Loop over frequencies
    for i in xrange(columns):
        # Calculate cumulative sum for array
        cumsumvals = np.cumsum(data[:,i])

        # Find value nearest requested percentile
        abs_cumsumvals_minus_percentile = abs(cumsumvals - percentile)
        minindex = abs_cumsumvals_minus_percentile.argmin()
        val = bins[minindex]

        percentiles.append(val)

    return percentiles

def html_bgcolor(snr,data):
    """@calculate html color

    @param snr
        snr to compute color for
    @param data
        array to compare snr to
    """

    data = np.append(data,snr)

    # Number of colors in array
    N = 256

    colormap = []
    for i in xrange(N):
        r,g,b,a = matplotlib.pyplot.cm.jet(i)
        r = int(round((r * 255),0))
        g = int(round((g * 255),0))
        b = int(round((b* 255),0))
        colormap.append((r,g,b))

    data = np.sort(data)
    itemIndex = np.where(data==snr)

    # Determine significance of snr (between 0 and 1)
    snrSig = itemIndex[0][0] / float(len(data)+1)

    # Determine color index of this significance
    index = int(np.floor(N * snrSig))

    # Return colors of this index
    thisColor = colormap[index]
    # Return rgb string containing these colors
    bgcolor = "rgb(%d,%d,%d)"%(thisColor[0],thisColor[1],thisColor[2])

    return snrSig, bgcolor

def html_hexcolor(snr,data):
    """@calculate html color

    @param snr
        snr to compute color for
    @param data
        array to compare snr to
    """

    data = np.append(data,snr)

    # Number of colors in array
    N = 256

    colormap = []
    for i in xrange(N):
        r,g,b,a = matplotlib.pyplot.cm.jet(i)
        r = int(round((r * 255),0))
        g = int(round((g * 255),0))
        b = int(round((b* 255),0))
        colormap.append((r,g,b))

    data = np.sort(data)
    itemIndex = np.where(data==snr)

    # Determine significance of snr (between 0 and 1)
    snrSig = itemIndex[0][0] / float(len(data)+1)

    # Determine color index of this significance
    index = int(np.floor(N * snrSig))

    # Return colors of this index
    thisColor = colormap[index]

    r = hex(thisColor[0]).split("x")
    r = r[1].rjust(2,"0")
    g = hex(thisColor[1]).split("x")
    g = g[1].rjust(2,"0")
    b = hex(thisColor[2]).split("x")
    b = b[1].rjust(2,"0")

    # Return rgb string containing these colors
    hexcolor = '#ff%s%s%s'%(r,g,b)

    return snrSig,hexcolor

def segment_struct(params):
    """@create seismon segment structure

    @param params
        seismon params structure
    """

    if params["doSegmentsDatabase"]:
        segmentlist, segmentlistValid = gwpy.segments.DataQualityFlag.query(params["segmentFlag"],
                                               params["gpsStart"],params["gpsEnd"],
                                               url=params["segmentDatabase"])
        
        params["segments"] = segmentlist
    elif params["doSegmentsTextFile"]:
        segmentlist = glue.segments.segmentlist()
        segs = np.loadtxt(params["segmentsTextFile"])
        for seg in segs:
            segmentlist.append(glue.segments.segment(seg[0],seg[1]))
        params["segments"] = segmentlist
        params["gpsStart"] = np.min(params["segments"])
        params["gpsEnd"] = np.max(params["segments"])
    else:
        segmentlist = [glue.segments.segment(params["gpsStart"],params["gpsEnd"])]
        params["segments"] = segmentlist

    return params

def frame_struct(params):
    """@create seismon frame structure

    @param params
        seismon params structure
    """

    gpsStart = np.min(params["gpsStart"])-1000
    gpsEnd = np.max(params["gpsEnd"])

    if params["noFrames"]:
        datacache = []
    elif params["ifo"] == "XG":
        frameDir = "/archive/frames/MBH/"
        frameList = [os.path.join(root, name)
            for root, dirs, files in os.walk(frameDir)
            for name in files]

        datacache = []
        for frame in frameList:
            thisFrame = frame.replace("file://localhost","")
            if thisFrame == "":
                continue

            thisFrameSplit = thisFrame.split(".")
            if thisFrameSplit[-1] == "log":
                continue

            thisFrameSplit = thisFrame.split("-")
            gps = float(thisFrameSplit[-2])
            dur = float(thisFrameSplit[-1].replace(".gwf",""))

            if gps+dur < gpsStart:
                continue
            if gps > gpsEnd:
                continue

            #cacheFile = glue.lal.CacheEntry("%s %s %d %d %s"%("XG","Homestake",gps,dur,frame))
            datacache.append(frame)
        datacache = glue.lal.Cache(map(glue.lal.CacheEntry.from_T050017, datacache))

    elif params["ifo"] == "LUNAR":
        frameDir = "/home/mcoughlin/Lunar/data/"
        frameList = [os.path.join(root, name)
            for root, dirs, files in os.walk(frameDir)
            for name in files]

        datacache = []
        for frame in frameList:
            datacache.append(frame)
    elif params["ifo"] == "IRIS":
        datacache = "IRIS"
    else:
        if params["frameType"] == "nds2":
            #conn = nds2.connection(params["ndsServer"])
            #y = conn.find_channels('*',nds2.channel.CHANNEL_TYPE_RAW,\
            #    nds2.channel.DATA_TYPE_FLOAT32, 128, 16384)

            #params["ndsConnection"] = conn
            pass

        else:
            connection = glue.datafind.GWDataFindHTTPConnection()
            datacache = connection.find_frame_urls(params["ifo"][0], params["frameType"],
                                                   gpsStart, gpsEnd,
                                                   urltype="file",
                                                   on_gaps="warn")
            connection.close()

    params["frame"] = datacache

    return params

def channel_struct(params,channelList):
    """@create seismon channel structure

    @param params
        seismon params structure
    @param channelList
        list of channels desired
    """

    latitude = -1
    longitude = -1
    # Create channel structure
    structproxy_channel = namedtuple( "structproxy_channel", "station station_underscore samplef calibration latitude longitude" )

    channel = []

    with open(channelList,'r') as f:

       for line in f:

           line_without_return = line.split("\n")
           line_split = line_without_return[0].split(" ")

           station = line_split[0]
           station_underscore = station.replace(":","_")

           samplef = float(line_split[1])
           calibration = float(line_split[2])

           latitude,longitude = getLatLon(params)

           if station in ["H1:HPI-BS_STSINF_A_Z_IN1_DQ","H0:PEM-LVEA_SEISZ"]:
               latitude = 46.455166000000
               longitude = -119.40743100000326
           elif station in ["H1:HPI-ETMX_STSINF_A_Z_IN1_DQ","H0:PEM-EX_SEISZ"]:
               latitude = 46.43394482046292
               longitude = -119.44972407175652
           elif station in ["H1:HPI-ETMY_STSINF_A_Z_IN1_DQ","H0:PEM-EY_SEISZ"]:
               latitude = 46.48429090731142
               longitude = -119.43824421717696
           elif all(x in station for x in ["L1","BS"]):
               latitude = 30.562906000000023
               longitude = -90.77422499999618
           elif all(x in station for x in ["L1","ETMX"]):
               latitude = 30.5519808137184
               longitude = -90.8139431511044
           elif all(x in station for x in ["L1","ETMY"]):
               latitude = 30.52854703505914
               longitude = -90.76152205811134

           if not params["channel"] == None:
               if not station in params["channel"]:
                   continue

           if params["ifo"] == "IRIS":
               import obspy.iris
               client = obspy.iris.Client()

               tstart = GPSToUTCDateTime(params["gpsStart"])
               tend = GPSToUTCDateTime(params["gpsEnd"])
               channelSplit = station.split(":")

               calibration = 1
               response = client.station(channelSplit[0], channelSplit[1], channelSplit[2], channelSplit[3],starttime=tstart,endtime=tend,level="resp")
               responseLines = response.split("\n")
               for line in responseLines:
                   index = line.find("SensitivityValue")
                   if not index == -1:
                       lineSplit = line.split("<")
                       lineSplit = lineSplit[1].split(">")
                       calibration = float(lineSplit[1])
                       break
               for line in responseLines:
                   index = line.find("SampleRate")
                   if not index == -1:
                       lineSplit = line.split("<")
                       lineSplit = lineSplit[1].split(">")
                       samplef = float(lineSplit[1])
                       break
               for line in responseLines:
                   index = line.find("Lat")
                   if not index == -1:
                       lineSplit = line.split("<")
                       lineSplit = lineSplit[1].split(">")
                       latitude = float(lineSplit[1])
                       break
               for line in responseLines:
                   index = line.find("Lon")
                   if not index == -1:
                       lineSplit = line.split("<")
                       lineSplit = lineSplit[1].split(">")
                       longitude = float(lineSplit[1])
                       break

           channel.append( structproxy_channel(station,station_underscore,samplef,calibration,latitude,longitude))

    params["channels"] = channel
    if params["referenceChannel"] == None:
        params["referenceChannel"] = params["channels"][0].station

    return params

def readParamsFromFile(file):
    """@read seismon params file

    @param file
        seismon params file
    """

    params = {}
    if os.path.isfile(file):
        with open(file,'r') as f:
            for line in f:
                line_without_return = line.split("\n")
                line_split = line_without_return[0].split(" ")
                params[line_split[0]] = line_split[1]
    return params

def setPath(params,segment):
    """@set seismon params

    @param params
        seismon params structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    if params["doEarthquakesOnline"]:
        # Output path for run
        params["path"] = params["dirPath"] + "/" + params["ifo"] + "/" + params["runName"] + '-Online' + '/' + "%.0f"%gpsStart + "-" + "%.0f"%gpsEnd
    else:
        params["path"] = params["dirPath"] + "/" + params["ifo"] + "/" + params["runName"] + "-" + "%.0f"%gpsStart + "-" + "%.0f"%gpsEnd

    if params["doAnalysis"] or params["doPlots"] or params["doEarthquakesAnalysis"] or params["doEarthquakesOnline"]:
        gwpy.seismon.seismon_utils.mkdir(params["path"])

    return params

def getIfo(params):

    if params["ifo"] == "H1":
        ifo = "LHO"
    elif params["ifo"] == "L1":
        ifo = "LLO"
    elif params["ifo"] == "G1":
        ifo = "GEO"
    elif params["ifo"] == "V1":
        ifo = "VIRGO"
    elif params["ifo"] == "C1":
        ifo = "FortyMeter"
    elif params["ifo"] == "XG":
        ifo = "Homestake"
    elif params["ifo"] == "IRIS":
        ifo = "LHO"
    elif params["ifo"] == "LUNAR":
        ifo = "LHO"

    return ifo

def getLatLon(params):

    if params["ifo"] == "H1":
        latitude = 46.6475
        longitude = -119.5986
    elif params["ifo"] == "L1":
        latitude = 30.4986
        longitude = -90.7483
    elif params["ifo"] == "G1":
        latitude = 52.246944
        longitude = 9.808333
    elif params["ifo"] == "V1":
        latitude = 43.631389
        longitude = 10.505
    elif params["ifo"] == "C1":
        latitude = 34.1391
        longitude = -118.1238
    elif params["ifo"] == "XG":
        latitude = 44.3465
        longitude = -103.7574
    elif params["ifo"] == "IRIS":
        latitude = 46.6475
        longitude = -119.5986
    elif params["ifo"] == "LUNAR":
        latitude = 46.6475
        longitude = -119.5986

    return latitude,longitude

def getAzimuth(params):

    if params["ifo"] == "H1":
        xazimuth = 5.65487724844 * (180.0/np.pi)
        yazimuth = 4.08408092164 * (180.0/np.pi)
    elif params["ifo"] == "L1":
        xazimuth = 4.40317772346 * (180.0/np.pi)
        yazimuth = 2.83238139666 * (180.0/np.pi)
    elif params["ifo"] == "G1":
        xazimuth = 1.19360100484 * (180.0/np.pi)
        yazimuth = 5.83039279401 * (180.0/np.pi)
    elif params["ifo"] == "V1":
        xazimuth = 0.33916285222 * (180.0/np.pi)
        yazimuth = 5.05155183261 * (180.0/np.pi)
    elif params["ifo"] == "C1":
        xazimuth = 3.14159265359 * (180.0/np.pi)
        yazimuth = 1.57079632679 * (180.0/np.pi)
    elif params["ifo"] == "XG":
        xazimuth = 0.0
        yazimuth = 0.0
    elif params["ifo"] == "IRIS":
        xazimuth = 0.0
        yazimuth = 0.0
    elif params["ifo"] == "LUNAR":
        xazimuth = 0.0
        yazimuth = 0.0

    return xazimuth,yazimuth

def GPSToUTCDateTime(gps):
    """@calculate UTC time from gps

    @param gps
        gps time
    """

    import obspy
    utc = lal.gpstime.gps_to_utc(int(gps))
    tt = time.strftime("%Y-%jT%H:%M:%S", utc)
    ttUTC = obspy.UTCDateTime(tt)

    return ttUTC

def UnixToUTCDateTime(gps):
    """@calculate UTC time from unix timestamp

    @param gps
        gps time
    """

    import obspy
    tt = datetime.fromtimestamp(gps)
    ttUTC = obspy.UTCDateTime(tt)

    return ttUTC

def retrieve_timeseries(params,channel,segment):
    """@retrieves timeseries for given channel and segment.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)

    dataFull = []
    if params["ifo"] == "IRIS":
        import obspy.iris
        client = obspy.iris.Client()
        tstart = gwpy.seismon.seismon_utils.GPSToUTCDateTime(gpsStart)
        tend = gwpy.seismon.seismon_utils.GPSToUTCDateTime(gpsEnd)

        channelSplit = channel.station.split(":")
        try:
            st = client.getWaveform(channelSplit[0], channelSplit[1], channelSplit[2], channelSplit[3],\
                tstart, tend)
        except:
            print "data read from IRIS failed... continuing\n"
            return dataFull

        data = np.array(st[0].data)
        data = data.astype(float)

        dataFull = gwpy.timeseries.TimeSeries(data, times=None, epoch=gpsStart, channel=channel.station, unit=None,sample_rate=channel.samplef, name=channel.station)

    elif params["ifo"] == "LUNAR":
        import obspy
        traces = []
        for frame in params["frame"]:
            st = obspy.read(frame)
            for trace in st:
                trace_station = "%s.%s.%s.%s"%(trace.stats["network"],trace.stats["station"],trace.stats["location"],trace.stats["channel"])
                if trace_station == channel.station:
                    traces.append(trace)
        st = obspy.core.stream.Stream(traces=traces)

        starttime = UnixToUTCDateTime(gpsStart)
        endtime = UnixToUTCDateTime(gpsEnd)
        st = st.slice(starttime, endtime)

        data = st[0].data
        data = data.astype(float)
        data = RunningMedian(data,701,2)

        sample_rate = st[0].stats.sampling_rate
        dataFull = gwpy.timeseries.TimeSeries(data, times=None, epoch=gpsStart, channel=channel.station, unit=None,sample_rate=sample_rate, name=channel.station)

        dataFull.resample(channel.samplef)

    else:

        #for frame in params["frame"]:
        #    print frame.path
        #    frame_data,data_start,_,dt,_,_ = Fr.frgetvect1d(frame.path,channel.station)

        #print params["frame"]
        #dataFull = gwpy.timeseries.TimeSeries.read(params["frame"], channel.station, epoch=gpsStart, duration=duration)
        #print "done"

        # make timeseries
        try:
            dataFull = gwpy.timeseries.TimeSeries.read(params["frame"], channel.station, start=gpsStart, end=gpsEnd)
        except:
            print "data read from frames failed... continuing\n"
            return dataFull

    return dataFull

def RunningMedian(data, M, factor):

    m = np.round(M/2)
    for i in xrange(len(data)):
        indexMin = np.max([1,i-m])
        indexMax = np.min([i+m,len(data)])
        data_slice_median = np.median(data[indexMin:indexMax])
        if np.absolute(data_slice_median)*factor < np.absolute(data[i]):
            data[i] = data_slice_median
    return data

def flag_struct(params,segment):
    """@create seismon flag structure

    @param params
        seismon params structure
    @param segment
        [start,end] gps
    """

    if not "flagList" in params:
        params["flagList"] = glue.segments.segmentlist()

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # set the times
    duration = np.ceil(gpsEnd-gpsStart)
    segmentlist = glue.segments.segmentlist()

    if params["doFlagsDatabase"]:
        print "Generating flags from database"
        dqsegments = gwpy.segments.DataQualityFlag.query(params["flagsFlag"],gpsStart,gpsEnd,url=params["flagsDatabase"])
        segmentlist = dqsegments.active
    elif params["doFlagsTextFile"]:
        lines = [line.strip() for line in open(params["flagsTextFile"])]
        for line in lines:
            lineSplit = line.split(",")
            seg = [float(lineSplit[0]),float(lineSplit[1])]
            segmentlist.append(glue.segments.segment(seg[0],seg[1]))

    elif params["doFlagsChannel"]:
        print "Generating flags from timeseries"
        if params["doPlots"]:
            plotDirectory = params["path"] + "/flags" 
            gwpy.seismon.seismon_utils.mkdir(plotDirectory)

            pngFile = os.path.join(plotDirectory,"timeseries.png")
            plot = gwpy.plotter.TimeSeriesPlot(figsize=[14,8])

        lines = [line.strip() for line in open(params["flagsTextFile"])]    
        for line in lines:

            lineSplit = line.split(" ")
            channel = lineSplit[0]
            samplef = int(lineSplit[1])
            threshold = float(lineSplit[2])

            try:
                dataFull = gwpy.timeseries.TimeSeries.read(params["frame"], channel, start=gpsStart, end=gpsEnd)
            except:
                print "data read from frames failed... continuing\n"
                continue

            if params["doPlots"]:
                label = channel.replace(":","_").replace("_","\_")
                plot.add_timeseries(dataFull,label=label)

            dataFullState = dataFull > threshold
            segmentlistflag = dataFullState.to_dqflag(round=True)
            segmentlist = segmentlist | segmentlistflag.active
 
        if params["doPlots"]:
            plot.ylabel = r"RMS Velocity [$\mu$m/s]"
            plot.add_legend(loc=1,prop={'size':10})

            plot.save(pngFile)
            plot.close()

    params["flagList"] = params["flagList"] | segmentlist
    return params

def segmentlist_duration(segmentlist):
    """@determine duration of segmentlist

    @param segmentlist
        glue segment list
    """

    duration = 0
    for seg in segmentlist:
        dur = seg[1] - seg[0]
        duration = duration + dur
    return duration

def run_flags_analysis(params,segment):
    """@run earthquakes prediction.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    noticesDirectory = os.path.join(params["path"],"notices")
    noticesFile = os.path.join(noticesDirectory,"notices.txt")

    segmentlist = glue.segments.segmentlist()
    segs = np.loadtxt(noticesFile)
    try:
        for seg in segs:
            segmentlist.append(glue.segments.segment(seg[0],seg[0]+seg[1]))
    except:
            segmentlist.append(glue.segments.segment(segs[0],segs[0]+segs[1]))
    earthquake_segmentlist = segmentlist
    earthquake_segmentlist.coalesce()
    earthquake_segmentlist_duration = segmentlist_duration(earthquake_segmentlist)
    earthquake_segmentlist_percentage = 100 * earthquake_segmentlist_duration / float(gpsEnd - gpsStart)

    flag_segmentlist = params["flagList"]
    flag_segmentlist_duration = segmentlist_duration(flag_segmentlist)
    flag_segmentlist_percentage = 100 * flag_segmentlist_duration / float(gpsEnd - gpsStart)

    earthquake_minus_flag_segmentlist = earthquake_segmentlist - flag_segmentlist
    earthquake_minus_flag_segmentlist.coalesce()
    earthquake_minus_flag_segmentlist_duration = segmentlist_duration(earthquake_minus_flag_segmentlist)
    earthquake_minus_flag_segmentlist_percentage = \
        100 * earthquake_minus_flag_segmentlist_duration/float(gpsEnd - gpsStart)
    flag_minus_earthquake_segmentlist = flag_segmentlist - earthquake_segmentlist
    flag_minus_earthquake_segmentlist.coalesce()
    flag_minus_earthquake_segmentlist_duration = segmentlist_duration(flag_minus_earthquake_segmentlist)
    flag_minus_earthquake_segmentlist_percentage = \
        100 * flag_minus_earthquake_segmentlist_duration/float(gpsEnd - gpsStart)

    print "Flags statistics"
    print "Earthquakes total time: %d s"%earthquake_segmentlist_duration
    print "Earthquakes percentage time: %.5e"%earthquake_segmentlist_percentage
    print "Flags total time: %d s"%flag_segmentlist_duration
    print "Flags percentage time: %.5e"%flag_segmentlist_percentage
    print "Earthquakes minus flag total time: %d s"%earthquake_minus_flag_segmentlist_duration
    print "Earthquakes minus flag percentage time: %.5e"%earthquake_minus_flag_segmentlist_percentage
    print "Flags minus earthquake total time: %d s"%flag_minus_earthquake_segmentlist_duration
    print "Flags minus earthquake percentage time: %.5e"%flag_minus_earthquake_segmentlist_percentage

def xcorr(x, y, normed=True, maxlags=None):
    """
    Call signature::

    xcorr(x, y, normed=True, detrend=mlab.detrend_none,
    usevlines=True, maxlags=10, **kwargs)

    """

    Nx = len(x)
    if Nx!=len(y):
        raise ValueError('x and y must be equal length')

    c = np.correlate(x, y, mode='full')

    if normed: c/= np.sqrt(np.dot(x,x) * np.dot(y,y))

    if maxlags is None: maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly positive < %d'%Nx)

    lags = np.arange(-maxlags,maxlags+1)
    c = c[Nx-1-maxlags:Nx+maxlags]

    return c,lags

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return
