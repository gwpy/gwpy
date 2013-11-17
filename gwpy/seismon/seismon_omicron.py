#!/usr/bin/python

import os, glob, optparse, shutil, warnings
import numpy as np
from subprocess import Popen, PIPE, STDOUT

import gwpy.seismon.seismon_utils

import gwpy.time, gwpy.timeseries
import gwpy.spectrum, gwpy.spectrogram
import gwpy.plotter, gwpy.table

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#               DEFINITIONS
#
# =============================================================================

def plot_triggers(params,channel,segment):
    """plot omicron triggers for given channel and segment.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    columns = ["peak_time","peak_time_ns","start_time","start_time_ns","stop_time","stop_time_ns","duration","central_freq","flow","fhigh","bandwidth","amplitude","snr"]

    trigger_threshold = 5

    omicronDirectory = os.path.join(params["path"],"omicron")
    omicronPath = os.path.join(omicronDirectory,channel.station)
    omicronXMLs = glob.glob(os.path.join(omicronPath,"*.xml"))

    format = "sngl_burst"
    table = []
    for omicronXML in omicronXMLs:
        table = gwpy.table.Table.read(omicronXML,format,columns=columns)
        table.add_column(table.ColumnClass(
            data=table['peak_time']+table['peak_time_ns']*1e-9,
            name='time'))

    if table == []:
       return
      
    textLocation = params["path"] + "/" + channel.station_underscore
    gwpy.seismon.seismon_utils.mkdir(textLocation)

    f = open(os.path.join(textLocation,"triggers.txt"),"w")
    for row in table:
        f.write("%.1f %e %e\n"%(row["peak_time"],row["central_freq"],row["snr"]))
    f.close()

    if params["doPlots"]:

        plotLocation = params["path"] + "/" + channel.station_underscore
        gwpy.seismon.seismon_utils.mkdir(plotLocation)

        if params["doEarthquakesAnalysis"]:
            pngFile = os.path.join(plotLocation,"omicron-%d-%d.png"%(gpsStart,gpsEnd))
        else:
            pngFile = os.path.join(plotLocation,"omicron.png")

        epoch = gwpy.time.Time(gpsStart, format='gps')
       
        #plot = gwpy.plotter.Plot(auto_refresh=True,figsize=[14,8])
        #plot.add_table(table, 'time', 'central_freq', colorcolumn='snr') 
        plot = gwpy.plotter.TablePlot(table, 'time', 'central_freq', colorcolumn='snr', figsize=[12,6])
        plot.add_colorbar(log=False, clim=[6, 20])
        plot.xlim = [gpsStart, gpsEnd]
        plot.xlabel = 'Time'
        plot.ylabel = 'Frequency [Hz]'
        plot.axes.set_yscale("log")
        plot.colorlabel = r'Signal-to-noise ratio (SNR)'
        plot.save(pngFile)
        plot.close()

def generate_triggers(params):
    """@generate omicron triggers.

    @param params
        seismon params dictionary
    """

    omicronDirectory = os.path.join(params["path"],"omicron")
    gwpy.seismon.seismon_utils.mkdir(omicronDirectory)

    gpsStart = 1e20
    gpsEnd = -1e20
    f = open(os.path.join(omicronDirectory,"frames.ffl"),"w")
    for frame in params["frame"]:
        f.write("%s %d %d 0 0\n"%(frame.path, frame.segment[0], frame.segment[1]-frame.segment[0]))
        gpsStart = min(gpsStart,frame.segment[0])
        gpsEnd = max(gpsEnd,frame.segment[1])
    f.close()

    paramsFile = omicron_params(params)
    f = open(os.path.join(omicronDirectory,"params.txt"),"w")
    f.write("%s"%(paramsFile))
    f.close()

    f = open(os.path.join(omicronDirectory,"segments.txt"),"w")
    f.write("%d %d\n"%(gpsStart,gpsEnd))
    f.close()

    omicron = "/home/detchar/opt/virgosoft/Omicron/v0r3/Linux-x86_64/omicron.exe"
    environmentSetup = "CMTPATH=/home/detchar/opt/virgosoft; export CMTPATH; source /home/detchar/opt/virgosoft/Omicron/v0r3/cmt/setup.sh"
    omicronCommand = "%s; %s %s %s"%(environmentSetup, omicron, os.path.join(omicronDirectory,"segments.txt"),os.path.join(omicronDirectory,"params.txt"))

    p = Popen(omicronCommand,shell=True,stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()

def omicron_params(params):
    """@generate omicron params file.

    @param params
        seismon params dictionary
    """

    omicronDirectory = os.path.join(params["path"],"omicron")

    channelList = ""
    samplerateList = ""
    for channel in params["channels"]:
        channelList = "%s %s"%(channelList,channel.station)
        samplerateList = "%s %d"%(samplerateList,channel.samplef)

    paramsFile = """

    DATA    FFL     %s/frames.ffl
    
    //** list of channels you want to process
    DATA    CHANNELS %s
    
    //** native sampling frequency (Hz) of working channels (as many as
    //the number of input channels)
    DATA    NATIVEFREQUENCY %s
    
    //** working sampling (one value for all channels)
    DATA    SAMPLEFREQUENCY 32
    
    //*************************************************************************************
    //************************        SEARCH PARAMETERS
    //*****************************
    //*************************************************************************************
    
    //** chunk duration in seconds (must be an integer)
    PARAMETER       CHUNKDURATION   864
    
    //** segment duration in seconds (must be an integer)
    PARAMETER       BLOCKDURATION   512
    
    //** overlap duration between segments in seconds (must be an integer)
    PARAMETER       OVERLAPDURATION  160
    
    //** search frequency range
    PARAMETER       FREQUENCYRANGE  0.1      10
    
    //** search Q range
    PARAMETER       QRANGE          3.3166  141
    
    //** maximal mismatch between 2 consecutive tiles (0<MM<1)
    PARAMETER       MISMATCHMAX     0.2
    
    //*************************************************************************************
    //************************            TRIGGERS
    //*****************************
    //*************************************************************************************
    
    //** tile SNR threshold
    TRIGGER         SNRTHRESHOLD    5
    
    //** maximum number of triggers per file
    TRIGGER         NMAX            500000
    
    //*************************************************************************************
    //************************             OUTPUT
    //*****************************
    //*************************************************************************************
    
    //** full path to output directory
    OUTPUT  DIRECTORY       %s/
    
    OUTPUT  FORMAT   xml
    
    """%(omicronDirectory,channelList,samplerateList,omicronDirectory)

    return paramsFile

