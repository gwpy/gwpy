#!/usr/bin/python

import os, glob, optparse, shutil, warnings, matplotlib, pickle, math, copy, pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal, scipy.stats, scipy.fftpack, scipy.ndimage.filters
from collections import namedtuple
from operator import itemgetter
import gwpy.seismon.seismon_NLNM, gwpy.seismon.seismon_html
import gwpy.seismon.seismon_eqmon

import bokeh.objects, bokeh.glyphs, bokeh.plotting

__author__ = "Michael Coughlin <michael.coughlin@ligo.org>"
__date__ = "2012/8/26"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def create_plot(x,y,sess):
    source = bokeh.objects.ColumnDataSource(data=dict(x=x, y=y))

    xdr = bokeh.objects.Range1d(start=np.min(x) + 5, end=np.max(x) - 5)
    ydr = bokeh.objects.Range1d(start=np.min(y), end=np.max(y))

    #line = bokeh.glyphs.Line(x="x", y="y", line_color="blue", line_width=2)
    circle = bokeh.glyphs.Circle(x="x", y="y", fill="red", radius=5, line_color="black")
    glyph_renderer = bokeh.objects.GlyphRenderer(
        data_source = source,
        xdata_range = xdr,
        ydata_range = ydr,
        glyph = circle)

    pantool = bokeh.objects.PanTool(dataranges=[xdr, ydr], dimensions=("width","height"))
    zoomtool = bokeh.objects.ZoomTool(dataranges=[xdr,ydr], dimensions=("width","height"))
    resizetool = bokeh.objects.ResizeTool()
    selectiontool = bokeh.objects.SelectionTool()
    previewsavetool = bokeh.objects.PreviewSaveTool()
    overlay = bokeh.objects.BoxSelectionOverlay(tool=selectiontool)

    plot = bokeh.objects.Plot(x_range=xdr, y_range=ydr, data_sources=[source],
        border= 80)
    xaxis = bokeh.objects.LinearAxis(plot=plot, dimension=0, axis_label="X")
    yaxis = bokeh.objects.LinearAxis(plot=plot, dimension=1, axis_label="Y")
    xgrid = bokeh.objects.Rule(plot=plot, dimension=0)
    ygrid = bokeh.objects.Rule(plot=plot, dimension=1)

    plot.renderers.append(glyph_renderer)
    plot.renderers.append(overlay)
    plot.tools = [pantool,zoomtool,resizetool,selectiontool,previewsavetool]

    sess.add(plot, glyph_renderer, xaxis, yaxis, xgrid, ygrid, source, xdr, ydr,
        pantool, zoomtool, resizetool, selectiontool,previewsavetool,overlay)
    sess.plotcontext.children.append(plot)

    return sess

def earthquakes_plot(params,channel,sess):

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

    attributeDics = gwpy.seismon.seismon_eqmon.retrieve_earthquakes(params)
    attributeDics = sorted(attributeDics, key=itemgetter("Magnitude"), reverse=True)

    gps = []
    amp = []
    mag = []

    for attributeDic in attributeDics:

        #attributeDic = calculate_traveltimes(attributeDic)

        traveltimes = attributeDic["traveltimes"][ifo]

        gps.append(max(traveltimes["Rtimes"]))
        amp.append(traveltimes["Rfamp"][0])
        mag.append(attributeDic["Magnitude"])

    gps = np.array(gps)
    amp = np.array(amp)
    mag = np.array(mag)

    indexes = np.where(amp > 0)
    gps = gps[indexes]
    amp = amp[indexes]

    gps = (gps - 1056672016.0)/86400.0
    amp = np.log10(amp)

    indexes = np.where(amp > -10)
    gps = gps[indexes]
    amp = amp[indexes]

    sess = create_plot(gps,amp,sess)

    return sess

def psd_plot(params,channel,sess):

    psdLocation = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore
    psdLocation = os.path.join(psdLocation,str(params["fftDuration"]))
    files = glob.glob(os.path.join(psdLocation,"*.txt"))

    files = sorted(files)

    ttStart = []
    ttEnd = []
    amp = []

    # Break up entire frequency band into 6 segments
    ff_ave = [1/float(128), 1/float(64),  0.1, 1, 3, 5, 10]

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < params["gpsStart"]) or (thisTTEnd > params["gpsEnd"]):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data = np.loadtxt(file)
        thisSpectra = data[:,1]
        thisFreq = data[:,0]

        freqAmps = []

        for i in xrange(len(ff_ave)-1):
            newSpectraNow = []
            for j in xrange(len(thisFreq)):
                if ff_ave[i] <= thisFreq[j] and thisFreq[j] <= ff_ave[i+1]:
                    newSpectraNow.append(thisSpectra[j])
            freqAmps.append(np.mean(newSpectraNow))

        thisAmp = freqAmps[1]
        amp.append(thisAmp)

    ttStart = np.array(ttStart)
    ttEnd = np.array(ttEnd)
    amp = np.array(amp)

    gps = (ttStart - 1056672016.0) / 86400.0
    amp = np.log10(amp)

    sess = create_plot(gps,amp,sess)

    return sess
 
def channel_page(params, channel):

    plotLocation = params["path"] + "/" + channel.station_underscore
    if not os.path.isdir(plotLocation):
        os.makedirs(plotLocation)

    outputFile = os.path.join(plotLocation,"bokeh.html")

    sess = bokeh.session.HTMLFileSession(outputFile)
    sess = earthquakes_plot(params,channel,sess)
    sess = psd_plot(params,channel,sess)
    sess.save(js="relative", css="relative", rootdir=os.path.abspath("."))
    print "Wrote", outputFile


