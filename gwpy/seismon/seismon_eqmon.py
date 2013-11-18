#!/usr/bin/python

import os, sys, time, glob, math, matplotlib, random, string
import numpy as np
from datetime import datetime
from operator import itemgetter
import glue.segments, glue.segmentsUtils
from lxml import etree
import scipy.spatial
import smtplib, email.mime.text

import lal.gpstime

import gwpy.seismon.seismon_utils, gwpy.seismon.seismon_eqmon_plot
import gwpy.seismon.seismon_pybrain

def run_earthquakes(params,segment):
    """@run earthquakes prediction.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    timeseriesDirectory = os.path.join(params["path"],"timeseries")
    gwpy.seismon.seismon_utils.mkdir(timeseriesDirectory)
    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    gwpy.seismon.seismon_utils.mkdir(earthquakesDirectory)
    noticesDirectory = os.path.join(params["path"],"notices")
    gwpy.seismon.seismon_utils.mkdir(noticesDirectory)
    segmentsDirectory = os.path.join(params["path"],"segments")
    gwpy.seismon.seismon_utils.mkdir(segmentsDirectory)
    predictionDirectory = params["dirPath"] + "/Text_Files/Prediction/"
    gwpy.seismon.seismon_utils.mkdir(predictionDirectory)

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    attributeDics = retrieve_earthquakes(params,gpsStart,gpsEnd)
    attributeDics = sorted(attributeDics, key=itemgetter("Magnitude"), reverse=True)

    earthquakesFile = os.path.join(earthquakesDirectory,"earthquakes.txt")
    earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
    timeseriesFile = os.path.join(timeseriesDirectory,"amp.txt")
    noticesFile = os.path.join(noticesDirectory,"notices.txt")
    segmentsFile = os.path.join(segmentsDirectory,"segments.txt")
    predictionFile = os.path.join(predictionDirectory,"%d-%d.txt"%(gpsStart,gpsEnd))

    f = open(earthquakesFile,"w+")
    g = open(noticesFile,"w+")
    h = open(segmentsFile,"w+")

    threshold = 10**(-7)
    #threshold = 0

    amp = 0
    segmentlist = glue.segments.segmentlist()

    for attributeDic in attributeDics:

        if params["doEarthquakesVelocityMap"]:
            attributeDic = calculate_traveltimes_velocitymap(attributeDic)
        if params["doEarthquakesLookUp"]:
            attributeDic = calculate_traveltimes_lookup(attributeDic)        
    
        traveltimes = attributeDic["traveltimes"][ifo]

        arrival = np.min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        departure = np.max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        arrival_floor = np.floor(arrival / 100.0) * 100.0
        departure_ceil = np.ceil(departure / 100.0) * 100.0

        check_intersect = (arrival >= gpsStart) and (departure <= gpsEnd)

        if check_intersect:
            amp += traveltimes["Rfamp"][0]

            if traveltimes["Rfamp"][0] >= threshold:

                f.write("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"])))

                print "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.5e %d %d %.1f %.1f %e\n"%(attributeDic["GPS"],attributeDic["Magnitude"],max(traveltimes["Ptimes"]),max(traveltimes["Stimes"]),max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),traveltimes["Rfamp"][0],arrival_floor,departure_ceil,attributeDic["Latitude"],attributeDic["Longitude"],max(traveltimes["Distances"]))

                g.write("%.1f %.1f %.5e\n"%(arrival,departure-arrival,traveltimes["Rfamp"][0]))
                h.write("%.0f %.0f\n"%(arrival_floor,departure_ceil))

                segmentlist.append(glue.segments.segment(arrival_floor,departure_ceil))
    
    f.close()
    g.close()
    h.close()

    f = open(timeseriesFile,"w+")
    f.write("%e\n"%(amp))
    f.close()

    f = open(predictionFile,"w+")
    f.write("%e\n"%(amp))
    f.close()

    write_info(earthquakesXMLFile,attributeDics)

    if params["doEarthquakesOnline"]:
        sender = params["userEmail"]
        receivers = [params["userEmail"]]

        lines = [line for line in open(earthquakesFile)]
        if lines == []:
            return segmentlist

        message = ""
        for line in lines:
            message = "%s\n%s"%(message,line)

        s = smtplib.SMTP('localhost')
        s.sendmail(sender,receivers, message)         
        s.quit()
        print "mail sent"

    return segmentlist

def run_earthquakes_analysis(params,segment):
    """@run earthquakes analysis.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")
    attributeDics = gwpy.seismon.seismon_utils.read_eqmons(earthquakesXMLFile)

    minDiff = 10*60
    coincident = []
    for i in xrange(len(attributeDics)):
        attributeDic1 = attributeDics[i]
        for j in xrange(len(attributeDics)):
            if j <= i:
                continue
            attributeDic2 = attributeDics[j]
            gpsDiff = attributeDic1["GPS"] - attributeDic2["GPS"]
            if np.absolute(gpsDiff) < minDiff:
                coincident.append(j)
    coincident = list(set(coincident))
    print "%d coincident earthquakes"%len(coincident)
    indexes = list(set(range(len(attributeDics))) - set(coincident))
    attributeDicsKeep = []
    for index in indexes:
        attributeDicsKeep.append(attributeDics[index])
    attributeDics = attributeDicsKeep

    data = {}
    data["prediction"] = loadPredictions(params,segment)
    data["earthquakes_all"] = loadEarthquakes(params,attributeDics)

    data["channels"] = {}
    for channel in params["channels"]:
        data["channels"][channel.station_underscore] = {}
        data["channels"][channel.station_underscore]["info"] = channel
        data["channels"][channel.station_underscore]["psd"] = loadChannelPSD(params,channel,segment)
        data["channels"][channel.station_underscore]["timeseries"] = loadChannelTimeseries(params,channel,segment)
        data["channels"][channel.station_underscore]["earthquakes"] = loadChannelEarthquakes(params,channel,attributeDics)
    data["earthquakes"] = {}
    for attributeDic in attributeDics:
        data["earthquakes"][attributeDic["eventName"]] = {}
        data["earthquakes"][attributeDic["eventName"]]["attributeDic"] = attributeDic
        data["earthquakes"][attributeDic["eventName"]]["data"] = loadEarthquakeChannels(params,attributeDic)

    if params["doKML"]:
        kmlName = os.path.join(earthquakesDirectory,"earthquakes_time.kml")
        create_kml(params,attributeDics,data,"time",kmlName)
        kmlName = os.path.join(earthquakesDirectory,"earthquakes_amplitude.kml") 
        create_kml(params,attributeDics,data,"amplitude",kmlName)

    if params["doEarthquakesTraining"]:
        gwpy.seismon.seismon_pybrain.earthquakes_training(params,attributeDics,data)
    if params["doEarthquakesTesting"]:
        gwpy.seismon.seismon_pybrain.earthquakes_testing(params,attributeDics,data)

    save_predictions(params,data)

    if params["doPlots"]:

        plotName = os.path.join(earthquakesDirectory,"prediction.png")
        gwpy.seismon.seismon_eqmon_plot.prediction(data,plotName)
        plotName = os.path.join(earthquakesDirectory,"residual.png")
        gwpy.seismon.seismon_eqmon_plot.residual(data,plotName)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_timeseries.png")
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station(params,data,"timeseries",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_psd.png")
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station(params,data,"psd",plotName)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_amplitude.png")
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_time.png")
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"time",plotName)
        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_residual.png")
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"residual",plotName)

        name = ""
        for key in data["channels"].iterkeys():
            if name == "":
               name = key
            else:
               name = "%s_%s"%(name,key)

        plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_heatmap_time_%s.png"%name)
        gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance_heatmap(params,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"station_amplitude.png")
        gwpy.seismon.seismon_eqmon_plot.station_plot(params,attributeDics,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"station_time.png")
        gwpy.seismon.seismon_eqmon_plot.station_plot(params,attributeDics,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_station_amplitude.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_station_plot(params,attributeDics,data,"amplitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_station_time.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_station_plot(params,attributeDics,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_magnitudes.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_plot(params,attributeDics,"Magnitude",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_traveltimes.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_plot(params,attributeDics,"Traveltimes",plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap_restimates.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_plot(params,attributeDics,"Restimates",plotName)

        plotName = os.path.join(earthquakesDirectory,"worldmap_channel_traveltimes.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_channel_plot(params,data,"time",plotName)

        plotName = os.path.join(earthquakesDirectory,"restimates.png")
        gwpy.seismon.seismon_eqmon_plot.restimates(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"magnitudes.png")
        gwpy.seismon.seismon_eqmon_plot.magnitudes(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"magnitudes_latencies.png")
        gwpy.seismon.seismon_eqmon_plot.magnitudes_latencies(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"latencies_sent.png")
        gwpy.seismon.seismon_eqmon_plot.latencies_sent(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"latencies_written.png")

        gwpy.seismon.seismon_eqmon_plot.latencies_written(params,attributeDics,plotName)
        plotName = os.path.join(earthquakesDirectory,"traveltimes%s.png"%params["ifo"])
        gwpy.seismon.seismon_eqmon_plot.traveltimes(params,attributeDics,ifo,gpsEnd,plotName)
        plotName = os.path.join(earthquakesDirectory,"worldmap.png")
        gwpy.seismon.seismon_eqmon_plot.worldmap_wavefronts(params,attributeDics,gpsEnd,plotName)

        if params["doEarthquakesVelocityMap"]:
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_velocitymap.png")
            gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"velocitymap",plotName)
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_velocitymapresidual.png")
            gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"velocitymapresidual",plotName)
            plotName = os.path.join(earthquakesDirectory,"velocitymap.png")
            gwpy.seismon.seismon_eqmon_plot.worldmap_velocitymap(params,plotName)

        if params["doEarthquakesLookUp"]:
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_lookup.png")
            gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"lookup",plotName)
            plotName = os.path.join(earthquakesDirectory,"earthquakes_distance_lookupresidual.png")
            gwpy.seismon.seismon_eqmon_plot.earthquakes_station_distance(params,data,"lookupresidual",plotName)

def save_predictions(params,data):
    """@save file for generating predictions

    @param params
        seismon params dictionary
    @param data
        channel data dictionary
    """

    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    predictionsDirectory = os.path.join(earthquakesDirectory,"predictions")
    gwpy.seismon.seismon_utils.mkdir(predictionsDirectory)

    threshold = 1.5e-6

    for key in data["channels"].iterkeys():
        channel_data = data["channels"][key]["earthquakes"]

        predictionFile = os.path.join(predictionsDirectory,"%s.txt"%key)
        f = open(predictionFile,"w")
        for gps,arrival,departure,latitude, longitude, distance, magnitude, depth,ampMax,ampPrediction,ttDiff in zip(channel_data["gps"],channel_data["arrival"],channel_data["departure"],channel_data["latitude"],channel_data["longitude"],channel_data["distance"],channel_data["magnitude"],channel_data["depth"],channel_data["ampMax"],channel_data["ampPrediction"],channel_data["ttDiff"]):

            if (ampMax < threshold) and (ampPrediction<threshold):
                continue

            f.write("%.0f %.0f %.0f %.2f %.2f %e %.2f %.2f %e %.2f\n"%(gps,arrival,departure,latitude, longitude, distance, magnitude, depth,ampMax,ttDiff))
        f.close()

def create_kml(params,attributeDics,data,type,kmlFile):
    """@create kml

    @param params
        seismon params dictionary
    @param attributeDics
        list of eqmon structures
    @param data
        channel data dictionary
    @param type
        type of worldmap plot
    @param kmlFile
        name of file
    """

    import simplekml

    # Create an instance of Kml
    kml = simplekml.Kml(open=1)

    for attributeDic in attributeDics:
        pnt = kml.newpoint(coords = [(attributeDic["Longitude"],attributeDic["Latitude"])])
        pnt.name = attributeDic["eventName"]
        
        pnt.lookat.longitude = attributeDic["Longitude"]
        pnt.lookat.latitude = attributeDic["Latitude"]

    for channel in params["channels"]:
        channel_data = data["channels"][channel.station_underscore]

        if len(channel_data["timeseries"]["data"]) == 0:
            continue

        pnt = kml.newpoint(coords = [(channel_data["info"].longitude,channel_data["info"].latitude)])
        pnt.name = channel_data["info"].station

        pnt.lookat.longitude = channel_data["info"].longitude
        pnt.lookat.latitude = channel_data["info"].latitude

        #pnt.lookat.longitude = 0
        #pnt.lookat.latitude = 0

        if type == "amplitude":
            z = channel_data["timeseries"]["data"][0] * 1e6
            array = np.linspace(1,2000,1000)
        elif type == "time":
            z = channel_data["timeseries"]["ttMax"][0] - attributeDic["GPS"]
            array = np.linspace(0,10000,1000)

        snrSig,hexcolor = gwpy.seismon.seismon_utils.html_hexcolor(z,array)

        #pnt.style.labelstyle.scale = 0.1  # Text twice as big
        pnt.style.iconstyle.color = hexcolor  # Blue
        pnt.style.iconstyle.scale = 3  # Icon thrice as big
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/road_shield3.png'
        #pnt.style.iconstyle.icon.href = None

    kml.save(kmlFile)

def loadPredictions(params,segment):
    """@load earthquakes predictions.

    @param params
        seismon params dictionary
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    predictionDirectory = params["dirPath"] + "/Text_Files/Prediction/"
    files = glob.glob(os.path.join(predictionDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    amp = []

    for file in files:

        fileSplit = file.split("/")

        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)
        thisAmp = data_out

        amp.append(thisAmp)

    ttStart = np.array(ttStart)
    ttEnd = np.array(ttEnd)
    amp = np.array(amp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["data"] = amp

    return data

def loadEarthquakes(params,attributeDics):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    tt = []
    ttArrival = []
    amp = []
    distance = []
    names = []

    for attributeDic in attributeDics:

        traveltimes = attributeDic["traveltimes"][ifo]

        names.append(attributeDic["eventName"])
        tt.append(attributeDic["GPS"])
        ttArrival.append(max(traveltimes["RthreePointFivetimes"]))
        amp.append(traveltimes["Rfamp"][0])
        distance.append(max(traveltimes["Distances"]))

    tt = np.array(tt)
    ttArrival = np.array(ttArrival)
    amp = np.array(amp)
    distance = np.array(distance)

    data = {}
    data["tt"] = tt
    data["ttArrival"] = ttArrival
    data["data"] = amp
    data["distance"] = distance
    data["names"] = names

    return data

def loadEarthquakeChannels(params,attributeDic):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)
    traveltimes = attributeDic["traveltimes"][ifo]

    ttMax = []
    ttDiff = []
    distance = []
    velocity = []
    ampMax = []
    ampPrediction = []
    residual = []

    for channel in params["channels"]:
        earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
        earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))

        if not os.path.isfile(earthquakesFile):
            continue

        data_out = np.loadtxt(earthquakesFile)
        ttMax.append(data_out[0])
        ttDiff.append(data_out[1])
        distance.append(data_out[2])
        velocity.append(data_out[3])
        ampMax.append(data_out[4])
        ampPrediction.append(traveltimes["Rfamp"][0])
        thisResidual = (data_out[4] - traveltimes["Rfamp"][0])/traveltimes["Rfamp"][0]
        residual.append(thisResidual)

    ttMax = np.array(ttMax)
    ttDiff = np.array(ttDiff)
    distance = np.array(distance)
    velocity = np.array(velocity)
    ampMax = np.array(ampMax)
    ampPrediction = np.array(ampPrediction)
    residual = np.array(residual)

    data = {}
    data["tt"] = ttMax
    data["ttDiff"] = ttDiff
    data["distance"] = distance
    data["velocity"] = velocity
    data["ampMax"] = ampMax
    data["ampPrediction"] = ampPrediction
    data["residual"] = residual

    return data

def loadChannelEarthquakes(params,channel,attributeDics):
    """@load earthquakes dictionaries.

    @param params
        seismon params dictionary
    @param attributeDics
        list of seismon earthquake structures
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    ttMax = []
    ttDiff = []
    distance = []
    velocity = []
    ampMax = [] 
    ampPrediction = []
    depth = []
    magnitude = []
    latitude = []
    longitude = []
    arrival = []
    departure = []
    gps = []
    residual = []
    Rvelocitymaptime = []
    RvelocitymaptimeDiff = []
    RvelocitymaptimeResidual = []
    Rlookuptime = []
    RlookuptimeDiff = []
    RlookuptimeResidual = [] 
 
    print "Large residuals"
    for attributeDic in attributeDics:

        if params["ifo"] == "IRIS":
            attributeDic = gwpy.seismon.seismon_eqmon.ifotraveltimes(attributeDic, "IRIS", channel.latitude, channel.longitude)
            traveltimes = attributeDic["traveltimes"]["IRIS"]
        else:
            traveltimes = attributeDic["traveltimes"][ifo]

        earthquakesDirectory = params["dirPath"] + "/Text_Files/Earthquakes/" + channel.station_underscore + "/" + str(params["fftDuration"])
        earthquakesFile = os.path.join(earthquakesDirectory,"%s.txt"%(attributeDic["eventName"]))

        if not os.path.isfile(earthquakesFile):
            continue

        thisArrival = np.min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        thisDeparture = np.max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        arrival_floor = np.floor(thisArrival / 100.0) * 100.0
        departure_ceil = np.ceil(thisDeparture / 100.0) * 100.0

        data_out = np.loadtxt(earthquakesFile)
        ttMax.append(data_out[0])
        ttDiff.append(data_out[1])
        distance.append(data_out[2])
        velocity.append(data_out[3])
        ampMax.append(data_out[4])
        ampPrediction.append(traveltimes["Rfamp"][0])
        depth.append(attributeDic["Depth"])
        magnitude.append(attributeDic["Magnitude"])
        latitude.append(attributeDic["Latitude"])
        longitude.append(attributeDic["Longitude"])
        arrival.append(arrival_floor)
        departure.append(departure_ceil)
        gps.append(attributeDic["GPS"])
        thisResidual = (data_out[4] - traveltimes["Rfamp"][0])/traveltimes["Rfamp"][0]
        residual.append(thisResidual)

        if params["doEarthquakesVelocityMap"]:
            thisRvelocitymaptime = max(traveltimes["Rvelocitymaptimes"])
            Rvelocitymaptime.append(thisRvelocitymaptime)
            thisRvelocitymaptimeDiff = thisRvelocitymaptime - attributeDic["GPS"]
            RvelocitymaptimeDiff.append(thisRvelocitymaptimeDiff)
            thisRvelocitymaptimeResidual = (thisRvelocitymaptimeDiff - data_out[1]) / data_out[1]
            RvelocitymaptimeResidual.append(thisRvelocitymaptimeResidual)
        if params["doEarthquakesLookUp"]:
            thisRlookuptime = traveltimes["Rlookuptime"][0]
            Rlookuptime.append(thisRlookuptime)
            thisRlookuptimeDiff = thisRlookuptime - attributeDic["GPS"]
            RlookuptimeDiff.append(thisRlookuptimeDiff)
            thisRlookuptimeResidual = (thisRlookuptimeDiff - data_out[1]) / data_out[1]
            RlookuptimeResidual.append(thisRlookuptimeResidual)

        if thisResidual > 100:
            print "%.0f %.0f %.0f %e"%(attributeDic["GPS"],arrival_floor,departure_ceil,thisResidual)

    ttMax = np.array(ttMax)
    ttDiff = np.array(ttDiff)
    distance = np.array(distance)
    velocity = np.array(velocity)
    ampMax = np.array(ampMax)
    ampPrediction = np.array(ampPrediction)
    depth = np.array(depth)
    magnitude = np.array(magnitude)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    arrival = np.array(arrival)
    departure = np.array(departure)
    gps = np.array(gps)
    residual = np.array(residual)
    Rvelocitymaptime = np.array(Rvelocitymaptime)
    RvelocitymaptimeDiff = np.array(RvelocitymaptimeDiff)
    RvelocitymaptimeResidual = np.array(RvelocitymaptimeResidual)
    Rlookuptime = np.array(Rlookuptime)
    RlookuptimeDiff = np.array(RlookuptimeDiff)
    RlookuptimeResidual = np.array(RlookuptimeResidual)

    data = {}
    data["tt"] = ttMax
    data["ttDiff"] = ttDiff
    data["distance"] = distance
    data["velocity"] = velocity
    data["ampMax"] = ampMax
    data["ampPrediction"] = ampPrediction
    data["depth"] = depth
    data["magnitude"] = magnitude
    data["latitude"] = latitude
    data["longitude"] = longitude
    data["arrival"] = arrival
    data["departure"] = departure 
    data["gps"] = gps
    data["residual"] = residual
    data["Rvelocitymaptime"] = Rvelocitymaptime
    data["RvelocitymaptimeDiff"] = RvelocitymaptimeDiff
    data["RvelocitymaptimeResidual"] = RvelocitymaptimeResidual
    data["Rlookuptime"] = Rlookuptime
    data["RlookuptimeDiff"] = RlookuptimeDiff
    data["RlookuptimeResidual"] = RlookuptimeResidual

    return data

def loadChannelPSD(params,channel,segment):
    """@load channel PSDs.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    # Break up entire frequency band into 6 segments
    ff_ave = [1/float(128), 1/float(64),  0.1, 1, 3, 5, 10]

    psdDirectory = params["dirPath"] + "/Text_Files/PSD/" + channel.station_underscore + "/" + str(params["fftDuration"])

    files = glob.glob(os.path.join(psdDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    amp = []

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)
        thisSpectra_out = data_out[:,1]
        thisFreq_out = data_out[:,0]

        freqAmps = []
        for i in xrange(len(ff_ave)-1):
            newSpectraNow = []
            for j in xrange(len(thisFreq_out)):
                if ff_ave[i] <= thisFreq_out[j] and thisFreq_out[j] <= ff_ave[i+1]:
                    newSpectraNow.append(thisSpectra_out[j])
                    freqAmps.append(np.mean(newSpectraNow))

        thisAmp = freqAmps[1]
        amp.append(thisAmp)

    ttStart = np.array(ttStart)
    ttEnd = np.array(ttEnd)
    amp = np.array(amp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["data"] = amp

    return data

def loadChannelTimeseries(params,channel,segment):
    """@load channel timeseries.

    @param params
        seismon params dictionary
    @param channel
        seismon channel structure
    @param segment
        [start,end] gps
    """

    gpsStart = segment[0]
    gpsEnd = segment[1]

    timeseriesDirectory = params["dirPath"] + "/Text_Files/Timeseries/" + channel.station_underscore + "/" + str(params["fftDuration"])

    files = glob.glob(os.path.join(timeseriesDirectory,"*.txt"))
    files = sorted(files)

    ttStart = []
    ttEnd = []
    ttMax = []
    amp = []

    for file in files:

        fileSplit = file.split("/")
        txtFile = fileSplit[-1].replace(".txt","")
        txtFileSplit = txtFile.split("-")
        thisTTStart = int(txtFileSplit[0])
        thisTTEnd = int(txtFileSplit[1])

        if (thisTTStart < gpsStart) or (thisTTEnd > gpsEnd):
            continue

        ttStart.append(thisTTStart)
        ttEnd.append(thisTTEnd)

        data_out = np.loadtxt(file)

        thisttMax = data_out[1,0]
        thisAmp = data_out[1,1]
        ttMax.append(thisttMax)
        amp.append(thisAmp)

    data = {}
    data["ttStart"] = ttStart
    data["ttEnd"] = ttEnd
    data["ttMax"] = ttMax
    data["data"] = amp

    return data

def write_info(file,attributeDics):
    """@write eqmon file

    @param file
        eqmon file
    @param attributeDics
        list of eqmon structures
    """

    baseroot = etree.Element('eqmon')
    for attributeDic in attributeDics:
        root = etree.SubElement(baseroot,attributeDic["eventName"])
        for key, value in attributeDic.items():
            if not key == "traveltimes":
                element = etree.SubElement(root,key)
                element.text = str(value)
        element = etree.SubElement(root,'traveltimes')
        for key, value in attributeDic["traveltimes"].items():
            subelement = etree.SubElement(element,key)
            for category in value:
                subsubelement = etree.SubElement(subelement,category)
                subsubelement.text = write_array(value[category])

    tree = etree.ElementTree(baseroot)
    tree.write(file, pretty_print=True, xml_declaration=True)

def write_array(array):
    """@create string of array values

    @param array
        array of values
    """

    if isinstance(array, float):
        text = str(array)
    else:
        text = ' '.join([str(x) for x in array])
    return text

def parse_xml(element):
    """@parse xml element.

    @param element
        xml element
    """

    subdic = {}

    numChildren = 0
    for subelement in element.iterchildren():
        tag = str(subelement.tag)
        tag = tag.replace("{http://www.usgs.gov/ansseqmsg}","")
        tag = tag.replace("{http://quakeml.org/xmlns/quakeml/1.2}","")
        tag = tag.replace("{http://quakeml.org/xmlns/bed/1.2}","")
        subdic[tag] = parse_xml(subelement)
        numChildren += 1

    if numChildren == 0:
        value = str(element.text)
        value = value.replace("{http://www.usgs.gov/ansseqmsg}","")
    else:
        value = subdic

    tag = str(element.tag)
    tag = tag.replace("{http://www.usgs.gov/ansseqmsg}","")
    tag = tag.replace("{http://quakeml.org/xmlns/quakeml/1.2}","")
    tag = tag.replace("{http://quakeml.org/xmlns/bed/1.2}","")
    dic = value

    return dic

def read_eqxml(file,eventName):
    """@read eqxml file.

    @param file
        eqxml file
    @param eventName
        name of earthquake event
    """

    tree = etree.parse(file)
    root = tree.getroot()
    dic = parse_xml(root)

    attributeDic = {}

    if not "Origin" in dic["Event"] or not "Magnitude" in dic["Event"]["Origin"]:
        return attributeDic

    attributeDic["Longitude"] = float(dic["Event"]["Origin"]["Longitude"])
    attributeDic["Latitude"] = float(dic["Event"]["Origin"]["Latitude"])
    attributeDic["Depth"] = float(dic["Event"]["Origin"]["Depth"])
    attributeDic["eventID"] = dic["Event"]["EventID"]
    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = float(dic["Event"]["Origin"]["Magnitude"]["Value"])

    if "Region" in dic["Event"]["Origin"]:
        attributeDic["Region"] = dic["Event"]["Origin"]["Region"]
    else:
        attributeDic["Region"] = "N/A"

    attributeDic["Time"] = dic["Event"]["Origin"]["Time"]
    timeString = attributeDic["Time"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())

    attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['UTC'] = float(dt.strftime("%s"))

    attributeDic["Sent"] = dic["Sent"]
    timeString = attributeDic["Sent"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = float(dt.strftime("%s"))

    attributeDic["DataSource"] = dic["Source"]
    attributeDic["Version"] = dic["Event"]["Version"]

    if "Type" in dic["Event"]:
        attributeDic["Type"] = dic["Event"]["Type"]
    else:
        attributeDic["Type"] = "N/A"  

    if dic["Event"]["Origin"]["Status"] == "Automatic":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.fromtimestamp(time.mktime(tm))

    attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def read_quakeml(file,eventName):
    """@read quakeml file.

    @param file
        quakeml file
    @param eventName
        name of earthquake event
    """

    tree = etree.parse(file)
    root = tree.getroot()
    dic = parse_xml(root)

    attributeDic = {}

    if "origin" not in dic["eventParameters"]["event"]:
        return attributeDic

    attributeDic["Longitude"] = float(dic["eventParameters"]["event"]["origin"]["longitude"]["value"])
    attributeDic["Latitude"] = float(dic["eventParameters"]["event"]["origin"]["latitude"]["value"])
    attributeDic["Depth"] = float(dic["eventParameters"]["event"]["origin"]["depth"]["value"]) / 1000
    attributeDic["eventID"] = ""
    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = float(dic["eventParameters"]["event"]["magnitude"]["mag"]["value"])

    attributeDic["Time"] = dic["eventParameters"]["event"]["origin"]["time"]["value"]
    timeString = attributeDic["Time"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['UTC'] = float(dt.strftime("%s"))

    attributeDic["Sent"] = dic["eventParameters"]["event"]["creationInfo"]["creationTime"]
    timeString = attributeDic["Sent"].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())
    attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = float(dt.strftime("%s"))

    attributeDic["DataSource"] = dic["eventParameters"]["event"]["creationInfo"]["agencyID"]
    #attributeDic["Version"] = float(dic["eventParameters"]["event"]["creationInfo"]["version"])
    attributeDic["Type"] = dic["eventParameters"]["event"]["type"]

    if "evalulationMode" in dic["eventParameters"]["event"]:
        if dic["eventParameters"]["event"]["origin"]["evaluationMode"] == "automatic":
            attributeDic["Review"] = "Automatic"
        else:
            attributeDic["Review"] = "Manual"
    else:
        attributeDic["Review"] = "Unknown"

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.fromtimestamp(time.mktime(tm))

    attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def read_eqmon(params,file):
    """@read eqmon file.

    @param params
        seismon params struct
    @param file
        name of eqmon file
    """

    attributeDic = {}
    tree = etree.parse(file)
    root = tree.getroot()       # get the document root
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
        #arrivalMin = min([max(traveltimes["Rtimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        #arrivalMax = max([max(traveltimes["Rtimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        arrivalMin = min([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])
        arrivalMax = max([max(traveltimes["Rtwotimes"]),max(traveltimes["RthreePointFivetimes"]),max(traveltimes["Rfivetimes"]),max(traveltimes["Stimes"]),max(traveltimes["Ptimes"])])

        attributeDic["traveltimes"][ifoName]["arrivalMin"] = arrivalMin
        attributeDic["traveltimes"][ifoName]["arrivalMax"] = arrivalMax
        #if params["gps"] <= attributeDic["traveltimes"][ifoName]["arrivalMax"]:
        #    attributeDic["doPlots"] = 1
    return attributeDic

def jsonread(event):
    """@read json event.

    @param event
        json event
    """

    attributeDic = {}
    attributeDic["Longitude"] = event["geometry"]["coordinates"][0]
    attributeDic["Latitude"] = event["geometry"]["coordinates"][1]
    attributeDic["Depth"] = event["geometry"]["coordinates"][2]
    attributeDic["eventID"] = event["properties"]["code"]
    attributeDic["eventName"] = event["properties"]["ids"].split(",")[1]
    attributeDic["Magnitude"] = event["properties"]["mag"]
    attributeDic["UTC"] = float(event["properties"]["time"]) / 1000.0
    attributeDic["DataSource"] = event["properties"]["sources"].replace(",","")
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = event["properties"]["place"]

    if event["properties"]["status"] == "AUTOMATIC":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    Time = time.gmtime(attributeDic["UTC"])
    dt = datetime.fromtimestamp(time.mktime(Time))

    attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    SentTime = time.gmtime()
    dt = datetime.fromtimestamp(time.mktime(SentTime))
    attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()

    attributeDic['Time'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", Time)
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.fromtimestamp(time.mktime(tm))
    attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def irisread(event):
    """@read iris event.

    @param event
        iris event
    """

    attributeDic = {}

    attributeDic['Time'] = str(event.origins[0].time)
    timeString = attributeDic['Time'].replace("T"," ").replace("Z","")
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S.%f")
    tm = time.struct_time(dt.timetuple())

    attributeDic["Longitude"] = event.origins[0].longitude
    attributeDic["Latitude"] = event.origins[0].latitude
    attributeDic["Depth"] = event.origins[0].depth
    attributeDic["eventID"] = event.origins[0].region

    attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['UTC'] = float(dt.strftime("%s"))

    eventID = "%.0f"%attributeDic['GPS']
    eventName = ''.join(["iris",str(eventID)])

    attributeDic["eventName"] = eventName
    attributeDic["Magnitude"] = event.magnitudes[0].mag
    attributeDic["DataSource"] = "IRIS"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = event.origins[0].region

    if event.origins[0].evaluation_status == "AUTOMATIC":
        attributeDic["Review"] = "Automatic"
    else:
        attributeDic["Review"] = "Manual"

    SentTime = time.gmtime()
    dt = datetime.fromtimestamp(time.mktime(SentTime))
    attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['SentUTC'] = time.time()
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    dt = datetime.fromtimestamp(time.mktime(tm))

    attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(dt))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def databaseread(event):

    attributeDic = {}
    eventSplit = event.split(",")

    date = eventSplit[0]

    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(date[11:13])
    minute = int(date[14:16])
    second = int(date[17:19])

    timeString = "%d-%02d-%02d %02d:%02d:%02d"%(year,month,day,hour,minute,second)
    dt = datetime.strptime(timeString, "%Y-%m-%d %H:%M:%S")

    eventID = int(eventSplit[11])
    eventName = ''.join(["db",str(eventID)])

    attributeDic["Longitude"] = float(eventSplit[2])
    attributeDic["Latitude"] = float(eventSplit[1])
    attributeDic["Depth"] = float(eventSplit[3])
    attributeDic["eventID"] = float(eventID)
    attributeDic["eventName"] = eventName
    try:
        attributeDic["Magnitude"] = float(eventSplit[4])
    except:
        attributeDic["Magnitude"] = 0
    tm = time.struct_time(dt.timetuple())
    attributeDic['GPS'] = float(lal.gpstime.utc_to_gps(tm))
    attributeDic['UTC'] = float(dt.strftime("%s"))
    attributeDic["DataSource"] = "DB"
    attributeDic["Version"] = 1.0
    attributeDic["Type"] = 1.0
    attributeDic['Region'] = "N/A"
    attributeDic["Review"] = "Manual"

    SentTime = time.gmtime()
    attributeDic['SentGPS'] = float(lal.gpstime.utc_to_gps(SentTime))
    attributeDic['SentUTC'] = time.time()

    attributeDic['Time'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", tm)
    attributeDic['Sent'] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", SentTime)

    attributeDic = calculate_traveltimes(attributeDic)
    tm = time.struct_time(time.gmtime())
    attributeDic['WrittenGPS'] = float(lal.gpstime.utc_to_gps(tm))
    attributeDic['WrittenUTC'] = float(time.time())

    return attributeDic

def calculate_traveltimes_lookup(attributeDic):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes_lookup(attributeDic, "LHO", 46.6475, -119.5986)
    attributeDic = ifotraveltimes_lookup(attributeDic, "LLO", 30.4986, -90.7483)
    attributeDic = ifotraveltimes_lookup(attributeDic, "GEO", 52.246944, 9.808333)
    attributeDic = ifotraveltimes_lookup(attributeDic, "VIRGO", 43.631389, 10.505)
    attributeDic = ifotraveltimes_lookup(attributeDic, "FortyMeter", 34.1391, -118.1238)
    attributeDic = ifotraveltimes_lookup(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def calculate_traveltimes_velocitymap(attributeDic):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes_velocitymap(attributeDic, "LHO", 46.6475, -119.5986)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "LLO", 30.4986, -90.7483)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "GEO", 52.246944, 9.808333)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "VIRGO", 43.631389, 10.505)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "FortyMeter", 34.1391, -118.1238)
    attributeDic = ifotraveltimes_velocitymap(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def calculate_traveltimes(attributeDic): 
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    """

    if not "traveltimes" in attributeDic:
        attributeDic["traveltimes"] = {}

    if not "Latitude" in attributeDic and not "Longitude" in attributeDic:
        return attributeDic

    attributeDic = ifotraveltimes(attributeDic, "LHO", 46.6475, -119.5986)
    attributeDic = ifotraveltimes(attributeDic, "LLO", 30.4986, -90.7483)
    attributeDic = ifotraveltimes(attributeDic, "GEO", 52.246944, 9.808333)
    attributeDic = ifotraveltimes(attributeDic, "VIRGO", 43.631389, 10.505)
    attributeDic = ifotraveltimes(attributeDic, "FortyMeter", 34.1391, -118.1238)
    attributeDic = ifotraveltimes(attributeDic, "Homestake", 44.3465, -103.7574)

    return attributeDic

def do_kdtree(combined_x_y_arrays,points):
    """@calculate nearest points.

    @param combined_x_y_arrays
        list of x,y map points
    @param points
        list of x,y points
    """

    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

def ampRf(M,r,h,Rf0,Rfs,Q0,Qs,cd,ch,rs):
    # Rf amplitude estimate
    # M = magnitude
    # r = distance [km]
    # h = depth [km]

    # Rf0 = Rf amplitude parameter
    # Rfs = exponent of power law for f-dependent Rf amplitude
    # Q0 = Q-value of Earth for Rf waves at 1Hz
    # Qs = exponent of power law for f-dependent Q
    # cd = speed parameter for surface coupling  [km/s]
    # ch = speed parameter for horizontal propagation  [km/s]
    # rs

    # exp(-2*pi*h.*fc./cd), coupling of source to surface waves
    # exp(-2*pi*r.*fc./ch./Q), dissipation

    fc = 10**(2.3-M/2)
    Q = Q0/(fc**Qs)
    Af = Rf0/(fc**Rfs)

    Rf = 1e-3 * M*Af*np.exp(-2*np.pi*h*fc/cd)*np.exp(-2*np.pi*r*(fc/ch)/Q)/(r**(rs))

    return Rf

def ifotraveltimes_lookup(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.taup.taup import getTravelTimes
        from obspy.core.util.geodetics import gps2DistAzimuth
    except:
        print "Enable ObsPy if updated earthquake estimates desired...\n"
        return attributeDic

    distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
    distances = np.linspace(0,distance,1000)
    degrees = (distances/6370000)*(180/np.pi)

    #predictionFile = '/home/mcoughlin/Seismon/H1/H1ER4-1057881616-1061856016/earthquakes/predictions/H1_HPI-BS_STSINF_A_Z_IN1_DQ.txt'
    predictionFile = '/home/mcoughlin/Seismon/H1/H1S5-815097613-875145614/earthquakes/predictions/H0_PEM-LVEA_SEISZ.txt'
    predictions = np.loadtxt(predictionFile)

    combined_x_y_arrays = np.dstack([predictions[:,3],predictions[:,4]])[0]
    points_list = np.dstack([attributeDic["Latitude"], attributeDic["Longitude"]])

    index = do_kdtree(combined_x_y_arrays,points_list)[0]

    distanceNearest,fwdNearest,backNearest = gps2DistAzimuth(predictions[index,3],predictions[index,4],ifolat,ifolon)

    time = attributeDic["GPS"]
    time = time + predictions[index,9] * (distance/distanceNearest)

    attributeDic["traveltimes"][ifo]["Rlookuptime"] = time

    return attributeDic

def ifotraveltimes_velocitymap(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.taup.taup import getTravelTimes
        from obspy.core.util.geodetics import gps2DistAzimuth
    except:
        print "Enable ObsPy if updated earthquake estimates desired...\n"
        return attributeDic

    distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
    distances = np.linspace(0,distance,1000)
    degrees = (distances/6370000)*(180/np.pi)

    distance_delta = distances[1] - distances[0]

    periods = [25.0,27.0,30.0,32.0,35.0,40.0,45.0,50.0,60.0,75.0,100.0,125.0,150.0,200.0,250.0]
    frequencies = 1 / np.array(periods)
   
    fc = 10**(2.3-attributeDic["Magnitude"]/2)
    index = np.argmin(np.absolute(frequencies - fc))

    lats = []
    lons = []
    Rvelocitytimes = []
    velocities = []

    velocityFile = '/home/mcoughlin/Seismon/velocity_maps/GR025_1_GDM52.pix'
    velocity_map = np.loadtxt(velocityFile)
    base_velocity = 3.59738 

    for distance, degree in zip(distances, degrees):

        lon, lat, baz = shoot(attributeDic["Longitude"], attributeDic["Latitude"], fwd, distance/1000)
        lats.append(lat)
        lons.append(lon)

    combined_x_y_arrays = np.dstack([velocity_map[:,0],velocity_map[:,1]])[0]
    points_list = np.dstack([lats, lons])

    indexes = do_kdtree(combined_x_y_arrays,points_list)[0]

    time = attributeDic["GPS"]

    for distance, degree, index in zip(distances, degrees,indexes):

        velocity = 1000 * (1 + 0.01*velocity_map[index,3])*base_velocity

        time_delta = distance_delta / velocity
        time = time + time_delta

        Rvelocitytimes.append(time)
        velocities.append(velocity/1000)

    attributeDic["traveltimes"][ifo]["Rvelocitymaptimes"] = Rvelocitytimes
    attributeDic["traveltimes"][ifo]["Rvelocitymapvelocities"] = velocities

    return attributeDic

def ifotraveltimes(attributeDic,ifo,ifolat,ifolon):
    """@calculate travel times of earthquake

    @param attributeDic
        earthquake stucture
    @param ifo
        ifo name
    @param ifolat
        ifo latitude
    @param ifolon
        ifo longitude
    """

    try:
        from obspy.taup.taup import getTravelTimes
        from obspy.core.util.geodetics import gps2DistAzimuth
    except:
        print "Enable ObsPy if updated earthquake estimates desired...\n"
        return attributeDic

    distance,fwd,back = gps2DistAzimuth(attributeDic["Latitude"],attributeDic["Longitude"],ifolat,ifolon)
    distances = np.linspace(0,distance,100)
    degrees = (distances/6370000)*(180/np.pi)

    Rf0 = 0.89256174
    Rfs = 1.3588703
    Q0 = 4169.7511
    Qs = -0.017424297
    cd = 254.13458
    ch = 10.331297
    rs = 1.0357451

    Rfamp = ampRf(attributeDic["Magnitude"],distances[-1]/1000.0,attributeDic["Depth"],Rf0,Rfs,Q0,Qs,cd,ch,rs)
    
    Pamp = 1e-6
    Samp = 1e-5

    lats = []
    lons = []
    Ptimes = []
    Stimes = []
    #Rtimes = []
    Rtwotimes = []
    RthreePointFivetimes = []
    Rfivetimes = []
    Rfamps = []

    # Pmag = T * 10^(Mb - 5.9 - 0.01*dist)
    # Rmag = T * 10^(Ms - 3.3 - 1.66*log_10(dist))
    T = 20

    for distance, degree in zip(distances, degrees):

        lon, lat, baz = shoot(attributeDic["Longitude"], attributeDic["Latitude"], fwd, distance/1000)
        lats.append(lat)
        lons.append(lon)

        tt = getTravelTimes(delta=degree, depth=attributeDic["Depth"])
        #tt.append({'phase_name': 'R', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/3500, 'd2T/dD2': 0, 'dT/dh': 0})
        tt.append({'phase_name': 'Rtwo', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/2000, 'd2T/dD2': 0, 'dT/dh': 0})
        tt.append({'phase_name': 'RthreePointFive', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/3500, 'd2T/dD2': 0, 'dT/dh': 0})
        tt.append({'phase_name': 'Rfive', 'dT/dD': 0, 'take-off angle': 0, 'time': distance/5000, 'd2T/dD2': 0, 'dT/dh': 0})

        Ptime = -1
        Stime = -1
        Rtime = -1
        for phase in tt:
            if Ptime == -1 and phase["phase_name"][0] == "P":
                Ptime = attributeDic["GPS"]+phase["time"]
            if Stime == -1 and phase["phase_name"][0] == "S":
                Stime = attributeDic["GPS"]+phase["time"]
            #if Rtime == -1 and phase["phase_name"][0] == "R":
            #    Rtime = attributeDic["GPS"]+phase["time"]
            if phase["phase_name"] == "Rtwo":
                Rtwotime = attributeDic["GPS"]+phase["time"]
            if phase["phase_name"] == "RthreePointFive":
                RthreePointFivetime = attributeDic["GPS"]+phase["time"]
            if phase["phase_name"] == "Rfive":
                Rfivetime = attributeDic["GPS"]+phase["time"]
        Ptimes.append(Ptime)
        Stimes.append(Stime)
        #Rtimes.append(Rtime)
        Rtwotimes.append(Rtwotime)
        RthreePointFivetimes.append(RthreePointFivetime)
        Rfivetimes.append(Rfivetime)


    traveltimes = {}
    traveltimes["Latitudes"] = lats
    traveltimes["Longitudes"] = lons
    traveltimes["Distances"] = distances
    traveltimes["Degrees"] = degrees
    traveltimes["Ptimes"] = Ptimes
    traveltimes["Stimes"] = Stimes
    #traveltimes["Rtimes"] = Rtimes
    traveltimes["Rtwotimes"] = Rtwotimes
    traveltimes["RthreePointFivetimes"] = RthreePointFivetimes
    traveltimes["Rfivetimes"] = Rfivetimes
    traveltimes["Rfamp"] = [Rfamp]
    traveltimes["Pamp"] = [Pamp]
    traveltimes["Samp"] = [Samp]

    attributeDic["traveltimes"][ifo] = traveltimes

    return attributeDic

def eventDiff(attributeDics, magnitudeDiff, latitudeDiff, longitudeDiff):
    """@calculate difference between two events

    @param attributeDics
        list of earthquake stuctures
    @param magnitudeDiff
        difference in magnitudes
    @param latitudeDiff
        difference in latitudes
    @param longitudeDiff
        difference in longitudes
    """

    if len(attributeDics) > 1:
        for i in xrange(len(attributeDics)-1):
            if "Magnitude" in attributeDics[i] and "Magnitude" in attributeDics[i+1] and \
                "Latitude" in attributeDics[i] and "Latitude" in attributeDics[i+1] and\
                "Longitude" in attributeDics[i] and "Longitude" in attributeDics[i+1]:

                magnitudeDiff.append(attributeDics[i]["Magnitude"]-attributeDics[i+1]["Magnitude"])
                latitudeDiff.append(attributeDics[i]["Latitude"]-attributeDics[i+1]["Latitude"])
                longitudeDiff.append(attributeDics[i]["Longitude"]-attributeDics[i+1]["Longitude"])
    return magnitudeDiff, latitudeDiff, longitudeDiff

def great_circle_distance(latlong_a, latlong_b):
    """@calculate distance between two points

    @param latlong_a
        first point
    @param latlong_b
        second point
    """

    EARTH_CIRCUMFERENCE = 6378.137 # earth circumference in kilometers

    lat1, lon1 = latlong_a
    lat2, lon2 = latlong_b

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = EARTH_CIRCUMFERENCE * c
    
    return d

def retrieve_earthquakes(params,gpsStart,gpsEnd):
    """@retrieve earthquakes information.

    @param params
        seismon params dictionary
    """

    attributeDics = []
    eventfilesLocation = os.path.join(params["eventfilesLocation"],params["eventfilesType"])
    files = glob.glob(os.path.join(eventfilesLocation,"*.xml"))

#    for numFile in xrange(100):
    for numFile in xrange(len(files)):

        file = files[numFile]

        fileSplit = file.replace(".xml","").split("-")
        gps = float(fileSplit[-1])
        if (gps < gpsStart - 3600) or (gps > gpsEnd):
           continue

        attributeDic = read_eqmon(params,file)

        if attributeDic["Magnitude"] >= params["earthquakesMinMag"]:
            attributeDics.append(attributeDic)

    return attributeDics

def equi(m, centerlon, centerlat, radius):
    """@calculate circle around specified point

    @param m
        basemap projection
    @param centerlon
        longitude of center
    @param centerlat
        latitude of center
    @param radius
        radius of circle
    """

    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    #m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X,Y = m(X,Y)
    return X,Y

def shoot(lon, lat, azimuth, maxdist=None):
    """@Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq

    @param lon
        longitude
    @param lat
        latitude
    @param azimuth
        azimuth
    @param maxdist
        maximum distance

    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS= 0.00000000005
    if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
        alert("Only N-S courses are meaningful, starting at a pole!")

    a=6378.13/1.852
    f=1/298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf==0):
        b=0.
    else:
        b=2. * np.arctan2 (tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs (y - c) > EPS):

        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
              d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180./np.pi
    glat2 *= 180./np.pi
    baz *= 180./np.pi

    return (glon2, glat2, baz)

