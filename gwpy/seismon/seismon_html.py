#!/usr/bin/python

"""
%prog

Michael Coughlin (coughlim@carleton.edu)

This program checks for earthquakes.

"""

import os, time, glob, pickle
import gwpy.seismon.seismon_utils

__author__ = "Michael Coughlin <coughlim@carleton.edu>"
__date__ = "2012/2/7"
__version__ = "0.1"

# =============================================================================
#
#                               DEFINITIONS
#
# =============================================================================

def summary_page(params):
    """@html summary page.

    @param params
        seismon params dictionary
    @param channels
        lists of seismon channel structures
    """

    ifo = gwpy.seismon.seismon_utils.getIfo(params)

    title = "Seismon Summary Page for %s (%d)"%(params["dateString"],params["gpsStart"])

    contents=["""
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
    <html>
    <head>
    <title>%s</title>
    <link rel="stylesheet" type="text/css" href="../style/main.css">
    <script src="../style/sorttable.js"></script>
    </head>
    <body>
    <table style="text-align: left; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    <tr>
    <td style="text-align: center; vertical-align: top; background-color:SpringGreen;"><big><big><big><big><span style="font-weight: bold;">%s</span></big></big></big></big>
    </td>
    </tr>
    </tbody>
    </table>
    <br>
    <br>

    """%(title,title)]

    table = ["""
    <table class="sortable" id="sort" style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto; background-color: white;" border="1" cellpadding="2" cellspacing="2">
    """]

    table = ["""
    <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    """]

    count = 0
    for channel in params["channels"]:

        textLocation = params["path"] + "/" + channel.station_underscore

        file = os.path.join(textLocation,"sig.txt")
        if not os.path.isfile(file):
            continue

        lines = [line.strip() for line in open(file)]

        sigDict = []
        for line in lines:
            sig = {}
            lineSplit = line.split(" ")
            sig["flow"] = float(lineSplit[0])
            sig["fhigh"] = float(lineSplit[1])
            sig["meanPSD"] = float(lineSplit[2])
            sig["sig"] = float(lineSplit[3])
            sig["bgcolor"] = lineSplit[4]
            sigDict.append(sig)

        if count == 0:
            table.append("""<tr><td>Frequency Band (Hz)</td>""")
            for sig in sigDict:
                table.append("""<td>%.4f - %.4f</td>"""%(sig["flow"],sig["fhigh"]))
            table.append("""</tr>""")
        table.append("""<tr><td><a href= "./%s/psd.html">%s</a></td>"""%(channel.station_underscore,channel.station))
        for sig in sigDict:
            table.append("""<td style="background-color: %s">%.4f</td>"""%(sig["bgcolor"],sig["sig"]))
        table.append("""</tr>""")

        count = count+1

    table.append("</tbody></table><br><br>")

    # add tables and list
    contents.append("".join(table))

    earthquakesDirectory = os.path.join(params["path"],"earthquakes")
    earthquakesXMLFile = os.path.join(earthquakesDirectory,"earthquakes.xml")

    if os.path.isfile(earthquakesXMLFile):
        attributeDics = gwpy.seismon.seismon_utils.read_eqmons(earthquakesXMLFile)

        table = ["""
        <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
        <tbody>
        <tr><td>ID</td><td>Magnitude</td><td>Latitude</td><td>Longitude</td><td>Depth</td><td>R-arrival</td><td>R-amp</td></tr>"""]

        for attributeDic in attributeDics:

            traveltimes = attributeDic["traveltimes"][ifo]

            gpsStart = max(traveltimes["RthreePointFivetimes"]) - 200
            gpsEnd = max(traveltimes["RthreePointFivetimes"]) + 200

            check_intersect = (gpsEnd >= params["gpsStart"]) and (params["gpsEnd"] >= gpsStart)
            if check_intersect:
                table.append('<tr><td><a href="%s/%s">%s</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.1f</td><td>%.2e</td></tr>'%("http://earthquake.usgs.gov/earthquakes/eventpage",attributeDic["eventName"],attributeDic["eventName"],attributeDic["Magnitude"],attributeDic["Latitude"],attributeDic["Longitude"],attributeDic["Depth"],max(traveltimes["RthreePointFivetimes"]),traveltimes["Rfamp"][0]))

        table.append("</tbody></table><br><br>")
        contents.append("".join(table))


    table = ["""
    <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    <tr>
    <td style="vertical-align: top;">Recent Earthquake Magnitudes (t = 0 is current time)</td>
    <td style="vertical-align: top;">Time until earthquake phase arrivals (p = pressure, s = shear, r = surface)</td>
    </tr>
    <tr>
    <td style="vertical-align: top;"><a href="./earthquakes/magnitudes.png"><img alt="" src="./earthquakes/magnitudes.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    <td style="vertical-align: top;"><a href="./earthquakes/traveltimes%s.png"><img alt="" src="./earthquakes/traveltimes%s.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    </tr>
    </tbody></table><br><br>
    """%(params["ifo"],params["ifo"])]
    contents.append("".join(table))

    table = ["""
    <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    <tr>
    <td><a href= "https://ldas-jobs.ligo.caltech.edu/~mcoughlin/EQMon/EQMon.pdf">README</a></td>
    </tr>
    </tbody></table><br><br>
    """]
    contents.append("".join(table))

    ################################# closing ##################################
    user=os.environ['USER']
    curTime=time.strftime('%m-%d-%Y %H:%M:%S',time.localtime())
    contents.append("""
    <small>
    This page was created by user %s on %s
    </small>
    </body>
    </html>
    """%(user,curTime))

    page = ''.join(contents)

    return page

def seismon_page(channel,textLocation):
    """@html channel page.

    @param channel
        seismon channel structure
    @param textLocation
        location of seismon text files
    """

    ############################## header ######################################
    title = "Seismon page for %s"%channel.station

    contents=["""
    <html>
    <head>
    <meta content="text/html; charset=ISO-8859-1"
    http-equiv="content-type">
    <title>%s</title>
    </head>
    <body>
    <table style="text-align: left; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    <tr>
    <td style="text-align: center; vertical-align: top; background-color:SpringGreen;"><big><big><big><big><span style="font-weight: bold;">%s</span></big></big></big></big>
    </td>
    </tr>
    </tbody>
    </table>
    <br>
    <br>

    """%(title,title)]

    table = ["""
    <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    """]

    file = os.path.join(textLocation,"sig.txt")
    lines = [line.strip() for line in open(file)]  

    sigDict = []
    for line in lines:
        sig = {}
        lineSplit = line.split(" ")
        sig["flow"] = float(lineSplit[0])
        sig["fhigh"] = float(lineSplit[1])
        sig["meanPSD"] = float(lineSplit[2])
        sig["sig"] = float(lineSplit[3])
        sig["bgcolor"] = lineSplit[4]
        sigDict.append(sig)

    table.append("""<tr><td>Frequency Band (Hz)</td>""")
    for sig in sigDict: 
        table.append("""<td>%.4f - %.4f</td>"""%(sig["flow"],sig["fhigh"]))
    table.append("""</tr>""")
    table.append("""<tr><td>Mean ASD ((m/s)/rtHz)</td>""")
    for sig in sigDict:
        table.append("""<td>%.4e</td>"""%(sig["meanPSD"]))
    table.append("""</tr>""")
    table.append("""<tr><td>ASD Significance (red = loud, green = average, blue = quiet)</td>""")
    for sig in sigDict:
        table.append("""<td style="background-color: %s">%.4f</td>"""%(sig["bgcolor"],sig["sig"]))
    table.append("""</tr>""")

    table.append("</tbody></table><br><br>")

    # add tables and list
    contents.append("".join(table))

    table = ["""
    <table style="text-align: center; width: 1260px; height: 67px; margin-left:auto; margin-right: auto;" border="1" cellpadding="1" cellspacing="1">
    <tbody>
    <tr>
    <td style="vertical-align: top;">Frequency vs. Time</td>
    <td style="vertical-align: top;">Glitch-gram (Omicron)</td>
    </tr>
    <tr>
    <td style="vertical-align: top;"><a href="./tf.png"><img alt="" src="./tf.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    <td style="vertical-align: top;"><a href="./omicron.png"><img alt="" src="./omicron.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    </tr>

    <tr>
    <td style="vertical-align: top;">ASD (m) (including 10, 50, and 90 percentiles)</td>
    <td style="vertical-align: top;">Timeseries (raw and lowpassed at 1 Hz)</td>
    </tr>
    <tr>
    <td style="vertical-align: top;"><a href="./disp.png"><img alt="" src="./disp.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    <td style="vertical-align: top;"><a href="./timeseries.png"><img alt="" src="./timeseries.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    </tr>

    <tr>
    <td style="vertical-align: top;">ASD (m/s) (including 10, 50, and 90 percentiles)</td>
    <td style="vertical-align: top;">Mean ASD vs. time</td>
    </tr>
    <tr>
    <td style="vertical-align: top;"><a href="./psd.png"><img alt="" src="./psd.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    <td style="vertical-align: top;"><a href="./bands.png"><img alt="" src="./bands.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    </tr>

    <tr>
    <td style="vertical-align: top;">Spectral Variation (including 10, 50, and 90 percentiles in white, color scale indicates percentage of time in a particular bin)</td>
    </tr>
    <tr>
    <td style="vertical-align: top;"><a href="./specvar.png"><img alt="" src="./specvar.png" style="border: 0px solid ; width: 630px; height: 432px;"></a><br></td>
    </tr>
    </tbody></table><br><br>
    """]
    contents.append("".join(table))

    ################################# closing ##################################
    user=os.environ['USER']
    curTime=time.strftime('%m-%d-%Y %H:%M:%S',time.localtime())
    contents.append("""
    <small>
    This page was created by user %s on %s
    </small>
    </body>
    </html>
    """%(user,curTime))

    page = ''.join(contents)

    return page

