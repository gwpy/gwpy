#!/usr/bin/python

import os, sys, time, glob, math, matplotlib, random, string
import numpy as np

import gwpy.seismon.seismon_utils, gwpy.seismon.seismon_eqmon_plot

import gwpy.plotter

def earthquakes_training(params,attributeDics,data):
    """@run mla training

    @param params
        seismon params dictionary
    @param attributeDics
        list of eqmon structures
    @param data
        channel data dictionary
    """

    import pybrain.datasets, pybrain.tools.shortcuts, pybrain.supervised.trainers
    import pybrain.tools.customxml

    net = pybrain.tools.shortcuts.buildNetwork(3, 20, 1)
    ds = pybrain.datasets.SupervisedDataSet(3, 1)
    for key in data["channels"].iterkeys():
        channel_data = data["channels"][key]["earthquakes"]

        for magnitude,depth,distance,ampMax in zip(channel_data["magnitude"],channel_data["depth"],channel_data["distance"],channel_data["ampMax"]):
            ds.addSample((magnitude, depth, distance), (ampMax,))

    t = pybrain.supervised.trainers.BackpropTrainer(net, learningrate = 0.01, momentum = 0.5, verbose = False)
    t.trainOnDataset(ds, 1000)
    t.testOnData(verbose=False)

    trained = False
    acceptableError = 1.0e-7

    # train until acceptable error reached
    while trained == False :
        error = t.train()
        if error < acceptableError :
            trained = True

    inpts = []
    magnitudes = []
    targets = []
    estimates = []
    for inpt, target in ds:
        val = net.activate(inpt)
        inpts.append(inpt)
        magnitudes.append(inpt[0])
        targets.append(target)
        estimates.append(val)

    magnitudes = np.array(magnitudes)
    estimates = np.array(estimates)
    targets = np.array(targets)

    trainingDirectory = params["path"] + "/earthquakes_training"
    gwpy.seismon.seismon_utils.mkdir(trainingDirectory)
    trainingFile = os.path.join(trainingDirectory,"training.xml")
    pybrain.tools.customxml.networkwriter.NetworkWriter.writeToFile(net,trainingFile)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/earthquakes_training"
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"training.png")

        plot = gwpy.plotter.Plot(auto_refresh=True,figsize=[14,8])
        plot.add_scatter(magnitudes,1e6 * targets,color='b',label="true")
        plot.add_scatter(magnitudes,1e6 * estimates,color='g',label="predicted")
        plot.xlabel = 'Magnitude'
        plot.ylabel = r"Velocity [$\mu$m/s]"
        plot.axes.set_yscale("log")
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()

def earthquakes_testing(params,attributeDics,data):
    """@run mla testing

    @param params
        seismon params dictionary
    @param attributeDics
        list of eqmon structures
    @param data
        channel data dictionary
    """

    import pybrain.datasets, pybrain.tools.shortcuts, pybrain.supervised.trainers
    import pybrain.tools.customxml

    net = pybrain.tools.customxml.networkreader.NetworkReader.readFrom(params["earthquakesTrainingFile"])
    ds = pybrain.datasets.SupervisedDataSet(3, 1)
    for key in data["channels"].iterkeys():
        channel_data = data["channels"][key]["earthquakes"]

        for magnitude,depth,distance,ampMax in zip(channel_data["magnitude"],channel_data["depth"],channel_data["distance"],channel_data["ampMax"]):
            ds.addSample((magnitude, depth, distance), (ampMax,))

    inpts = []
    magnitudes = []
    targets = []
    estimates = []
    for inpt, target in ds:
        val = net.activate(inpt)
        inpts.append(inpt)
        magnitudes.append(inpt[0])
        targets.append(target)
        estimates.append(val)

    magnitudes = np.array(magnitudes)
    estimates = np.array(estimates)
    targets = np.array(targets)

    if params["doPlots"]:

        plotDirectory = params["path"] + "/earthquakes_testing"
        gwpy.seismon.seismon_utils.mkdir(plotDirectory)

        pngFile = os.path.join(plotDirectory,"testing.png")

        plot = gwpy.plotter.Plot(auto_refresh=True,figsize=[14,8])
        plot.add_scatter(magnitudes,1e6 * targets,color='b',label="true")
        plot.add_scatter(magnitudes,1e6 * estimates,color='g',label="predicted")
        plot.xlabel = 'Magnitude'
        plot.ylabel = r"Velocity [$\mu$m/s]"
        plot.axes.set_yscale("log")
        plot.add_legend(loc=1,prop={'size':10})
        plot.save(pngFile,dpi=200)
        plot.close()
