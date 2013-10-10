# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides plotting utilities for visualising GW data

The standard data types (`TimeSeries`, `Table`, `DataQualityFlag`) can
all be easily visualised using the relevant plotting objects, with 
many configurable parameters both interactive, and in saving to disk.
"""

import os
import matplotlib
try:
    os.environ['DISPLAY']
except KeyError:
    matplotlib.use('agg', warn=False)

from matplotlib import pyplot
rcParams = matplotlib.rcParams

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .tex import USE_TEX

from .timeseries import *
from .spectrogram import *
from .spectrum import *
from .segments import *
from .core import Plot

GWPY_PLOT_PARAMS = {
    "axes.grid": True,
    "axes.axisbelow": False,
    "axes.labelsize": 22,
    'figure.subplot.bottom': 0.13,
    'figure.subplot.left': 0.15,
    'figure.subplot.right': 0.88,
    'figure.subplot.top': 0.88,
    "image.aspect": 'auto',
    "image.interpolation": 'nearest',
    "image.origin": 'lower',
    "xtick.labelsize": 20,
    "ytick.labelsize": 20}
if rcParams['text.usetex'] or USE_TEX:
    GWPY_PLOT_PARAMS.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern"]})
rcParams.update(GWPY_PLOT_PARAMS)


def figure(*args, **kwargs):
    kwargs.setdefault('FigureClass', Plot)
    return pyplot.figure(*args, **kwargs)
figure.__doc__ = pyplot.figure.__doc__
