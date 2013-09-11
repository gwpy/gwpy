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
    IS_INTERACTIVE = False
else:
    IS_INTERACTIVE = matplotlib.is_interactive()

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .core import *
from .ticks import *
from .histogram import *
from .series import *
from .timeseries import *
from .table import *
from .gwf import *

USE_TEX = os.system('which pdflatex > %s 2>&1' % os.devnull) == 0

GWPY_PLOT_PARAMS = {
    "axes.grid": True,
    "axes.axisbelow": False,
    "axes.labelsize": 22,
    "image.aspect": 'auto',
    "image.interpolation": 'nearest',
    "image.origin": 'lower',
    "xtick.labelsize": 20,
    "ytick.labelsize": 20}
if USE_TEX:
    GWPY_PLOT_PARAMS.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Computer Modern"]})
mpl.rcParams.update(GWPY_PLOT_PARAMS)
