# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Utilities for building plots in LAL.
"""

import os
if not os.getenv('DISPLAY', None):
    import matplotlib
    matplotlib.use('agg', warn=False)

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from .core import *
from .ticks import *
from .histogram import *
from .table import *
from .series import *
from .gwf import *

LAL_PLOT_PARAMS = {
    "text.usetex": True,
    "axes.grid": True,
    "axes.axisbelow": False,
    "axes.labelsize": 22,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "image.aspect": 'auto',
    "image.interpolation": 'nearest',
    "image.origin": 'lower',
    "xtick.labelsize": 20,
    "ytick.labelsize": 20}
mpl.rcParams.update(LAL_PLOT_PARAMS)

