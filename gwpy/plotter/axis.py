# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides extensions for the core plotter relating to
axis objects.
"""

from gwpy import time

from . import ticks
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

TIME_FORMATS = time.TIME_FORMATS.keys()


def set_axis_format(axis, format_, **kwargs):
    """Set the tick formatting and location for the given axis to
    the specified format
    """
    if format_ in TIME_FORMATS:
        set_axis_time_format(axis, format_, **kwargs)
    else:
        raise NotImplementedError("Axis format '%s' has not been implemented"
                                  % format_)


def set_axis_time_format(axis, format_, **kwargs):
    locargs = dict()
    locargs['scale'] = kwargs.get('scale', None)
    locargs['epoch'] = kwargs.get('epoch', None)
    locator = ticks.AutoTimeLocator(**locargs)
    axis.set_major_locator(locator)
    formatter = ticks.TimeFormatter(format=format_, **kwargs)
    axis.set_major_formatter(formatter)
