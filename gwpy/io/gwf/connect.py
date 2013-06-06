# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read GWF files into arrays
"""

from astropy.io import registry
from astropy import units

from pylal.Fr import *
from ...data import TimeSeries
from ...detector import Channel
from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_gwf(filepath, channel, epoch=None, duration=None, verbose=False):
    if isinstance(filepath, file):
        filepath = filepath.name
    if epoch is None:
        epoch = -1
    if duration is None:
        duration = -1
    data, epoch, epoch_offset, dt, x_unit, y_unit = frgetvect1d(
            filepath, channel, start=epoch, span=duration, verbose=verbose)
    epoch += epoch_offset
    unit = units.Unit(y_unit)
    channel = Channel(channel, sample_rate = 1/float(dt))
    return TimeSeries(data, channel=channel, epoch=epoch, unit=unit)


def identify_gwf(*args, **kwargs):
    filename = args[1][0]
    if isinstance(filename, file):
        filename = filename.name
    if not isinstance(filename, basestring) or not filename.endswith('gwf'):
        return False
    else:
        return True

registry.register_reader("gwf", TimeSeries, read_gwf, force=True)
registry.register_identifier("gwf", TimeSeries, identify_gwf)

