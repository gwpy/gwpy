# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read GWF files into arrays
"""

from astropy.io import registry
from astropy import units

from lalframe import frread

from ...data import TimeSeries
from ...detector import Channel
from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_gwf(filepath, channel, epoch=None, duration=None, verbose=False):
    if isinstance(filepath, file):
        filepath = filepath.name
    lalts = frread.read_timeseries(filepath, channel, start=epoch,
                                   duration=duration, verbose=verbose)
    return TimeSeries.from_lal(lalts)


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

