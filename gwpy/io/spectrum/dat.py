
"""Read a Spectrum from a dat file

These files must be in two-colum (frequency, amplitude) format
"""

import numpy
from astropy.io import registry

from ...spectrum.core import Spectrum
from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_dat(filepath, fcol=0, ampcol=1, **kwargs):
    """Read a `Spectrum` from a txt file
    """
    frequency, amplitude = numpy.loadtxt(filepath, usecols=[fcol, ampcol],
                                         unpack=True)
    return Spectrum(amplitude, frequencies=frequency, **kwargs)


def identify_dat(*args, **kwargs):
    """Identify the given file as a dat file, rather than anything else

    Returns
    -------
    True
        if the filename endswith .txt or .dat
    False
        otherwise
    """
    filename = args[1][0]
    if not isinstance(filename, basestring):
        filename = filename.name
    if filename.endswith('txt') or filename.endswith('dat'):
        return True
    return False


# register this file-reader with the Spectrum class
registry.register_reader('dat', Spectrum, read_dat, force=True)
registry.register_identifier('dat', Spectrum, identify_dat)
