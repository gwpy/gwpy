# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Core library for Window class
"""

import numpy

from scipy import signal

from ..data import Series

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def get_window(window, length, *args, **kwargs):
    """Generate a new `Window`

    Parameters
    ----------
    window : `str`
        name of the desired window type, for a full list of available
        types, see :func:`scipy.signal.get_window`
    length : `int`
        length of the desired window
    **kwargs
        other parameters required for a specific window type
    """
    if isinstance(window, (tuple, list)):
        args = window[1:] + args
        window = window[0]
    if window in ['blackman', 'black', 'blk']:
        WindowClass = BlackmanWindow
    elif window in ['triangle', 'triang', 'tri']:
        WindowClass = TriangularWindow
    elif window in ['hamming', 'hamm', 'ham']:
        WindowClass = HammingWindow
    elif window in ['bartlett', 'bart', 'brt']:
        WindowClass = BartlettWindow
    elif window in ['hanning', 'hann', 'han']:
        WindowClass = HanningWindow
    elif window in ['blackmanharris', 'blackharr', 'bkh']:
        WindowClass = BlackmanHarrisWindow
    elif window in ['parzen', 'parz', 'par']:
        WindowClass = ParzenWindow
    elif window in ['bohman', 'bman', 'bmn']:
        WindowClass = BohmanWindow
    elif window in ['nuttall', 'nutl', 'nut']:
        WindowClass = NuttallWindow
    elif window in ['barthann', 'brthan', 'bth']:
        WindowClass = BarthannWindow
    elif window in ['flattop', 'flat', 'flt']:
        WindowClass = FlatTopWindow
    elif window in ['kaiser', 'ksr']:
        WindowClass = KaiserWindow
    elif window in ['gaussian', 'gauss', 'gss']:
        WindowClass = GaussianWindow
    elif window in ['general gaussian', 'general_gaussian',
                    'general gauss', 'general_gauss', 'ggs']:
        WindowClass = GeneralGaussianWindow
    elif window in ['boxcar', 'box', 'ones', 'rect', 'rectangular']:
        WindowClass = BoxCarWindow
    elif window in ['slepian', 'slep', 'optimal', 'dpss', 'dss']:
        WindowClass = SlepianWindow
    elif window in ['chebwin', 'cheb']:
        WindowClass = DolphChebyshevWindow
    else:
        raise ValueError("Unknown window type.")
    return WindowClass(length, *args, **kwargs)


class Window(Series):
    """Representaion of a time-domain window

    This class is meant to be subclassed.
    """
    _metadata_slots = ['name', 'symmetric']

    def __getattr__(self, attr):
        if attr in self._metadata_slots:
            return self.metadata[attr]
        else:
            return self.__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr in self._metadata_slots:
            self.metadata[attr] = value
        else:
            super(Window, self).__setattr__(attr, value)

    def plot(self, **kwargs):
        from ..plotter import Plot
        out.add_line(numpy.arange(self.size), self.data)
        out = Plot(**kwargs)
        out.xlabel = 'Sample'
        out.ylabel = 'Amplitude'
        out.title = '%s window' % self.__class__.__name__[:-6]
        return out

    @classmethod
    def from_lal(self, lalwindow):
        """Convert an XLAL Window into GWpy `Window` format
        """

    def to_lal(self, dtype='real8'):
        """Convert this `Window` into an XLAL format
        """
        from lal import lal
        if dtype.lower() == 'real4':
            seq = lal.CreateREAL4Sequence(int(self.size))
            seq.data = numpy.array(self, dtype=seq.data.dtype, copy=True)
            return lal.CreateREAL4WindowFromSequence(seq)
        else:
            seq = lal.CreateREAL8Sequence(int(self.size))
            seq.data = numpy.array(self, dtype=seq.data.dtype, copy=True)
            lalwindow = lal.CreateREAL8WindowFromSequence(seq)
            return lalwindow


class SimpleWindow(Window):
    """Representaion of a simple time-domain window

    This meta-class makes it easier to define the simple windows that
    require only a length input.
    """
    _name = None
    _metadata_slots = ['name', 'symmetric']
    def __new__(cls, length, symmetric=True):
        """Generate a new `Window`
        """
        data = signal.get_window(cls._name, length, fftbins=not symmetric)
        new = super(SimpleWindow, cls).__new__(cls, data, name=cls._name,
                                               symmetric=symmetric)
        return new


class BarthannWindow(SimpleWindow):
    """Representation of a Barthann window

    For details see :func:`scipy.signal.barthann`
    """
    _name = 'barthann'


class BartlettWindow(SimpleWindow):
    """Representation of a Bartlett window

    For details see :func:`scipy.signal.bartlett`
    """
    _name = 'bartlett'


class BlackmanWindow(SimpleWindow):
    """Representation of a Blackman window

    For details see :func:`scipy.signal.blackman`
    """
    _name = 'blackman'


class BlackmanHarrisWindow(SimpleWindow):
    """Representation of a Blackman-Harris window

    For details see :func:`scipy.signal.blackmanharris`
    """
    _name = 'blackmanharris'


class BohmanWindow(SimpleWindow):
    """Representation of a Bohman window

    For details see :func:`scipy.signal.bohman`
    """
    _name = 'bohman'


class BoxCarWindow(SimpleWindow):
    """Representation of a BoxCar window

    For details see :func:`scipy.signal.boxcar`
    """
    _name = 'boxcar'


class DolphChebyshevWindow(Window):
    """Representation of a Dolph-Chebyshev window

    For details see :func:`scipy.signal.chebwin`
    """
    _name = 'chebwin'
    _metadata_slots = ['name', 'attenuation', 'symmetric']
    def __init__(self, length, attenuation, symmetric=True):
        """Generate a new `Window`
        """
        data = signal.get_window((self.name, attenuation), length,
                                 fftbins=not symmetric)
        return super(DolphChebyshevWindow, self).__init__(data, name=self.name,
                                                   attenuation=attenuation,
                                                   symmetric=symmetric)


class FlattopWindow(SimpleWindow):
    """Representation of a Flattop window

    For details see :func:`scipy.signal.flattop`
    """
    _name = 'flattop'


class GaussianWindow(Window):
    """Representation of a Gaussian window

    For details see :func:`scipy.signal.gaussian`
    """
    _name = 'gaussian'
    _metadata_slots = ['name', 'std', 'symmetric']
    def __init__(self, length, std, symmetric=True):
        """Generate a new `Window`
        """
        data = signal.get_window((self.name, std), length,
                                 fftbins=not symmetric)
        return super(GaussianWindow, self).__init__(data, name=self.name,
                                             std=std, symmetric=symmetric)


class GeneralGaussianWindow(Window):
    """Representation of a general Gaussian window

    For details see :func:`scipy.signal.general_gaussian`
    """
    _name = 'general_gaussian'
    _metadata_slots = ['name', 'p', 'sigma', 'symmetric']
    def __init__(self, length, p, sigma, symmetric=True):
        """Generate a new `Window`
        """
        data = signal.get_window((self.name, p, sigma), length,
                                 fftbins=not symmetric)
        return super(GeneralGaussianWindow, self).__init__(data, name=self.name, p=p,
                                                    sigma=sigma,
                                                    symmetric=symmetric)


class HammingWindow(SimpleWindow):
    """Representation of a Hamming window

    For details see :func:`scipy.signal.hamming`
    """
    _name = 'hamming'


class HannWindow(SimpleWindow):
    """Representation of a Hann window

    For details see :func:`scipy.signal.hann`
    """
    _name = 'hann'


class HanningWindow(SimpleWindow):
    """Representation of a Hanning window

    For details see :func:`scipy.signal.hanning`
    """
    _name = 'hanning'


class KaiserWindow(Window):
    """Representation of a Kaiser window

    For details see :func:`scipy.signal.kaiser`
    """
    _name = 'kaiser'
    _metadata_slots = ['name', 'beta', 'symmetric']
    def __new__(cls, length, beta=24, symmetric=True):
        data = signal.kaiser(length, beta, sym=symmetric)
        return super(KaiserWindow, cls).__new__(cls, data, name=cls._name,
                                                beta=beta, symmetric=symmetric)


class NuttallWindow(SimpleWindow):
    """Representation of a Nuttall window

    For details see :func:`scipy.signal.nuttall`
    """
    _name = 'nuttall'


class ParzenWindow(SimpleWindow):
    """Representation of a Parzen window

    For details see :func:`scipy.signal.parzen`
    """
    _name = 'parzen'


class SlepianWindow(Window):
    """Representation of a Slepian window

    For details see :func:`scipy.signal.slepian`
    """
    _name = 'slepian'
    _metadata_slots = ['name', 'width', 'symmetric']
    def __new__(cls, length, width, symmetric=True):
        data = signal.get_window((cls._name, width), length,
                                 fftbins=not symmetric)
        return super(SlepianWindow, cls).__new__(cls, data, name=cls._name, width=width,
                                          symmetric=symmetric)


class TriangularWindow(SimpleWindow):
    """Representation of a Triang window

    For details see :func:`scipy.signal.triang`
    """
    _name = 'triang'
