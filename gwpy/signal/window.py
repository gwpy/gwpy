# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Utilities for signal-processing with windows
"""

import numpy

from math import ceil

try:
    from scipy.signal.windows._windows import _win_equiv as WINDOWS
except ImportError:  # scipy < 1.8.0
    from scipy.signal.windows.windows import _win_equiv as WINDOWS

from scipy.special import expit

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def canonical_name(name):
    """Find the canonical name for the given window in scipy.signal

    Parameters
    ----------
    name : `str`
        the name of the window you want

    Returns
    -------
    realname : `str`
        the name of the window as implemented in `scipy.signal.window`

    Raises
    -------
    ValueError
        if ``name`` cannot be resolved to a window function in `scipy.signal`

    Examples
    --------
    >>> from gwpy.signal.window import canonical_name
    >>> canonical_name('hann')
    'hann'
    >>> canonical_name('ksr')
    'kaiser'
    """
    if name.lower() == 'planck':  # make sure to handle the Planck window
        return 'planck'
    try:  # use equivalence introduced in scipy 0.16.0
        # pylint: disable=protected-access
        return WINDOWS[name.lower()].__name__
    except KeyError:  # no match
        raise ValueError('no window function in scipy.signal equivalent to %r'
                         % name,)


# -- recommended overlap ------------------------------------------------------
# source: http://edoc.mpg.de/395068

ROV = {
    'boxcar': 0,
    'bartlett': .5,
    'barthann': .5,
    'blackmanharris': .661,
    'flattop': .8,
    'hann': .5,
    'hamming': .5,
    'nuttall': .656,
    'triang': .5
}


def recommended_overlap(name, nfft=None):
    """Returns the recommended fractional overlap for the given window

    If ``nfft`` is given, the return is in samples

    Parameters
    ----------
    name : `str`
        the name of the window you are using

    nfft : `int`, optional
        the length of the window

    Returns
    -------
    rov : `float`, `int`
        the recommended overlap (ROV) for the given window, in samples if
        ``nfft` is given (`int`), otherwise fractional (`float`)

    Examples
    --------
    >>> from gwpy.signal.window import recommended_overlap
    >>> recommended_overlap('hann')
    0.5
    >>> recommended_overlap('blackmanharris', nfft=128)
    85
    """
    try:
        name = canonical_name(name)
    except KeyError as exc:
        raise ValueError(str(exc))
    try:
        rov = ROV[name]
    except KeyError:
        raise ValueError("no recommended overlap for %r window" % name)
    if nfft:
        return int(ceil(nfft * rov))
    return rov


# -- Planck taper window ------------------------------------------------------
# source: https://arxiv.org/abs/1003.2939


def planck(N, nleft=0, nright=0):
    """Return a Planck taper window.

    Parameters
    ----------
    N : `int`
        Number of samples in the output window

    nleft : `int`, optional
        Number of samples to taper on the left, should be less than `N/2`

    nright : `int`, optional
        Number of samples to taper on the right, should be less than `N/2`

    Returns
    -------
    w : `ndarray`
        The window, with the maximum value normalized to 1 and at least one
        end tapered smoothly to 0.

    Examples
    --------
    To taper 0.1 seconds on both ends of one second of data sampled at 2048 Hz:

    >>> from gwpy.signal.window import planck
    >>> w = planck(2048, nleft=205, nright=205)

    References
    ----------
    .. [1] McKechan, D.J.A., Robinson, C., and Sathyaprakash, B.S. (April
           2010). "A tapering window for time-domain templates and simulated
           signals in the detection of gravitational waves from coalescing
           compact binaries". Classical and Quantum Gravity 27 (8).
           :doi:`10.1088/0264-9381/27/8/084020`

    .. [2] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
    """
    # construct a Planck taper window
    w = numpy.ones(N)
    if nleft:
        w[0] *= 0
        zleft = numpy.array([nleft * (1./k + 1./(k-nleft))
                            for k in range(1, nleft)])
        w[1:nleft] *= expit(-zleft)
    if nright:
        w[N-1] *= 0
        zright = numpy.array([-nright * (1./(k-nright) + 1./k)
                             for k in range(1, nright)])
        w[N-nright:N-1] *= expit(-zright)
    return w
