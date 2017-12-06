# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

from math import ceil

from scipy.signal import windows as scipy_windows

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
    >>> canonical_name('hanning')
    'hann'
    >>> canonical_name('ksr')
    'kaiser'
    """
    try:  # use equivalence introduced in scipy 0.16.0
        # pylint: disable=protected-access
        return scipy_windows._win_equiv[name.lower()].__name__
    except AttributeError:  # old scipy
        try:
            return getattr(scipy_windows, name.lower()).__name__
        except AttributeError:  # no match
            pass  # raise later
    except KeyError:  # no match
        pass  # raise later

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
