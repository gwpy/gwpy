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

"""A collection of average power spectral density calculation routines

This module defines a set of builtin methods with which to calculate
an average power spectral density. Users can define their own through
the registry method `register_method`.
"""

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

spectrum_methods = OrderedDict()
density_methods = OrderedDict()


def register_method(func, name=None, force=False, scaling='density'):
    """Register a method of calculating an average spectrogram.

    Parameters
    ----------
    func : `callable`
        function to execute
    name : `str`, optional
        name of the method, defaults to ``func.__name__``
    """
    from ..timeseries.core import TimeSeries
    # get name and format doc addition
    if name is None:
        name = func.__name__
    path = '.'.join([func.__module__, func.__name__])
    doc = '        - %r: :meth:`%s`' % (name, path)

    def append_doc(obj, newdoc):
        lines = obj.__doc__.splitlines()
        if not lines[-1].strip(' '):
            suffix = lines.pop(-1)
        else:
            suffix = ''
        obj.__func__.__doc__ = '\n'.join(lines + [newdoc, suffix])

    # determine spectrum method type, and append doc
    if scaling == 'density':
        methods = density_methods
        append_doc(TimeSeries.psd, doc)
        append_doc(TimeSeries.asd, doc)
    elif scaling == 'spectrum':
        methods = spectrum_methods
        append_doc(TimeSeries.power_spectrum, doc)
    else:
        raise ValueError("Unknown Spectrum type: %r" % scaling)
    # record method in registry
    if name in methods and not force:
        raise KeyError("'%s' already registered, use force=True to override."
                       % name)
    methods[name] = func


def get_method(name, scaling='density'):
    """Return the Specrum generator registered with the given name.
    """
    if scaling == 'density':
        try:
            return density_methods[name]
        except KeyError:
            raise ValueError("No PSD method registered with name %r" % name)
    elif scaling == 'spectrum':
        try:
            return spectrum_methods[name]
        except KeyError:
            raise ValueError("No power spectrum method registered with "
                             "name %r" % name)
    else:
        raise ValueError("Unknown Spectrum type: %r" % scaling)
