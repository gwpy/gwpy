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

"""Registry for FFT averaging methods
"""

import re
from collections import OrderedDict

from six import string_types

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# registry dict for FFT averaging methods
METHODS = {
    'spectrum': OrderedDict(),
    'density': OrderedDict(),
    'other': OrderedDict(),
}


def register_method(func, name=None, force=False, scaling='density'):
    """Register a method of calculating an average spectrogram.

    Parameters
    ----------
    func : `callable`
        function to execute

    name : `str`, optional
        name of the method, defaults to ``func.__name__``
    """
    if name is None:
        name = func.__name__
    name = name.lower().replace('-', '_')

    # determine spectrum method type, and append doc
    try:
        methods = METHODS[scaling]
    except KeyError as exc:
        exc.args = ("Unknown FFT method scaling: %r" % scaling,)
        raise

    # if name already registered, don't override unless forced
    if name in methods and not force:
        raise KeyError("'%s' already registered, use `force=True` to override"
                       % name)
    methods[name] = func


def get_method(name, scaling='density'):
    """Return the Specrum generator registered with the given name.
    """
    # find right group
    try:
        methods = METHODS[scaling]
    except KeyError as exc:
        exc.args = ('Unknown FFT scaling: %r' % scaling,)
        raise

    # find method
    name = name.lower().replace('-', '_')
    try:
        return methods[name]
    except KeyError as exc:
        exc.args = ("no FFT method (scaling=%r) registered with name %r"
                    % (scaling, name),)
        raise


def update_doc(obj, scaling='density'):
    """Update the docstring of ``obj`` to reference available FFT methods
    """
    header = 'The available methods are:'

    # remove the old format list
    lines = obj.__doc__.splitlines()
    try:
        pos = [i for i, line in enumerate(lines) if header in line][0]
    except IndexError:
        pass
    else:
        lines = lines[:pos]

    # work out the indentation
    matches = [re.search(r'(\S)', line) for line in lines[1:]]
    indent = min(match.start() for match in matches if match)

    # build table of methods
    from astropy.table import Table
    rows = []
    for method in METHODS[scaling]:
        func = METHODS[scaling][method]
        rows.append((method, '`%s.%s`' % (func.__module__, func.__name__)))
    format_str = Table(rows=rows, names=['Method name', 'Function']).pformat(
        max_lines=-1, max_width=80, align=('>', '<'))
    format_str[1] = format_str[1].replace('-', '=')
    format_str.insert(0, format_str[1])
    format_str.append(format_str[0])
    format_str.extend(['', 'See :ref:`gwpy-signal-fft` for more details'])

    lines.extend([' ' * indent + line for line in [header, ''] + format_str])

    # and overwrite the docstring
    obj.__doc__ = '\n'.join(lines)
