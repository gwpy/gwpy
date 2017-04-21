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
from operator import itemgetter

from ...utils.compat import OrderedDict

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
    except KeyError as e:
        e.args = ("Unknown FFT method scaling: %r" % scaling,)
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
    except KeyError as e:
        e.args = ('Unknown FFT scaling: %r' % scaling,)
        raise

    # find method
    name = name.lower().replace('-', '_')
    try:
        return methods[name]
    except KeyError as e:
        e.args = ("No FFT method (scaling=%r) registered with name %r"
                  % (scaling, name),)
        raise


def update_doc(obj, scaling='density'):
    """Update the docstring of ``obj`` to reference available FFT methods
    """
    header = 'The available methods are:\n\n'
    doc = obj.__doc__

    # work out indent
    try:
        for line in doc.splitlines()[1:]:
            if line:
                break
    except AttributeError:
        line = ''
    indent = ' ' * (len(line) - len(line.lstrip(' ')))

    # strip out existing methods table
    try:
        maindoc, _ = doc.split(header, 1)
    except AttributeError:  # None
        maindoc = ''
    except ValueError:
        maindoc = doc

    # build table of methods
    from astropy.table import Table
    rows = []
    for method in METHODS[scaling]:
        f = METHODS[scaling][method]
        rows.append((method, '`%s.%s`' % (f.__module__, f.__name__)))
    if rows:
        rows = list(zip(*sorted(rows, key=itemgetter(1, 0))))
    methodtable = Table(rows, names=('Method name', 'Function'))
    newdoc = methodtable.pformat(max_lines=-1, max_width=80)
    tablehead = re.sub('-', '=', newdoc[1])
    newdoc[1] = tablehead
    newdoc.insert(0, tablehead)
    newdoc.append(tablehead)
    newdoc.extend(['', 'See :ref:`gwpy-signal-fft` for more details'])

    # re-write docstring
    doc = '%s\n%s' % (maindoc.rstrip('\n'), header)
    for line in newdoc:
        doc += '%s%s\n' % (indent, line)
    doc = doc.lstrip('\n')
    try:
        obj.__doc__ = doc
    except AttributeError:  # python 2.x
        obj.__func__.__doc__ = doc
