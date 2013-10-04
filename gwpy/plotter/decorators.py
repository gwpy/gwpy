#!/usr/bin/env python

# Copyright (C) 2012 Duncan M. Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Decorators
"""

import threading

from matplotlib.figure import Figure

from .decorator import decorator

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

mydata = threading.local()


def auto_refresh(f):
    return decorator(_auto_refresh, f)


def _auto_refresh(f, *args, **kwargs):
    if 'refresh' in kwargs:
        refresh = kwargs.pop('refresh')
    else:
        refresh = True
    # The following is necessary rather than using mydata.nesting = 0 at the
    # start of the file, because doing the latter caused issues with the Django
    # development server.
    mydata.nesting = getattr(mydata, 'nesting', 0) + 1
    try:
        return f(*args, **kwargs)
    finally:
        mydata.nesting -= 1
        if hasattr(args[0], 'figure'):
            if refresh and mydata.nesting == 0 and args[0].figure._auto_refresh:
                args[0].figure.canvas.draw()
        elif isinstance(args[0], Figure):
            if refresh and mydata.nesting == 0 and args[0]._auto_refresh:
                args[0].canvas.draw()
