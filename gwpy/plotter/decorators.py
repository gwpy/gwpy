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

"""Decorator for GWpy plotting
"""

import threading

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .decorator import decorator

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

mydata = threading.local()


@decorator
def auto_refresh(f, *args, **kwargs):
    refresh = kwargs.pop('refresh', True)
    # The following is necessary rather than using mydata.nesting = 0 at the
    # start of the file, because doing the latter caused issues with the Django
    # development server.
    mydata.nesting = getattr(mydata, 'nesting', 0) + 1
    try:
        return f(*args, **kwargs)
    finally:
        mydata.nesting -= 1
        if isinstance(args[0], Axes):
            if refresh and mydata.nesting == 0 and args[0].figure._auto_refresh:
                args[0].figure.refresh()
        elif isinstance(args[0], Figure):
            if refresh and mydata.nesting == 0 and args[0]._auto_refresh:
                args[0].refresh()

@decorator
def axes_method(f, *args, **kwargs):
    figure = args[0]
    axes = [ax for ax in figure.axes if ax not in figure._coloraxes]
    if len(axes) == 0:
        raise RuntimeError("No axes found for which '%s' is applicable"
                           % f.__name__)
    if len(axes) != 1:
        raise RuntimeError("{0} only applicable for a Plot with a single set "
                           "of data axes. With multiple data axes, you should "
                           "access the {0} method of the relevant Axes (stored "
                           "in ``Plot.axes``) directly".format(f.__name__))
    axesf = getattr(axes[0], f.__name__)
    return axesf(*args[1:], **kwargs)
