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
from functools import wraps

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

mydata = threading.local()


def auto_refresh(func):
    """Decorate `func` to refresh the containing figure when complete
    """
    @wraps(func)
    def wrapper(artist, *args, **kwargs):
        """Call the method, and refresh the figure on exit
        """
        refresh = kwargs.pop('refresh', True)
        mydata.nesting = getattr(mydata, 'nesting', 0) + 1
        try:
            return func(artist, *args, **kwargs)
        finally:
            mydata.nesting -= 1
            if isinstance(artist, Axes):
                if (refresh and mydata.nesting == 0 and
                        artist.figure.get_auto_refresh()):
                    artist.figure.refresh()
            elif isinstance(artist, Figure):
                if (refresh and mydata.nesting == 0 and
                        artist.get_auto_refresh()):
                    artist.refresh()
            else:
                raise TypeError("Cannot determine containing Figure for "
                                "auto_refresh() decorator")
    return wrapper


def axes_method(func):
    """Decorate `func` to call the same method of the contained `Axes`

    Raises
    ------
    RuntimeError
        if multiple `Axes` are found when the method is called.
        This method makes no attempt to decide which `Axes` to use
    """
    @wraps(func)
    def wrapper(figure, *args, **kwargs):
        """Find the relevant `Axes` and call the method
        """
        axes = [ax for ax in figure.axes if ax not in figure._coloraxes]
        if len(axes) == 0:
            raise RuntimeError("No axes found for which '%s' is applicable"
                               % func.__name__)
        if len(axes) != 1:
            raise RuntimeError("{0} only applicable for a Plot with a "
                               "single set of data axes. With multiple "
                               "data axes, you should access the {0} "
                               "method of the relevant Axes (stored in "
                               "``Plot.axes``) directly".format(func.__name__))
        axesf = getattr(axes[0], func.__name__)
        return axesf(*args, **kwargs)
    return wrapper
