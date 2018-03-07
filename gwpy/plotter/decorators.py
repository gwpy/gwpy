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

import warnings
from functools import wraps

from matplotlib.figure import Figure
from matplotlib.axes import Axes

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def auto_refresh(func):
    """Decorate `func` to refresh the containing figure when complete
    """
    @wraps(func)
    def wrapper(artist, *args, **kwargs):
        """Call the method, and refresh the figure on exit
        """
        refresh = kwargs.pop('refresh', False)
        try:
            return func(artist, *args, **kwargs)
        finally:
            try:
                refresh |= artist.figure.get_auto_refresh()
            except AttributeError:
                pass
            else:
                if not refresh:
                    pass
                elif isinstance(artist, Axes):
                    artist.figure.refresh()
                elif isinstance(artist, Figure):
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
        warnings.warn('Plot.{0}() is been deprecated, and will be removed in '
                      'an upcoming release; please update your code to call '
                      'Axes.{0}() directly'.format(func.__name__),
                      DeprecationWarning)
        axes = [ax for ax in figure.axes if ax not in figure._coloraxes]
        if not axes:
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
