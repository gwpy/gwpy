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

"""Extension of the :class:`~matplotlib.axes.Axes` class with
user-friendly attributes
"""

from six import string_types

from matplotlib.axes import Axes as _Axes

from .decorators import auto_refresh
from . import (rcParams, tex)

from .. import version
__version__ = version.version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

class Axes(_Axes):
    """An extension of the core matplotlib :class:`~matplotlib.axes.Axes`.

    These custom `Axes` provide only some simpler attribute accessors.

    Notes
    -----
    A new set of `Axes` should be constructed via::

        >>> plot.add_subplots(111, projection='xxx')

    where plot is a :class:`~gwpy.plotter.Plot` figure, and ``'xxx'``
    is the name of the `Axes` you want to add.
    """
    def __init__(self, *args, **kwargs):
        super(Axes, self).__init__(*args, **kwargs)
        self.xaxis.labelpad = 10
    __init__.__doc__ = _Axes.__init__.__doc__

    # -----------------------------------------------
    # text properties

    # x-axis label
    @property
    def xlabel(self):
        """Label for the x-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.xaxis.label
    @xlabel.setter
    @auto_refresh
    def xlabel(self, text):
        if isinstance(text, string_types):
            self.set_xlabel(text)
        else:
            self.xaxis.label = text
    @xlabel.deleter
    @auto_refresh
    def xlabel(self):
        self.set_xlabel("")

    # y-axis label
    @property
    def ylabel(self):
        """Label for the y-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.yaxis.label
    @ylabel.setter
    @auto_refresh
    def ylabel(self, text):
        if isinstance(text, string_types):
            self.set_ylabel(text)
        else:
            self.yaxis.label = text
    @ylabel.deleter
    @auto_refresh
    def ylabel(self):
        self.set_ylabel("")

    # -----------------------------------------------
    # limit properties

    @property
    def xlim(self):
        """Limits for the x-axis

        :type: `tuple`
        """
        return self.get_xlim()
    @xlim.setter
    @auto_refresh
    def xlim(self, limits):
        self.set_xlim(*limits)
    @xlim.deleter
    @auto_refresh
    def xlim(self):
        self.relim()
        self.autoscale_view(scalex=True, scaley=False)

    @property
    def ylim(self):
        """Limits for the y-axis

        :type: `tuple`
        """
        return self.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, limits):
        self.set_ylim(*limits)
    @ylim.deleter
    def ylim(self):
        self.relim()
        self.autoscale_view(scalex=False, scaley=True)

    # -----------------------------------------------
    # scale properties

    @property
    def logx(self):
        """Display the x-axis with a logarithmic scale

        :type: `bool`
        """
        return self.axes.get_xscale() == "log"
    @logx.setter
    @auto_refresh
    def logx(self, log):
        self.axes.set_xscale(log and "log" or "linear")

    @property
    def logy(self):
        """Display the y-axis with a logarithmic scale

        :type: `bool`
        """
        return self.axes.get_yscale() == "log"
    @logy.setter
    @auto_refresh
    def logy(self, log):
        self.axes.set_yscale(log and "log" or "linear")

    # -------------------------------------------
    # Axes methods

    @auto_refresh
    def resize(self, pos, which='both'):
        """Set the axes position with::

            pos = [left, bottom, width, height]

        in relative 0,1 coords, or *pos* can be a
        :class:`~matplotlib.transforms.Bbox`

        There are two position variables: one which is ultimately
        used, but which may be modified by :meth:`apply_aspect`, and a
        second which is the starting point for :meth:`apply_aspect`.
        """
        return super(Axes, self).set_position(pos, which='both')

    @auto_refresh
    def add_label_unit(self, unit, axis='x'):
        attr = "%slabel" % axis
        label = getattr(self, 'get_%slabel' % axis)()
        if not label:
            label = unit.__doc__
        if rcParams.get("text.usetex", False):
            unitstr = tex.unit_to_latex(unit)
        else:
            unitstr = unit.to_string()
        set_ = getattr(self, 'set_%slabel' % axis)
        if label:
            set_("%s [%s]" % (label, unitstr))
        else:
            set_(unitstr)

    def legend(self, *args, **kwargs):
        # set kwargs
        alpha = kwargs.pop("alpha", 0.8)
        linewidth = kwargs.pop("linewidth", 8)

        # make legend
        legend = super(Axes, self).legend(*args, **kwargs)
        # find relevant axes
        lframe = legend.get_frame()
        lframe.set_alpha(alpha)
        [l.set_linewidth(linewidth) for l in legend.get_lines()]
        return legend
    legend.__doc__ = _Axes.legend.__doc__
