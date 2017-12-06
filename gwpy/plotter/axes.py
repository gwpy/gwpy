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

"""Extension of the `~matplotlib.axes.Axes` class with
user-friendly attributes
"""

from six import string_types

from matplotlib import rcParams
from matplotlib.axes import Axes as _Axes
from matplotlib.artist import Artist
from matplotlib.projections import register_projection

from .decorators import auto_refresh
from . import html

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['Axes']


class Axes(_Axes):
    """An extension of the core matplotlib `~matplotlib.axes.Axes`.

    These custom `Axes` provide only some simpler attribute accessors.

    Notes
    -----
    A new set of `Axes` should be constructed via::

        >>> plot.add_subplots(111, projection='xxx')

    where ``plot`` is a `~gwpy.plotter.Plot`, and ``'xxx'``
    is the name of the `Axes` you want to add.
    """
    projection = 'rectilinear'

    # -- text properties ------------------------

    # x-axis label
    @property
    def xlabel(self):
        """Label for the x-axis

        :type: `~matplotlib.text.Text`
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

        :type: `~matplotlib.text.Text`
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

    # -- limit properties -----------------------

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

    # -- scale properties -----------------------

    @property
    def logx(self):
        """Display the x-axis with a logarithmic scale

        :type: `bool`
        """
        return self.get_xscale() == "log"

    @logx.setter
    @auto_refresh
    def logx(self, log):
        if log and not self.logx:
            self.set_xscale('log')
        elif self.logx and not log:
            self.set_xscale('linear')

    @property
    def logy(self):
        """Display the y-axis with a logarithmic scale

        :type: `bool`
        """
        return self.get_yscale() == "log"

    @logy.setter
    @auto_refresh
    def logy(self, log):
        if log and not self.logy:
            self.set_yscale('log')
        elif self.logy and not log:
            self.set_yscale('linear')

    # -- Axes methods ---------------------------

    @auto_refresh
    def resize(self, pos, which='both'):
        """Set the axes position with::

        >>> pos = [left, bottom, width, height]

        in relative 0,1 coords, or *pos* can be a
        `~matplotlib.transforms.Bbox`

        There are two position variables: one which is ultimately
        used, but which may be modified by :meth:`apply_aspect`, and a
        second which is the starting point for :meth:`apply_aspect`.
        """
        return super(Axes, self).set_position(pos, which=which)

    def legend(self, *args, **kwargs):
        # set kwargs
        alpha = kwargs.pop("alpha", 0.8)
        linewidth = kwargs.pop("linewidth", 8)

        # make legend
        legend = super(Axes, self).legend(*args, **kwargs)
        # find relevant axes
        if legend is not None:
            lframe = legend.get_frame()
            lframe.set_alpha(alpha)
            lframe.set_linewidth(rcParams['axes.linewidth'])
            for line in legend.get_lines():
                line.set_linewidth(linewidth)
        return legend
    legend.__doc__ = _Axes.legend.__doc__

    def html_map(self, imagefile, data=None, **kwargs):
        """Create an HTML map for some data contained in these `Axes`

        Parameters
        ----------
        data : `~matplotlib.artist.Artist`, `~gwpy.types.Series`, `array-like`
            data to map, one of an `Artist` already drawn on these axes (
            via :meth:`plot` or :meth:`scatter`, for example) or a data set

        imagefile : `str`
            path to image file on disk for the containing `Figure`

        mapname : `str`, optional
            ID to connect <img> tag and <map> tags, default: ``'points'``. This
            should be unique if multiple maps are to be written to a single
            HTML file.

        shape : `str`, optional
            shape for <area> tag, default: ``'circle'``

        standalone : `bool`, optional
            wrap map HTML with required HTML5 header and footer tags,
            default: `True`

        title : `str`, optional
            title name for standalone HTML page

        jquery : `str`, optional
            URL of jquery script, defaults to googleapis.com URL

        Returns
        -------
        HTML : `str`
            string of HTML markup that defines the <img> and <map>
        """
        if data is None:
            artists = self.lines + self.collections + self.images
            if len(artists) != 1:
                raise ValueError("Cannot determine artist to map, %d found."
                                 % len(artists))
            data = artists[0]
        if isinstance(data, Artist):
            return html.map_artist(data, imagefile, **kwargs)
        return html.map_data(data, self, imagefile, **kwargs)


register_projection(Axes)
