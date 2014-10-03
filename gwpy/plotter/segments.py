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

"""Extension of the simple Plot class for displaying segment objects
"""

import operator
import re

import numpy

from matplotlib.artist import allow_rasterization
from matplotlib.ticker import (Formatter, MultipleLocator, NullLocator)
from matplotlib.projections import register_projection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    from mpl_toolkits.axes_grid import make_axes_locatable

from .. import version
from . import rcParams
from ..segments import *
from ..timeseries import StateVector
from .timeseries import (TimeSeriesPlot, TimeSeriesAxes)
from .decorators import auto_refresh
from .utils import rUNDERSCORE

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class SegmentAxes(TimeSeriesAxes):
    """Custom `Axes` for a :class:`~gwpy.plotter.SegmentPlot`.

    This `SegmentAxes` provides custom methods for displaying any of

    - :class:`~gwpy.segments.DataQualityFlag`
    - :class:`~gwpy.segments.Segment`
    - :class:`~gwpy.segments.SegmentList`
    - :class:`~gwpy.segments.SegmentListDict`

    Parameters
    ----------
    insetlabels : `bool`, default: `False`
        display segment labels inside the axes. Prevents very long segment
        names from getting squeezed off the end of a standard figure

    See also
    --------
    gwpy.plotter.TimeSeriesAxes
        for documentation of other args and kwargs
    """
    name = 'segments'

    def __init__(self, *args, **kwargs):
        # set labelling format
        kwargs.setdefault('insetlabels', False)

        # make axes
        super(SegmentAxes, self).__init__(*args, **kwargs)

        # set y-axis labels
        self.yaxis.set_major_locator(MultipleLocator())
        formatter = SegmentFormatter()
        self.yaxis.set_major_formatter(formatter)

    # -------------------------------------------------------------------------
    # plotting methods

    def plot(self, *args, **kwargs):
        """Plot data onto these axes

        Parameters
        ----------
        args
            a single instance of

                - :class:`~gwpy.segments.DataQualityFlag`
                - :class:`~gwpy.segments.Segment`
                - :class:`~gwpy.segments.SegmentList`
                - :class:`~gwpy.segments.SegmentListDict`

        kwargs
            keyword arguments applicable to :meth:`~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        out = []
        args = list(args)
        while len(args):
            if isinstance(args[0], DataQualityDict):
                out.append(self.plot_dqdict(args.pop(0), **kwargs))
                continue
            elif isinstance(args[0], DataQualityFlag):
                out.append(self.plot_dqflag(args[0], **kwargs))
                args.pop(0)
                continue
            elif isinstance(args[0], SegmentListDict):
                out.extend(self.plot_segmentlistdict(args[0], **kwargs))
                args.pop(0)
                continue
            elif isinstance(args[0], SegmentList):
                out.append(self.plot_segmentlist(args[0], **kwargs))
                args.pop(0)
                continue
            elif isinstance(args[0], Segment):
                raise ValueError("Input must be DataQualityFlag, "
                                 "SegmentListDict, or SegmentList")
            break
        if len(args):
            out.append(super(SegmentAxes, self).plot(*args, **kwargs))
        self.autoscale(axis='y')
        return out

    @auto_refresh
    def plot_dqdict(self, flags, label='key', valid='x', **kwargs):
        """Plot a :class:`~gwpy.segments.DataQualityDict` onto these axes

        Parameters
        ----------
        flags : :class:`~gwpy.segments.DataQualityDict`
            data-quality dict to display
        label : `str`, optional
            labelling system to use, or fixed label for all `DataQualityFlags`.
            Special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the
              `DataQualityFlag`

            If anything else, that fixed label will be used for all lines.
        valid : `str`, `dict`, `None`, default: '/'
            display `valid` segments with the given hatching, or give a
            dict of keyword arguments to pass to
            :meth:`~SegmentAxes.plot_segmentlist`, or `None` to hide.
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : :class:`~matplotlib.patches.PatchCollection`
            list of :class:`~matplotlib.patches.Rectangle` patches
        """
        out = []
        for lab, flag in flags.iteritems():
            if label.lower() == 'name':
                lab = ts.name
            elif label.lower() != 'key':
                lab = label
            out.append(self.plot(flag, label=rUNDERSCORE.sub(r'\_', lab),
                       **kwargs))
        return out

    @auto_refresh
    def plot_dqflag(self, flag, y=None, valid='x', **kwargs):
        """Plot a :class:`~gwpy.segments.DataQualityFlag`
        onto these axes

        Parameters
        ----------
        flag : :class:`~gwpy.segments.DataQualityFlag`
            data-quality flag to display
        y : `float`, optional
            y-axis value for new segments
        height : `float`, optional, default: 0.8
            height for each segment block
        valid : `str`, `dict`, `None`, default: '/'
            display `valid` segments with the given hatching, or give a
            dict of keyword arguments to pass to
            :meth:`~SegmentAxes.plot_segmentlist`, or `None` to hide.
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : :class:`~matplotlib.patches.PatchCollection`
            list of :class:`~matplotlib.patches.Rectangle` patches
        """
        # get y axis position
        if y is None:
            y = len(self.collections)
        # get flag name
        name = kwargs.pop('label', flag.texname)

        # get epoch
        try:
            if not self.epoch:
                self.set_epoch(flag.valid[0][0])
            else:
                self.set_epoch(min(self.epoch, flag.valid[0][0]))
        except IndexError:
            pass
        # make valid collection
        if valid is not None:
            if isinstance(valid, dict):
                vkwargs = valid
            else:
                vkwargs = kwargs.copy()
                vkwargs.pop('label', None)
                vkwargs['fill'] = False
                vkwargs['hatch'] = valid
            vkwargs['collection'] = False
            vkwargs['zorder'] = -1000
            self.plot_segmentlist(flag.valid, y=y, label=None, **vkwargs)
        # make active collection
        collection = self.plot_segmentlist(flag.active, y=y, label=name,
                                           **kwargs)
        if len(self.collections) == 1:
            if len(flag.valid):
                self.set_xlim(*map(float, flag.extent))
            self.autoscale(axis='y')
        return collection

    @auto_refresh
    def plot_segmentlist(self, segmentlist, y=None, collection=True,
                         label=None, **kwargs):
        """Plot a :class:`~gwpy.segments.SegmentList` onto these axes

        Parameters
        ----------
        segmentlist : :class:`~gwpy.segments.SegmentList`
            list of segments to display
        y : `float`, optional
            y-axis value for new segments
        collection : `bool`, default: `True`
            add all patches as a
            :class:`~matplotlib.collections.PatchCollection`, doesn't seem
            to work for hatched rectangles
        label : `str`, optional
            custom descriptive name to print as y-axis tick label
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : :class:`~matplotlib.patches.PatchCollection`
            list of :class:`~matplotlib.patches.Rectangle` patches
        """
        if y is None:
            y = len(self.collections)
        patches = []
        for seg in segmentlist:
            patches.append(self.build_segment(seg, y, **kwargs))
        try:
            if not self.epoch:
                self.set_epoch(segmentlist[0][0])
            else:
                self.set_epoch(min(self.epoch, segmentlist[0][0]))
        except IndexError:
            pass
        if collection:
            coll = PatchCollection(patches, len(patches) != 0)
            coll.set_label(rUNDERSCORE.sub(r'\_', label))
            self.add_collection(coll)
        else:
            out = []
            for p in patches:
                out.append(self.add_patch(p))
            return out

    @auto_refresh
    def plot_segmentlistdict(self, segmentlistdict, y=None, dy=1, **kwargs):
        """Plot a :class:`~gwpy.segments.SegmentListDict` onto
        these axes

        Parameters
        ----------
        segmentlistdict : :class:`~gwpy.segments.SegmentListDict`
            (name, :class:`~gwpy.segments.SegmentList`) dict
        y : `float`, optional
            starting y-axis value for new segmentlists
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        collections : `list`
            list of :class:`~matplotlib.patches.PatchCollection` sets for
            each segmentlist
        """
        if y is None:
            y = len(self.collections)
        collections = []
        for name, segmentlist in segmentlistdict.iteritems():
            collections.append(self.plot_segmentlist(segmentlist, y=y,
                                                     label=name, **kwargs))
            y += dy
        return collections

    @staticmethod
    def build_segment(segment, y, height=.8, valign='center', **kwargs):
        """Build a :class:`~matplotlib.patches.Rectangle` to display
        a single :class:`~gwpy.segments.Segment`

        Parameters
        ----------
        segment : :class:`~gwpy.segments.Segment`
            [start, stop) GPS segment
        y : `float`
            y-axis position for segment
        height : `float`, optional, default: 1
            height (in y-axis units) for segment
        valign : `str`
            alignment of segment on y-axis value:
            `top`, `center`, or `bottom`
        **kwargs
            any other keyword arguments acceptable for
            :class:`~matplotlib.patches.Rectangle`

        Returns
        -------
        box : `~matplotlib.patches.Rectangle`
            rectangle patch for segment display
        """
        if valign.lower() == 'center':
            y0 = y - height/2.
        elif valign.lower() == 'top':
            y0 = y - height
        else:
            raise ValueError("valign must be one of 'top', 'center', or "
                             "'bottom'")
        return Rectangle((segment[0], y0), width=abs(segment), height=height,
                         **kwargs)

    def set_xlim(self, *args, **kwargs):
        out = super(SegmentAxes, self).set_xlim(*args, **kwargs)
        _xlim = self.get_xlim()
        try:
            texts = self.texts
        except AttributeError:
            pass
        else:
            for t in texts:
                if hasattr(t, '_is_segment_label') and t._is_segment_label:
                    t.set_x(_xlim[0] + (_xlim[1] - _xlim[0]) * 0.01)
        return out
    set_xlim.__doc__ = TimeSeriesAxes.set_xlim.__doc__

    def set_insetlabels(self, inset=None):
        self._insetlabels = (inset is None and not self._insetlabels) or inset

    def get_insetlabels(self):
        """Move the y-axis tick labels inside the axes
        """
        return self._insetlabels

    insetlabels = property(fget=get_insetlabels, fset=set_insetlabels,
                            doc=get_insetlabels.__doc__)

    @allow_rasterization
    def draw(self, *args, **kwargs):
        # inset the labels if requested
        for tick in self.get_yaxis().get_ticklabels():
            if self._insetlabels:
                tick.set_horizontalalignment('left')
                tick.set_position((0.01, tick.get_position()[1]))
                tick.set_bbox({'alpha': 0.5, 'facecolor': 'white',
                               'edgecolor': 'none'})
            else:
                tick.set_horizontalalignment('right')
                tick.set_position((0, tick.get_position()[1]))
                tick.set_bbox({})
        return super(SegmentAxes, self).draw(*args, **kwargs)

    draw.__doc__ = TimeSeriesAxes.draw.__doc__

register_projection(SegmentAxes)


class SegmentPlot(TimeSeriesPlot):
    """`Figure` for displaying a :class:`~gwpy.segments.DataQualityFlag`.

    Parameters
    ----------
    *flags : `DataQualityFlag`
        any number of :class:`~gwpy.segments.DataQualityFlag` to
        display on the plot
    insetlabels : `bool`, default: `False`
        display segment labels inside the axes. Prevents very long segment
        names from getting squeezed off the end of a standard figure
    **kwargs
        other keyword arguments as applicable for the
        :class:`~gwpy.plotter.Plot`
    """
    _DefaultAxesClass = SegmentAxes

    def __init__(self, *flags, **kwargs):
        """Initialise a new SegmentPlot
        """
        # separate kwargs into figure args and plotting args
        figargs = {}
        if 'figsize' in kwargs:
            figargs['figsize'] = kwargs.pop('figsize')
        sep = kwargs.pop('sep', False)
        epoch = kwargs.pop('epoch', None)

        # generate figure
        super(SegmentPlot, self).__init__(**figargs)

        # plot data
        if len(flags) == 1 and isinstance(flags[0], DataQualityDict):
            flags = flags[0].keys()
        for flag in flags:
            self.add_dataqualityflag(flag,
                                     projection=self._DefaultAxesClass.name,
                                     newax=sep, **kwargs)

        # set epoch
        if len(flags):
            span = reduce(operator.or_, [f.valid for f in flags]).extent()
            if not epoch:
                epoch = span[0]
            for ax in self.axes:
                ax.set_epoch(epoch)
                ax.set_xlim(*map(float, span))
            for ax in self.axes[:-1]:
                ax.set_xlabel("")

        if sep:
            for ax in self.axes:
                ax.set_ylim(-0.5, 0.5)
                ax.grid(b=False, which='both', axis='y')
        elif len(flags):
            ax.set_ylim(-0.5, len(flags)-0.5)
            ax.grid(b=False, which='both', axis='y')

    def add_dataqualityflag(self, flag, **kwargs):
        super(SegmentPlot, self).add_dataqualityflag(flag, **kwargs)
        if self.epoch is None:
            try:
                self.set_epoch(flag.valid[0][0])
            except IndexError:
                pass
    add_dataqualityflag.__doc__ = TimeSeriesPlot.add_dataqualityflag.__doc__

    def add_bitmask(self, mask, ax=None, width=0.2, pad=0.1,
                    visible=True, axes_class=SegmentAxes, topdown=False,
                    **plotargs):
        """Display a state-word bitmask on a new set of Axes.
        """
        # find default axes
        if ax is None:
            ax = self.axes[-1]

        # get new segment axes
        divider = make_axes_locatable(ax)
        max = divider.new_horizontal(size=width, pad=pad,
                                     axes_class=axes_class)
        max.set_xscale('gps')
        max.xaxis.set_major_locator(NullLocator())
        max.xaxis.set_minor_locator(NullLocator())
        max.yaxis.set_minor_locator(NullLocator())
        if visible:
            self.add_axes(max)
        else:
            return

        # format mask as a binary string and work out how many bits to set
        if isinstance(mask, int):
            mask = bin(mask)
        elif isinstance(mask, (unicode, str)) and 'x' in mask:
            mask = bin(int(mask, 16))
        maskint = int(mask, 2)
        if topdown:
            bits = list(range(len(mask.split('b', 1)[1])))[::-1]
        else:
            bits = list(range(len(mask.split('b', 1)[1])))

        # loop over bits
        plotargs.setdefault('facecolor', 'green')
        plotargs.setdefault('edgecolor', 'black')
        s = Segment(0, 1)
        for bit in bits:
            if maskint >> bit & 1:
                sl = SegmentList([s])
            else:
                sl = SegmentList()
            max.plot(sl, **plotargs)
        max.set_title('Bitmask')
        max.set_xlim(0, 1)
        max.set_xticks([])
        max.yaxis.set_ticklabels([])
        max.set_xlabel('')
        max.set_ylim(*ax.get_ylim())

        return max


class SegmentFormatter(Formatter):
    """Custom tick formatter for y-axis flag names
    """
    def __init__(self, flags={}):
        self.flags = flags

    def __call__(self, t, pos=None):
        # if segments have been plotted at this y-axis value, continue
        for coll in self.axis.axes.collections:
            if t in Segment(*coll.get_datalim(coll.axes.transData).intervaly):
                return coll.get_label()
        return ''
