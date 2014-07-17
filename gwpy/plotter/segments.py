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
from .timeseries import (TimeSeriesPlot, TimeSeriesAxes)
from .decorators import auto_refresh
from ..segments import *
from ..timeseries import StateVector

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class SegmentAxes(TimeSeriesAxes):
    """Custom `Axes` for a :class:`~gwpy.plotter.SegmentPlot`.

    This `SegmentAxes` provides custom methods for displaying any of

    - :class:`~gwpy.segments.flag.DataQualityFlag`
    - :class:`~gwpy.segments.segments.Segment`
    - :class:`~gwpy.segments.segments.SegmentList`
    - :class:`~gwpy.segments.segments.SegmentListDict`
    """
    name = 'segments'

    def __init__(self, *args, **kwargs):
        super(SegmentAxes, self).__init__(*args, **kwargs)
        self.yaxis.set_major_locator(MultipleLocator())

    def plot(self, *args, **kwargs):
        """Plot data onto these axes

        Parameters
        ----------
        args
            a single instance of

                - :class:`~gwpy.segments.flag.DataQualityFlag`
                - :class:`~gwpy.segments.segments.Segment`
                - :class:`~gwpy.segments.segments.SegmentList`
                - :class:`~gwpy.segments.segments.SegmentListDict`

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
        lim = len(self.collections)
        out = []
        args = list(args)
        while len(args):
            if isinstance(args[0], DataQualityFlag):
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
        if not lim:
            self.set_ylim(-0.1, len(self.collections) + 0.1)
        return out

    @auto_refresh
    def plot_dqflag(self, flag, y=None, valid='x', add_label=True, **kwargs):
        """Plot a :class:`~gwpy.segments.flag.DataQualityFlag`
        onto these axes

        Parameters
        ----------
        flag : :class:`~gwpy.segments.flag.DataQualityFlag`
            data-quality flag to display
        y : `float`, optional
            y-axis value for new segments
        height : `float`, optional, default: 0.8
            height for each segment block
        valid : `str`, `dict`, `None`, default: '/'
            display `valid` segments with the given hatching, or give a
            dict of keyword arguments to pass to
            :meth:`~SegmentAxes.plot_segmentlist`, or `None` to hide.
        add_label : `bool`, default: `True`
            add a label to the y-axis for this `DataQualityFlag`
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
        # add label
        if add_label:
            self.label_segments(y, name, inset=(add_label == 'inset'))
        return collection

    @auto_refresh
    def plot_segmentlist(self, segmentlist, y=None, collection=True,
                         label=None, add_label=True, **kwargs):
        """Plot a :class:`~gwpy.segments.segments.SegmentList` onto
        these axes

        Parameters
        ----------
        segmentlist : :class:`~gwpy.segments.segments.SegmentList`
            list of segments to display
        y : `float`, optional
            y-axis value for new segments
        collection : `bool`, default: `True`
            add all patches as a
            :class:`~matplotlib.collections.PatchCollection`, doesn't seem
            to work for hatched rectangles
        label : `str`, optional
            custom descriptive name to print as y-axis tick label
        add_label : `bool`, `str`, optional
            if `True` print label on y-axis, if ``'inset'`` print inside
            axes, otherwise ignore.
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
        if label and add_label:
            self.label_segments(y, label, inset=(add_label == 'inset'))
        if collection:
            return self.add_collection(PatchCollection(patches,
                                                       len(patches) != 0))
        else:
            out = []
            for p in patches:
                out.append(self.add_patch(p))
            return out

    @auto_refresh
    def plot_segmentlistdict(self, segmentlistdict, y=None, dy=1, **kwargs):
        """Plot a :class:`~gwpy.segments.segments.SegmentListDict` onto
        these axes

        Parameters
        ----------
        segmentlistdict : :class:`~gwpy.segments.segments.SegmentListDict`
            (name, :class:`~gwpy.segments.segments.SegmentList`) dict
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
        a single :class:`~gwpy.segments.segments.Segment`

        Parameters
        ----------
        segment : :class:`~gwpy.segments.segments.Segment`
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

    @auto_refresh
    def label_segments(self, y, label, inset=False, **insetparams):
        """Replace the default Y-axis label with a custom string at a
        given Y-axis position ``y``.

        Parameters
        ----------
        y : `int`
            Y-axis position to modify
        label : `str`
            custom text to insert
        inset : `bool`, optional, default: `False`
            place the label inside the axes, rather than outside
            (default)
        **insetparams
            other keyword arguments for the inset box
        """
        if label is not None:
            label = re.sub('r\\+_+', '\_', label)
        # find existing label
        # set formatter
        formatter = self.yaxis.get_major_formatter()
        if not isinstance(formatter, SegmentFormatter):
            formatter = SegmentFormatter()
            self.yaxis.set_major_formatter(formatter)

        # hide existing label and add inset text
        if inset:
            xlim = self.get_xlim()
            x = xlim[0] + (xlim[1]-xlim[0]) * 0.01
            formatter.flags[y] = ""
            insetparams.setdefault('fontsize', rcParams['axes.labelsize'])
            insetparams.setdefault('horizontalalignment', 'left')
            insetparams.setdefault('verticalalignment', 'center')
            insetparams.setdefault('backgroundcolor', 'white')
            insetparams.setdefault('transform', self.transData)
            insetparams.setdefault('bbox',
                                   {'alpha': 0.5, 'facecolor': 'white',
                                    'edgecolor': 'none'})
            t = self.text(x, y - 0.1, label or '', **insetparams)
            t._is_segment_label = True
            return t
        else:
            formatter.flags[y] = label or ''
            return

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

register_projection(SegmentAxes)


class SegmentPlot(TimeSeriesPlot):
    """`Figure` for displaying a :class:`~gwpy.segments.flag.DataQualityFlag`.

    Parameters
    ----------
    *flags : `DataQualityFlag`
        any number of :class:`~gwpy.segments.flag.DataQualityFlag` to
        display on the plot
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
        if t in self.flags:
            return self.flags[t]
        else:
            return t
