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

import numpy
from matplotlib.projections import register_projection

from .. import version
from .timeseries import (TimeSeriesPlot, TimeSeriesAxes)
from ..segments import *

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class SegmentAxes(TimeSeriesAxes):
    """Axes designed to show `SegmentList`, and `DataQualityFlag`-format
    objects
    """
    name = 'segments'
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
        :meth:`~matplotlib.axes.Axes.plot`
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
                out.append(self.plot_segment(args[0], **kwargs))
                args.pop(0)
                continue
            break
        if len(args):
            out.append(super(SegmentAxes, self).plot(*args, **kwargs))
        if not lim:
            self.set_ylim(-0.1, len(self.collections) + 0.1)
        return out

register_projection(SegmentAxes)


class SegmentPlot(TimeSeriesPlot):
    """An extension of the
    :class:`~gwpy.plotter.timeseries.TimeSeriesPlot` class for
    displaying data from
    :class:`DataQualityFlags <~gwpy.segments.flagDataQualityFlag>`.

    Parameters
    ----------
    *flags : `DataQualityFlag`
        any number of :class:`~gwpy.segments.flag.DataQualityFlag` to
        display on the plot
    **kwargs
        other keyword arguments as applicable for the
        :class:`~gwpy.plotter.core.Plot`
    """
    _DefaultAxesClass = SegmentAxes
    def __init__(self, *flags, **kwargs):
        """Initialise a new SegmentPlot
        """
        sep = kwargs.pop('sep', False)
        epoch = kwargs.pop('epoch', None)
        labels = kwargs.pop('labels', None)
        valid = kwargs.pop('valid', 'x')

        # generate figure
        super(SegmentPlot, self).__init__(**kwargs)
        # plot data
        for flag in flags:
            self.add_dataqualityflag(flag,
                                     projection=self._DefaultAxesClass.name,
                                     newax=sep, valid=valid)

        # set epoch
        if len(flags):
            span = reduce(operator.or_, [f.valid for f in flags]).extent()
            if not epoch:
                epoch = span[0]
            for ax in self.axes:
                ax.set_epoch(epoch)
                ax.set_xlim(*span)
                if not hasattr(self, '_auto_gps') or self._auto_gps:
                    ax.auto_gps_scale()
            for ax in self.axes[:-1]:
                ax.set_xlabel("")

        # set labels
        if not labels:
            labels = []
            for flag in flags:
                name = ':'.join([str(p) for p in
                                 (flag.ifo, flag.name, flag.version) if p is
                                 not None])
                labels.append(name.replace('_', r'\_'))
        if sep:
            for ax, label in zip(self.axes, labels):
                ax.set_yticks([0])
                ax.set_yticklabels([label])
                ax.set_ylim(-0.5, 0.5)
        else:
            ticks = numpy.arange(len(flags))
            ax = self.axes[0]
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_ylim(-0.5, len(flags)-0.5)
        ax.grid(b=False, which='both', axis='y')

    def add_dataqualityflag(self, flag, **kwargs):
        super(SegmentPlot, self).add_dataqualityflag(flag, **kwargs)
        if not self.epoch:
            try:
                self.set_epoch(flag.valid[0][0])
            except IndexError:
                pass
    add_dataqualityflag.__doc__ = TimeSeriesPlot.add_dataqualityflag.__doc__
