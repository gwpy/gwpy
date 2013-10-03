# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Extension of the simple Plot class for displaying segment objects
"""

from matplotlib.projections import register_projection

from ..version import version as version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

from .timeseries import TimeSeriesAxes
from ..segments import *


class TimeSegmentAxes(TimeSeriesAxes):
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
                out.extend(self.plot_segmentlist(args[0], **kwargs))
                args.pop(0)
                continue
            elif isinstance(args[0], Segment):
                out.append(self.plot_segment(args[0], **kwargs))
                args.pop(0)
                continue
            break
        if len(args):
            out.append(super(TimeSegmentAxes, self).plot(*args, **kwargs))
        if not lim:
            self.set_ylim(-0.1, len(self.collections) + 0.1)
        return out

register_projection(TimeSegmentAxes)
