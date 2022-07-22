# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2022)
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

"""Plotting utilities for segments
"""

from matplotlib.artist import allow_rasterization
from matplotlib.colors import is_color_like
from matplotlib.ticker import (Formatter, MultipleLocator)
from matplotlib.projections import register_projection
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import ligo.segments

from .axes import Axes
from .colors import tint
from .text import to_string

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']


class SegmentAxes(Axes):
    """Custom `Axes` to display segments.

    This `SegmentAxes` provides custom methods for displaying any of

    - `~gwpy.segments.DataQualityFlag`
    - `~gwpy.segments.Segment` or :class:`ligo.segments.segment`
    - `~gwpy.segments.SegmentList` or :class:`ligo.segments.segmentlist`
    - `~gwpy.segments.SegmentListDict` or
      :class:`ligo.segments.segmentlistdict`

    Parameters
    ----------
    insetlabels : `bool`, default: `False`
        display segment labels inside the axes. Prevents very long segment
        names from getting squeezed off the end of a standard figure

    See also
    --------
    gwpy.plot.Axes
        for documentation of other args and kwargs
    """
    name = 'segments'

    def __init__(self, *args, **kwargs):
        # default to auto-gps scale on the X-axis
        kwargs.setdefault('xscale', 'auto-gps')

        # set labelling format
        kwargs.setdefault('insetlabels', False)

        # make axes
        super().__init__(*args, **kwargs)

        # set y-axis labels
        self.yaxis.set_major_locator(MultipleLocator())
        formatter = SegmentFormatter()
        self.yaxis.set_major_formatter(formatter)

    def _plot_method(self, obj):
        from ..segments import (DataQualityFlag, DataQualityDict)

        if isinstance(obj, DataQualityDict):
            return self.plot_dict
        if isinstance(obj, DataQualityFlag):
            return self.plot_flag
        if isinstance(obj, ligo.segments.segmentlistdict):
            return self.plot_segmentlistdict
        if isinstance(obj, ligo.segments.segmentlist):
            return self.plot_segmentlist
        raise TypeError(
            f"no known {type(self).__name__}.plot_xxx method "
            f"for {type(obj).__name__}",
        )

    def plot(self, *args, **kwargs):
        """Plot data onto these axes

        Parameters
        ----------
        args
            a single instance of

                - `~gwpy.segments.DataQualityFlag`
                - `~gwpy.segments.Segment`
                - `~gwpy.segments.SegmentList`
                - `~gwpy.segments.SegmentListDict`

            or equivalent types upstream from :mod:`ligo.segments`

        kwargs
            keyword arguments applicable to `~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the `~matplotlib.lines.Line2D` for this line layer

        See also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        out = []
        args = list(args)
        while args:
            try:
                plotter = self._plot_method(args[0])
            except TypeError:
                break
            out.append(plotter(args[0], **kwargs))
            args.pop(0)
        if args:
            out.extend(super().plot(*args, **kwargs))
        self.autoscale(enable=None, axis='both', tight=False)
        return out

    def plot_dict(self, flags, label='key', known='x', **kwargs):
        """Plot a `~gwpy.segments.DataQualityDict` onto these axes

        Parameters
        ----------
        flags : `~gwpy.segments.DataQualityDict`
            data-quality dict to display

        label : `str`, optional
            labelling system to use, or fixed label for all `DataQualityFlags`.
            Special values include

            - ``'key'``: use the key of the `DataQualityDict`,
            - ``'name'``: use the :attr:`~DataQualityFlag.name` of the
              `DataQualityFlag`

            If anything else, that fixed label will be used for all lines.

        known : `str`, `dict`, `None`, default: '/'
            display `known` segments with the given hatching, or give a
            dict of keyword arguments to pass to
            :meth:`~SegmentAxes.plot_segmentlist`, or `None` to hide.

        **kwargs
            any other keyword arguments acceptable for
            `~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : `~matplotlib.patches.PatchCollection`
            list of `~matplotlib.patches.Rectangle` patches
        """
        out = []
        for lab, flag in flags.items():
            if label.lower() == 'name':
                lab = flag.name
            elif label.lower() != 'key':
                lab = label
            out.append(self.plot_flag(flag, label=to_string(lab), known=known,
                                      **kwargs))
        return out

    def plot_flag(self, flag, y=None, **kwargs):
        """Plot a `~gwpy.segments.DataQualityFlag` onto these axes.

        Parameters
        ----------
        flag : `~gwpy.segments.DataQualityFlag`
            Data-quality flag to display.

        y : `float`, optional
            Y-axis value for new segments.

        height : `float`, optional,
            Height for each segment, default: `0.8`.

        known : `str`, `dict`, `None`
            One of the following

            - ``'fancy'`` - to use fancy format (try it and see)
            - ``'x'`` (or similar) - to use hatching
            - `str` to specify ``facecolor`` for known segmentlist
            - `dict` of kwargs to use
            - `None` to ignore known segmentlist

        **kwargs
            Any other keyword arguments acceptable for
            `~matplotlib.patches.Rectangle`.

        Returns
        -------
        collection : `~matplotlib.patches.PatchCollection`
            list of `~matplotlib.patches.Rectangle` patches for active
            segments
        """
        # get y axis position
        if y is None:
            y = self.get_next_y()

        # default a 'good' flag to green segments and vice-versa
        if flag.isgood:
            kwargs.setdefault('facecolor', '#33cc33')
            kwargs.setdefault('known', '#ff0000')
        else:
            kwargs.setdefault('facecolor', '#ff0000')
            kwargs.setdefault('known', '#33cc33')
        known = kwargs.pop('known')

        # get flag name
        name = kwargs.pop('label', flag.label or flag.name)

        # make active collection
        kwargs.setdefault('zorder', 0)
        coll = self.plot_segmentlist(flag.active, y=y, label=name,
                                     **kwargs)

        # make known collection
        if known not in (None, False):
            known_kw = {
                'facecolor': coll.get_facecolor()[0],
                'collection': 'ignore',
                'zorder': -1000,
            }
            if isinstance(known, dict):
                known_kw.update(known)
            elif known == 'fancy':
                known_kw.update(height=kwargs.get('height', .8)*.05)
            elif known in HATCHES:
                known_kw.update(fill=False, hatch=known)
            else:
                known_kw.update(fill=True, facecolor=known,
                                height=kwargs.get('height', .8)*.5)
            self.plot_segmentlist(flag.known, y=y, label=name, **known_kw)

        return coll  # return active collection

    def plot_segmentlist(self, segmentlist, y=None, height=.8, label=None,
                         collection=True, rasterized=None, **kwargs):
        """Plot a `~gwpy.segments.SegmentList` onto these axes

        Parameters
        ----------
        segmentlist : `~gwpy.segments.SegmentList`
            list of segments to display

        y : `float`, optional
            y-axis value for new segments

        collection : `bool`, default: `True`
            add all patches as a
            `~matplotlib.collections.PatchCollection`, doesn't seem
            to work for hatched rectangles

        label : `str`, optional
            custom descriptive name to print as y-axis tick label

        **kwargs
            any other keyword arguments acceptable for
            `~matplotlib.patches.Rectangle`

        Returns
        -------
        collection : `~matplotlib.patches.PatchCollection`
            list of `~matplotlib.patches.Rectangle` patches
        """
        # get colour
        facecolor = kwargs.pop('facecolor', kwargs.pop('color', '#629fca'))
        if is_color_like(facecolor):
            kwargs.setdefault('edgecolor', tint(facecolor, factor=.5))

        # get y
        if y is None:
            y = self.get_next_y()

        # build patches
        patches = [SegmentRectangle(seg, y, height=height, facecolor=facecolor,
                                    **kwargs) for seg in segmentlist]

        if collection:  # map to PatchCollection
            coll = PatchCollection(patches, match_original=patches,
                                   zorder=kwargs.get('zorder', 1))
            coll.set_rasterized(rasterized)
            coll._ignore = collection == 'ignore'
            coll._ypos = y
            out = self.add_collection(coll)
            # reset label with tex-formatting now
            #   matplotlib default label is applied by add_collection
            #   so we can only replace the leading underscore after
            #   this point
            if label is None:
                label = coll.get_label()
            coll.set_label(to_string(label))
        else:
            out = []
            for patch in patches:
                patch.set_label(label)
                patch.set_rasterized(rasterized)
                label = ''
                out.append(self.add_patch(patch))
        self.autoscale(enable=None, axis='both', tight=False)
        return out

    def plot_segmentlistdict(self, segmentlistdict, y=None, dy=1, **kwargs):
        """Plot a `~gwpy.segments.SegmentListDict` onto
        these axes

        Parameters
        ----------
        segmentlistdict : `~gwpy.segments.SegmentListDict`
            (name, `~gwpy.segments.SegmentList`) dict

        y : `float`, optional
            starting y-axis value for new segmentlists

        **kwargs
            any other keyword arguments acceptable for
            `~matplotlib.patches.Rectangle`

        Returns
        -------
        collections : `list`
            list of `~matplotlib.patches.PatchCollection` sets for
            each segmentlist
        """
        if y is None:
            y = self.get_next_y()
        collections = []
        for name, segmentlist in segmentlistdict.items():
            collections.append(self.plot_segmentlist(segmentlist, y=y,
                                                     label=name, **kwargs))
            y += dy
        return collections

    def get_next_y(self):
        """Find the next y-axis value at which a segment list can be placed

        This method simply counts the number of independent segmentlists or
        flags that have been plotted onto these axes.
        """
        return len(self.get_collections(ignore=False))

    def get_collections(self, ignore=None):
        """Return the collections matching the given `_ignore` value

        Parameters
        ----------
        ignore : `bool`, or `None`
            value of `_ignore` to match

        Returns
        -------
        collections : `list`
            if `ignore=None`, simply returns all collections, otherwise
            returns those collections matching the `ignore` parameter
        """
        if ignore is None:
            return self.collections
        return [c for c in self.collections if
                getattr(c, '_ignore', None) == ignore]

    def set_insetlabels(self, inset=None):
        """Set the labels to be inset or not

        Parameters
        ----------
        inset : `bool`, `None`
            if `None`, toggle the inset state, otherwise set the labels to
            be inset (`True) or not (`False`)
        """
        # pylint: disable=attribute-defined-outside-init
        self._insetlabels = not self._insetlabels if inset is None else inset

    def get_insetlabels(self):
        """Returns the inset labels state
        """
        return self._insetlabels

    insetlabels = property(fget=get_insetlabels, fset=set_insetlabels,
                           doc=get_insetlabels.__doc__)

    @allow_rasterization
    def draw(self, *args, **kwargs):  # pylint: disable=missing-docstring
        # inset the labels if requested
        for tick in self.get_yaxis().get_ticklabels():
            if self.get_insetlabels():
                # record parameters we are changing
                # pylint: disable=protected-access
                tick._orig_bbox = tick.get_bbox_patch()
                tick._orig_ha = tick.get_ha()
                tick._orig_pos = tick.get_position()
                # modify tick
                tick.set_horizontalalignment('left')
                tick.set_position((0.01, tick.get_position()[1]))
                tick.set_bbox({'alpha': 0.5, 'facecolor': 'white',
                               'edgecolor': 'none'})
            elif self.get_insetlabels() is False:
                # if label has been moved, reset things
                # pylint: disable=protected-access
                try:
                    tick.set_bbox(tick._orig_bbox)
                except AttributeError:
                    pass
                else:
                    tick.set_horizontalalignment(tick._orig_ha)
                    tick.set_position(tick._orig_pos)
                    del tick._orig_bbox
                    del tick._orig_ha
                    del tick._orig_pos
        return super().draw(*args, **kwargs)

    draw.__doc__ = Axes.draw.__doc__


register_projection(SegmentAxes)


class SegmentFormatter(Formatter):
    """Custom tick formatter for y-axis flag names
    """
    def __call__(self, t, pos=None):
        # if segments have been plotted at this y-axis value, continue
        for coll in self.axis.axes.get_collections(ignore=False):
            if t == coll._ypos:  # pylint: disable=protected-access
                return coll.get_label()
        for patch in self.axis.axes.patches:
            if not patch.get_label() or patch.get_label() == '_nolegend_':
                continue
            if t in ligo.segments.segment(*patch.get_bbox().intervaly):
                return patch.get_label()
        return ''


# -- segment patch ------------------------------------------------------------

class SegmentRectangle(Rectangle):
    def __init__(self, segment, y, height=.8, valign='center', **kwargs):
        """Build a `~matplotlib.patches.Rectangle` from a segment

        Parameters
        ----------
        segment : `~gwpy.segments.Segment`
            ``[start, stop)`` GPS segment

        y : `float`
            y-axis position for segment

        height : `float`, optional, default: 1
            height (in y-axis units) for segment

        valign : `str`
            alignment of segment on y-axis value:
                `top`, `center`, or `bottom`

        **kwargs
            any other keyword arguments acceptable for
            `~matplotlib.patches.Rectangle`

        Returns
        -------
        box : `~matplotlib.patches.Rectangle`
            rectangle patch for segment display
        """
        if valign.lower() == 'bottom':
            y0 = y
        elif valign.lower() in ['center', 'centre']:
            y0 = y - height/2.
        elif valign.lower() == 'top':
            y0 = y - height
        else:
            raise ValueError("valign must be one of 'top', 'center', or "
                             "'bottom'")
        width = segment[1] - segment[0]

        super().__init__((segment[0], y0), width=width,
                         height=height, **kwargs)
        self.segment = segment
