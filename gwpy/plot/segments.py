# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Plotting utilities for segments."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import igwn_segments
from matplotlib.artist import allow_rasterization
from matplotlib.collections import PatchCollection
from matplotlib.colors import is_color_like
from matplotlib.patches import Rectangle
from matplotlib.projections import register_projection
from matplotlib.ticker import (
    Formatter,
    MultipleLocator,
)

from .axes import Axes
from .colors import tint
from .text import to_string

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from igwn_segments import (
        segment,
        segmentlist,
        segmentlistdict,
    )
    from matplotlib.artists import Artist
    from matplotlib.backend_bases import RendererBase
    from matplotlib.collections import Collection

    from gwpy.segments import (
        DataQualityDict,
        DataQualityFlag,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


class SegmentAxes(Axes):
    """Custom `Axes` to display segments.

    This `SegmentAxes` provides custom methods for displaying any of

    - `~gwpy.segments.DataQualityFlag`
    - `~gwpy.segments.Segment` or :class:`igwn_segments.segment`
    - `~gwpy.segments.SegmentList` or :class:`igwn_segments.segmentlist`
    - `~gwpy.segments.SegmentListDict` or
      :class:`igwn_segments.segmentlistdict`

    Parameters
    ----------
    insetlabels : `bool`, default: `False`
        display segment labels inside the axes. Prevents very long segment
        names from getting squeezed off the end of a standard figure

    See Also
    --------
    gwpy.plot.Axes
        for documentation of other args and kwargs
    """

    name = "segments"

    def __init__(
        self,
        *args,
        xscale: str = "auto-gps",
        insetlabels: bool = False,
        **kwargs,
    ) -> None:
        """Initialise a new `SegmentAxes`."""
        # make axes
        super().__init__(
            *args,
            xscale=xscale,
            insetlabels=insetlabels,
            **kwargs,
        )

        # set y-axis labels
        self.yaxis.set_major_locator(MultipleLocator())
        formatter = SegmentFormatter()
        self.yaxis.set_major_formatter(formatter)

    def _plot_method(
        self,
        obj: DataQualityFlag | DataQualityDict | segmentlist | segmentlistdict,
    ) -> Callable:
        from ..segments import (
            DataQualityDict,
            DataQualityFlag,
        )

        if isinstance(obj, DataQualityDict):
            return self.plot_dict
        if isinstance(obj, DataQualityFlag):
            return self.plot_flag
        if isinstance(obj, igwn_segments.segmentlistdict):
            return self.plot_segmentlistdict
        if isinstance(obj, igwn_segments.segmentlist):
            return self.plot_segmentlist
        msg = (
            f"no known {type(self).__name__}.plot_xxx method "
            f"for {type(obj).__name__}"
        )
        raise TypeError(msg)

    def plot(
        self,
        *args: DataQualityFlag | DataQualityDict | segmentlist | segmentlistdict,
        **kwargs,
    ) -> list[Artist]:
        """Plot data onto these axes.

        Parameters
        ----------
        args
            A single instance of

                - `~gwpy.segments.DataQualityFlag`
                - `~gwpy.segments.Segment`
                - `~gwpy.segments.SegmentList`
                - `~gwpy.segments.SegmentListDict`

            or equivalent types upstream from :mod:`igwn_segments`.

        kwargs
            All keyword arguments are passed to `~matplotib.axes.Axes.plot`
            when drawing each object.

        Returns
        -------
        artists : `list` of `~matplotlib.artist.Artist`
            The list of things that were rendered.

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args` and ``**kwargs``.
        """
        out = []

        # loop over the various arguments, and plot them
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

        self.autoscale(
            enable=None,
            axis="both",
            tight=False,
        )
        return out

    def plot_dict(
        self,
        flags: DataQualityDict,
        label: str = "key",
        known: str | dict | None = "x",
        **kwargs,
    ) -> list[PatchCollection]:
        """Plot a `~gwpy.segments.DataQualityDict` onto these axes.

        Parameters
        ----------
        flags : `~gwpy.segments.DataQualityDict`
            The data-quality dict to display.

        label : `str`, optional
            Labelling system to use, or fixed label for all `DataQualityFlags`.
            Special values include

            ``'key'``
                Use the key of the `DataQualityDict`.

            ``'name'``
                Use the :attr:`~DataQualityFlag.name` of the `DataQualityFlag`.

            If anything else, that fixed label will be used for all lines.

        known : `str`, `dict`, `None`, optional
            Display `known` segments with the given hatching, or give a
            dict of keyword arguments to pass to
            :meth:`~SegmentAxes.plot_segmentlist`, or `None` to hide.

        kwargs
            All other keyword arguments are passed to `SegmentRectangle`.

        Returns
        -------
        artists : `list` of `~matplotlib.patches.PatchCollection`
            The list of patch collections that were drawn.
        """
        out = []
        for name, flag in flags.items():
            if label.lower() == "name":
                lab = flag.name
            elif label.lower() == "key":
                lab = name
            else:
                lab = label
            out.append(self.plot_flag(
                flag,
                label=to_string(lab),
                known=known,
                **kwargs,
            ))
        return out

    def plot_flag(
        self,
        flag: DataQualityFlag,
        y: float | None = None,
        **kwargs,
    ) -> PatchCollection | list[SegmentRectangle]:
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

        kwargs
            All other keyword arguments are passed to `SegmentRectangle`.

        Returns
        -------
        artists : `list` of `~matplotlib.patches.PatchCollection`
            The list of patch collections that were drawn.
        """
        # get y axis position
        if y is None:
            y = self.get_next_y()

        # default a 'good' flag to green segments and vice-versa
        if flag.isgood:
            kwargs.setdefault("facecolor", "#33cc33")
            kwargs.setdefault("known", "#ff0000")
        else:
            kwargs.setdefault("facecolor", "#ff0000")
            kwargs.setdefault("known", "#33cc33")
        known = kwargs.pop("known")

        # get flag name
        name = kwargs.pop("label", flag.label or flag.name)

        # make active collection
        kwargs.setdefault("zorder", 0)
        coll = self.plot_segmentlist(
            flag.active,
            y=y,
            label=name,
            **kwargs,
        )

        # make known collection
        if known not in (None, False):
            known_kw = {
                "facecolor": coll.get_facecolor()[0],
                "collection": "ignore",
                "zorder": -1000,
            }
            if isinstance(known, dict):
                known_kw.update(known)
            elif known == "fancy":
                known_kw.update(height=kwargs.get("height", .8) * .05)
            elif known in HATCHES:
                known_kw.update(fill=False, hatch=known)
            else:
                known_kw.update(
                    fill=True,
                    facecolor=known,
                    height=kwargs.get("height", .8) * .5,
                )
            self.plot_segmentlist(flag.known, y=y, label=name, **known_kw)

        return coll

    def plot_segmentlist(
        self,
        segmentlist: segmentlist,
        y: float | None = None,
        height: float = .8,
        label: str | None = None,
        *,
        collection: bool = True,
        rasterized: bool | None = None,
        **kwargs,
    ) -> PatchCollection | list[SegmentRectangle]:
        """Plot a `~gwpy.segments.SegmentList` onto these axes.

        Parameters
        ----------
        segmentlist : `~gwpy.segments.SegmentList`
            List of segments to display.

        y : `float`, optional
            Y-axis value for new segments.

        height : `float`, optional
            Height (in y-axis units) for segment.

        collection : `bool`, optional
            If `True` (default), bundle all patches as a
            `~matplotlib.collections.PatchCollection`, otherwise
            just return a `list` of `SegmentRectangle` patches.

        rasterized : `bool`, optional
            If `True`, rasterise the patches when drawing.
            Default is `False`.

        label : `str`, optional
            custom descriptive name to print as y-axis tick label

        kwargs
            All other keyword arguments are passed to `SegmentRectangle`.

        Returns
        -------
        patches : `~matplotlib.patches.PatchCollection` or `list[SegmentRectangle]`
            The drawn patches, bundled in a `PatchCollection` if ``collection=True``.
        """
        # get colour
        facecolor = kwargs.pop("facecolor", kwargs.pop("color", "#629fca"))
        if is_color_like(facecolor):
            kwargs.setdefault("edgecolor", tint(facecolor, factor=.5))

        # get y
        if y is None:
            y = self.get_next_y()

        # build patches
        patches = [
            SegmentRectangle(
                seg,
                y,
                height=height,
                facecolor=facecolor,
                **kwargs,
            )
            for seg in segmentlist
        ]

        if collection:  # map to PatchCollection
            coll = PatchCollection(
                patches,
                match_original=patches,
                zorder=kwargs.get("zorder", 1),
            )
            coll.set_rasterized(rasterized)
            coll._ignore = collection == "ignore"
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
                label = ""  # don't label any other patches
                out.append(self.add_patch(patch))
        self.autoscale(enable=None, axis="both", tight=False)
        return out

    def plot_segmentlistdict(
        self,
        segmentlistdict: segmentlistdict,
        y: float | None = None,
        dy: float = 1,
        **kwargs,
    ) -> PatchCollection | list[SegmentRectangle]:
        """Plot a `~gwpy.segments.SegmentListDict` onto these axes.

        Parameters
        ----------
        segmentlistdict : `~gwpy.segments.SegmentListDict`
            (name, `~gwpy.segments.SegmentList`) dict.

        y : `float`, optional
            Starting y-axis value for new segmentlists.

        dy : `float`, optional
            Y-axis separation between each (anchor of each) segmentlist.

        kwargs
            All other keyword arguments are passed to
            `~matplotlib.patches.Rectangle`.

        Returns
        -------
        collections : `list` of `PatchCollection`
            List of `~matplotlib.patches.PatchCollection` sets for
            each segmentlist.
        """
        if y is None:
            y = self.get_next_y()
        collections = []
        for name, segmentlist in segmentlistdict.items():
            collections.append(self.plot_segmentlist(
                segmentlist,
                y=y,
                label=name,
                **kwargs,
            ))
            y += dy
        return collections

    def get_next_y(self) -> int:
        """Find the next y-axis value at which a segment list can be placed.

        This method simply counts the number of independent segmentlists or
        flags that have been plotted onto these axes.
        """
        return len(self.get_collections(ignore=False))

    def get_collections(self, ignore: bool | None = None) -> list[Collection]:
        """Return the collections matching the given `_ignore` value.

        Parameters
        ----------
        ignore : `bool`, or `None`
            Value of `_ignore` to match.

        Returns
        -------
        collections : `list` of `~matplotlib.collections.Collection`
            If `ignore=None`, simply returns all collections, otherwise
            returns those collections matching the `ignore` parameter.
        """
        if ignore is None:
            return self.collections
        return [
            c for c in self.collections
            if getattr(c, "_ignore", None) == ignore
        ]

    def set_insetlabels(self, inset: bool | None = None) -> None:
        """Set the labels to be inset or not.

        Parameters
        ----------
        inset : `bool`, `None`
            Enable (`True`) or disable (`False`) inset labels.
            Default (`None`) toggles the current state.
        """
        self._insetlabels = not self._insetlabels if inset is None else inset

    def get_insetlabels(self) -> bool:
        """Return the inset labels state."""
        return self._insetlabels

    insetlabels = property(
        fget=get_insetlabels,
        fset=set_insetlabels,
        doc=get_insetlabels.__doc__,
    )

    @allow_rasterization
    @wraps(Axes.draw)
    def draw(self, renderer: RendererBase) -> None:
        """Draw the current `SegmentAxes`."""
        # inset the labels if requested
        for tick in self.get_yaxis().get_ticklabels():
            if self.get_insetlabels():
                # record parameters we are changing
                tick._orig_bbox = tick.get_bbox_patch()
                tick._orig_ha = tick.get_ha()
                tick._orig_pos = tick.get_position()
                # modify tick
                tick.set_horizontalalignment("left")
                tick.set_position((0.01, tick.get_position()[1]))
                tick.set_bbox({
                    "alpha": 0.5,
                    "edgecolor": "none",
                    "facecolor": "white",
                })
            elif self.get_insetlabels() is False:
                # if label has been moved, reset things
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
        return super().draw(renderer)

    draw.__doc__ = Axes.draw.__doc__


register_projection(SegmentAxes)


class SegmentFormatter(Formatter):
    """Custom tick formatter for y-axis flag names."""

    def __call__(
        self,
        t: float,
        pos: float | None = None,  # noqa: ARG002
    ) -> str:
        """Format ticks using segment names."""
        ax = self.axis.axes

        # if segments have been plotted at this y-axis value, continue
        for coll in ax.get_collections(ignore=False):
            if t == getattr(coll, "_ypos", None):
                return coll.get_label()

        for patch in ax.patches:
            # if this patch doesn't have a label or doesn't want one, carry on
            if not patch.get_label() or patch.get_label() == "_nolegend_":
                continue
            # otherwise if the axis position overlaps the bbox, emit the
            # patch label as the tick label
            if t in igwn_segments.segment(*patch.get_bbox().intervaly):
                return patch.get_label()

        return ""


# -- segment patch -------------------

class SegmentRectangle(Rectangle):
    """Custom `~matplotlib.patches.Rectangle` for a `~gwpy.segments.Segment`."""

    def __init__(
        self,
        segment: segment,
        y: float,
        height: float = .8,
        valign: Literal["top", "center", "bottom"] = "center",
        **kwargs,
    ) -> None:
        """Build a `~matplotlib.patches.Rectangle` from a segment.

        Parameters
        ----------
        segment : `~gwpy.segments.Segment`
            ``[start, stop)`` GPS segment.

        y : `float`
            Y-axis position for segment.

        height : `float`, optional
            Height (in y-axis units) for segment.

        valign : `str`, optional
            Alignment of segment on y-axis value, one of
            ``"top"``, ``"center"``, or ``"bottom"``.
            Default: ``"center"``.

        kwargs
            All other keyword arguments are passed to
            `~matplotlib.patches.Rectangle`.

        Raises
        ------
        ValueError
            If an invalid ``valign`` value is given.
        """
        if valign.lower() == "bottom":
            y0 = y
        elif valign.lower() in {"center", "centre"}:
            y0 = y - height / 2.
        elif valign.lower() == "top":
            y0 = y - height
        else:
            msg = "valign must be one of 'top', 'center', or 'bottom'"
            raise ValueError(msg)
        width = segment[1] - segment[0]

        super().__init__(
            (segment[0], y0),
            width=width,
            height=height,
            **kwargs,
        )
        self.segment = segment
