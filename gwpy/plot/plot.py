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

"""Extension of the basic matplotlib Figure for GWpy."""

from __future__ import annotations

import contextlib
import importlib
import itertools
import typing
from collections.abc import (
    KeysView,
    ValuesView,
)
from itertools import zip_longest

import numpy
from matplotlib import (
    _pylab_helpers,
    backends,
    figure,
    get_backend,
)
from matplotlib.artist import setp
from matplotlib.gridspec import GridSpec
from matplotlib.projections import get_projection_class

from . import (
    colorbar as gcbar,
    utils,
)
from .gps import GPS_SCALES
from .rc import (
    MPL_RCPARAMS,
    get_subplot_params,
    rcParams,
)

if typing.TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType
    from typing import (
        BinaryIO,
        Literal,
    )

    from astropy.units import UnitBase
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.colorbar import Colorbar
    from matplotlib.image import AxesImage
    from numpy.typing import ArrayLike

    from ..segments import DataQualityFlag
    from .segments import SegmentAxes

__all__ = ["Plot"]

try:
    IPYTHON = __IPYTHON__
except NameError:
    IPYTHON = False

IterableType = list | tuple | dict | KeysView | ValuesView


def interactive_backend() -> bool:
    """Return `True` if the current backend is interactive."""
    try:
        interactive_bk = backends.backend_registry.list_builtin(
            backends.BackendFilter.INTERACTIVE,
        )
    except AttributeError:  # matplotlib < 3.9.0
        from matplotlib.rcsetup import interactive_bk
    return get_backend() in interactive_bk


def get_backend_mod(name: str | None = None) -> ModuleType:
    """Return the imported module for the given backend name.

    Parameters
    ----------
    name : `str`, optional
        the name of the backend, defaults to the current backend.

    Returns
    -------
    backend_mod: `module`
        the module as returned by :func:`importlib.import_module`

    Examples
    --------
    >>> from gwpy.plot.plot import get_backend_mod
    >>> print(get_backend_mod('agg'))
    <module 'matplotlib.backends.backend_agg' from ... >
    """
    if name is None:
        name = get_backend()
    backend_name = (
        name[9:]
        if name.startswith("module://")
        else f"matplotlib.backends.backend_{name.lower()}"
    )
    return importlib.import_module(backend_name)


class Plot(figure.Figure):
    """An extension of the core matplotlib `~matplotlib.figure.Figure`.

    The `Plot` provides a number of methods to simplify generating
    figures from GWpy data objects, and modifying them on-the-fly in
    interactive mode.
    """

    def __init__(
        self,
        *data: ArrayLike | IterableType,
        **kwargs,
    ) -> None:
        """Initialise a new `Plot`."""
        # get default x-axis scale if all axes have the same x-axis units
        kwargs.setdefault(
            "xscale",
            _parse_xscale(_group_axes_data(data, flat=True)),
        )

        # set default size for time-axis figures
        if (
            kwargs.get("projection") == "segments"
            or kwargs.get("xscale") in GPS_SCALES
        ):
            kwargs.setdefault("figsize", (12, 6))
            kwargs.setdefault("xscale", "auto-gps")

        # initialise figure
        figure_kw = {
            key: kwargs.pop(key)
            for key in utils.FIGURE_PARAMS
            if key in kwargs
        }
        self._init_figure(**figure_kw)

        # initialise axes with data
        if data or kwargs.get("geometry"):
            self._init_axes(data, **kwargs)

    def _init_figure(self, **kwargs) -> None:
        from matplotlib import pyplot

        # create Figure
        num = kwargs.pop("num", max(pyplot.get_fignums() or {0}) + 1)
        self._parse_subplotpars(kwargs)
        super().__init__(**kwargs)

        # add interactivity (scraped from pyplot.figure())
        backend_mod = get_backend_mod()
        try:
            manager = backend_mod.new_figure_manager_given_figure(num, self)
        except AttributeError:
            upstream_mod = importlib.import_module(
                pyplot.new_figure_manager.__module__,
            )
            canvas = upstream_mod.FigureCanvasBase(self)
            manager = upstream_mod.FigureManagerBase(canvas, 1)
        manager._cidgcf = manager.canvas.mpl_connect(
            "button_press_event",
            lambda ev: _pylab_helpers.Gcf.set_active(manager),
        )
        _pylab_helpers.Gcf.set_active(manager)
        pyplot.draw_if_interactive()

    def _init_axes(
        self,
        data: ArrayLike | IterableType,
        method: str = "plot",
        *,
        xscale: str | None = None,
        sharex: str | bool = False,
        sharey: str | bool = False,
        geometry: tuple[int, int] | None = None,
        separate: bool | None = None,
        **kwargs,
    ) -> list[Axes]:
        """Populate this figure with data, creating `Axes` as necessary."""
        if isinstance(sharex, bool):
            sharex = "all" if sharex else "none"
        if isinstance(sharey, bool):
            sharey = "all" if sharey else "none"

        # parse keywords
        axes_kw = {
            key: kwargs.pop(key)
            for key in utils.AXES_PARAMS
            if key in kwargs
        }

        # handle geometry and group axes
        if geometry is not None and geometry[0] * geometry[1] == len(data):
            separate = True
        axes_groups = _group_axes_data(data, separate=separate)
        if geometry is None:
            geometry = (len(axes_groups), 1)
        nrows, ncols = geometry
        if axes_groups and nrows * ncols != len(axes_groups):
            # mismatching data and geometry
            msg = (
                f"cannot group data into {len(axes_groups)} with "
                f"a {nrows}x{ncols} grid"
            )
            raise ValueError(msg)

        # create grid spec
        gs = GridSpec(nrows, ncols)
        axarr = numpy.empty((nrows, ncols), dtype=object)

        # set default labels
        defxlabel = "xlabel" not in axes_kw
        defylabel = "ylabel" not in axes_kw
        flatdata = [s for group in axes_groups for s in group]
        for ax in ("x", "y"):
            unit = _common_axis_unit(flatdata, axis=ax)
            if unit:
                axes_kw.setdefault(
                    f"{ax}label",
                    unit.to_string("latex_inline_dimensional"),
                )

        # create axes for each group and draw each data object
        for group, (row, col) in zip_longest(
            axes_groups,
            itertools.product(range(nrows), range(ncols)),
            fillvalue=[],
        ):
            # create Axes
            shared_with = {
                "none": None,
                "all": axarr[0, 0],
                "row": axarr[row, 0],
                "col": axarr[0, col],
            }
            axes_kw["sharex"] = shared_with[sharex]
            axes_kw["sharey"] = shared_with[sharey]
            axes_kw["xscale"] = xscale if xscale else _parse_xscale(group)
            ax = axarr[row, col] = self.add_subplot(gs[row, col], **axes_kw)

            # plot data
            plot_func = getattr(ax, method)
            if method in ("imshow", "pcolormesh"):
                for obj in group:
                    plot_func(obj, **kwargs)
            elif group:
                plot_func(*group, **kwargs)

            # set default axis labels
            for axis, share, pos, n, def_ in (
                (ax.xaxis, sharex, row, nrows, defxlabel),
                (ax.yaxis, sharey, col, ncols, defylabel),
            ):
                # hide label if shared axis and not bottom left panel
                if share == "all" and pos < n - 1:
                    axis.set_label_text("")
                # otherwise set default status
                else:
                    axis.isDefault_label = def_

        return self.axes

    @staticmethod
    def _parse_subplotpars(kwargs: dict) -> None:
        """Parse and set ``"subplotpars"`` from ``kwargs``."""
        # dynamically set the subplot positions based on the figure size
        # -- only if the user hasn't customised the subplot params
        figsize = kwargs.get("figsize") or rcParams["figure.figsize"]
        subplotpars = get_subplot_params(figsize)
        use_subplotpars = (
            "subplotpars" not in kwargs
            and all(
                rcParams[f"figure.subplot.{pos}"]
                == MPL_RCPARAMS[f"figure.subplot.{pos}"]
                for pos in ("left", "bottom", "right", "top")
            )
        )
        if use_subplotpars:
            kwargs["subplotpars"] = subplotpars

    # -- Plot methods ----------------

    def refresh(self) -> None:
        """Refresh the current figure."""
        self.canvas.draw()

    def show(
        self,
        warn: bool = True,
        block: bool | None = None,
    ) -> None:
        """Display the current figure (if possible).

        If blocking, this method replicates the behaviour of
        :func:`matplotlib.pyplot.show()`, otherwise it just calls up to
        :meth:`~matplotlib.figure.Figure.show`.

        This method also supports repeatedly showing the same figure, even
        after closing the display window, which isn't supported by
        `pyplot.show` (AFAIK).

        Parameters
        ----------
        block : `bool`, optional
            open the figure and block until the figure is closed, otherwise
            open the figure as a detached window, default: `None`.
            If `None`, block if using an interactive backend and _not_
            inside IPython.

        warn : `bool`, optional
            print a warning if matplotlib is not running in an interactive
            backend and cannot display the figure, default: `True`.
        """
        # this method tries to reproduce the functionality of pyplot.show,
        # mainly for user convenience. However, as of matplotlib-3.0.0,
        # pyplot.show() ends up calling _back_ to Plot.show(),
        # so we have to be careful not to end up in a recursive loop
        import inspect
        try:
            callframe = inspect.currentframe().f_back
        except AttributeError:
            pass
        else:
            if "matplotlib" in callframe.f_code.co_filename:
                block = False

        # render
        super().show(warn=warn)

        # don't block on ipython with interactive backends
        if block is None and interactive_backend():
            block = not IPYTHON

        # block in GUI loop (stolen from mpl.backend_bases._Backend.show)
        if block:
            backend_mod = get_backend_mod()
            backend_mod.Show().mainloop()

    def save(self, fname: str | Path | BinaryIO, **kwargs) -> None:
        """Save the figure to disk.

        This method is an alias to :meth:`~matplotlib.figure.Figure.savefig`,
        all arguments are passed directory to that method.
        """
        self.savefig(fname, **kwargs)

    def close(self) -> None:
        """Close the plot and release its memory."""
        from matplotlib.pyplot import close
        for ax in self.axes[::-1]:
            # avoid matplotlib/matplotlib#9970
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            # clear the axes
            ax.cla()
        # close the figure
        close(self)

    # -- axes manipulation -----------

    def get_axes(self, projection: str | None = None) -> list[Axes]:
        """Find all `Axes`, optionally matching the given projection.

        Parameters
        ----------
        projection : `str`
            Name of axes types to return.

        Returns
        -------
        axlist : `list` of `~matplotlib.axes.Axes`
            The `Axes` that match the given ``projection``.
        """
        if projection is None:
            return self.axes
        return [ax for ax in self.axes if ax.name == projection.lower()]

    # -- colour bars ----------------------------

    def colorbar(
        self,
        mappable: Collection | AxesImage | None = None,
        *,
        cax: Axes | None = None,
        ax: Axes | None = None,
        fraction: float = 0.,
        emit: bool = True,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar to the current `Plot`.

        This method differs from the default
        :meth:`matplotlib.figure.Figure.colorbar` in that it doesn't
        resize the parent `Axes` to accommodate the colorbar, but rather
        draws a new Axes alongside it.

        Parameters
        ----------
        mappable : `~matplotlib.image.AxesImage`, `~matplotlib.collections.Collection`, optional
            Collection against which to map the colorbar.
            Default is the most-recently-added mappable artist.

        cax : `~matplotlib.axes.Axes`, optional
            Axes on which to draw colorbar.
            By default a new `Axes` will be created.

        ax : `~matplotlib.axes.Axes`
            Axes relative to which to position colorbar.
            The default is the `Axes` containing the ``mappable``.

        fraction : `float`, optional
            Fraction of original axes to use for colorbar.
            The default (``fraction=0``) is to not resize the
            original axes at all.

        emit : `bool`, optional
            If `True` update all mappables on `Axes` to match the same
            colouring as the colorbar.

        kwargs
            Other keyword arguments are passed to
            :meth:`~matplotlib.figure.Figure.colorbar`.

        Returns
        -------
        cbar : `~matplotlib.colorbar.Colorbar`
            The newly added `Colorbar`.

        Notes
        -----
        To revert to the default matplotlib behaviour, pass ``fraction=0.15``.

        See Also
        --------
        matplotlib.figure.Figure.colorbar
        matplotlib.colorbar.Colorbar

        Examples
        --------
        >>> import numpy
        >>> from gwpy.plot import Plot

        To plot a simple image and add a colorbar:

        >>> plot = Plot()
        >>> ax = plot.gca()
        >>> ax.imshow(numpy.random.randn(120).reshape((10, 12)))
        >>> plot.colorbar(label='Value')
        >>> plot.show()

        Colorbars can also be generated by directly referencing the parent
        axes:

        >>> Plot = Plot()
        >>> ax = plot.gca()
        >>> ax.imshow(numpy.random.randn(120).reshape((10, 12)))
        >>> ax.colorbar(label='Value')
        >>> plot.show()
        """
        return gcbar.colorbar(
            self,
            ax,
            mappable,
            cax=cax,
            fraction=fraction,
            emit=emit,
            **kwargs,
        )

    # -- extra methods ---------------

    def add_segments_bar(
        self,
        segments: DataQualityFlag,
        ax: Axes | None = None,
        height: float = 0.14,
        pad: float = 0.1,
        *,
        sharex: bool = True,
        location: Literal["top", "bottom"] = "bottom",
        **plotargs,
    ) -> SegmentAxes:
        """Add a segment bar `Plot` indicating state information.

        By default, segments are displayed in a thin horizontal set of Axes
        sitting immediately below the x-axis of the main,
        similarly to a colorbar.

        Parameters
        ----------
        segments : `~gwpy.segments.DataQualityFlag`
            A data-quality flag, or `SegmentList` denoting state segments
            about this Plot.

        ax : `Axes`, optional
            Specific `Axes` relative to which to position new `Axes`,
            defaults to :func:`~matplotlib.pyplot.gca()`.

        height : `float, `optional
            Height of the new axes, as a fraction of the anchor axes.

        pad : `float`, optional
            Padding between the new axes and the anchor, as a fraction of
            the anchor axes dimension.

        sharex : `True`, `~matplotlib.axes.Axes`, optional
            Either `True` to set ``sharex=ax`` for the new segment axes,
            or an `Axes` to use directly.

        location : `str`, optional
            Location for new segment axes, defaults to ``'bottom'``,
            acceptable values are ``'top'`` or ``'bottom'``.

        plotargs
            Other keyword arguments are passed to
            :meth:`~gwpy.plot.SegmentAxes.plot`.
        """
        # get axes to anchor against
        if not ax:
            ax = self.gca()

        # set options for new axes
        axes_kw = {
            "pad": pad,
            "sharex": ax if sharex is True else sharex or None,
            "axes_class": get_projection_class("segments"),
        }

        # map X-axis limit from old axes
        if axes_kw["sharex"] is ax and not ax.get_autoscalex_on():
            axes_kw["xlim"] = ax.get_xlim()

        # if axes uses GPS scaling, copy the epoch as well
        with contextlib.suppress(AttributeError):
            axes_kw["epoch"] = ax.get_epoch()

        # add new axes
        try:
            divider = ax.get_axes_locator()._axes_divider
        except AttributeError:
            # get_axes_locator() is None _or_ the _axes_divider property
            # has been removed
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)

        if location not in {"top", "bottom"}:
            msg = "segments can only be positoned at 'top' or 'bottom'"
            raise ValueError(msg)

        # create axes
        segax = divider.append_axes(location, height, **axes_kw)

        # update anchor axes
        if axes_kw["sharex"] is ax and location == "bottom":
            # map label
            segax.set_xlabel(ax.get_xlabel())
            segax.xaxis.isDefault_label = ax.xaxis.isDefault_label
            ax.set_xlabel("")
            # hide ticks on original axes
            setp(ax.get_xticklabels(), visible=False)

        # plot segments
        segax.plot(segments, **plotargs)
        segax.grid(
            visible=False,
            which="both",
            axis="y",
        )
        segax.autoscale(axis="y", tight=True)

        return segax


# -- utilities -----------------------

def _group_axes_data(
    inputs: IterableType,
    *,
    separate: bool | None = None,
    flat: bool = False,
) -> list[list[ArrayLike]]:
    """Determine the number of axes from the input args to this `Plot`.

    Parameters
    ----------
    inputs : `list` of array-like data sets
        A list of data arrays, or a list of lists of data sets.

    separate : `bool`, optional
        Plot each set of data on a separate `Axes`.

    flat : `bool`, optional
        Return a flattened list of data objects.

    Returns
    -------
    axesdata : `list` of lists of array-like data
        A `list` with one element per required `Axes` containing the
        array-like data sets for those `Axes`, unless ``flat=True``
        is given.

    Notes
    -----
    The logic for this method is as follows:

    - if a `list` of data arrays are given, and `separate=False`, use 1 `Axes`
    - if a `list` of data arrays are given, and `separate=True`, use N `Axes,
      one for each data array
    - if a nested `list` of data arrays are given, ignore `sep` and
      use one `Axes` for each group of arrays.

    Examples
    --------
    >>> from gwpy.plot import Plot
    >>> Plot._group_axes_data([1, 2], separate=False)
    [[1, 2]]
    >>> Plot._group_axes_data([1, 2], separate=True)
    [[1], [2]]
    >>> Plot._group_axes_data([[1, 2], 3])
    [[1, 2], [3]]
    """
    # determine auto-separation
    if separate is None and inputs:
        separate = bool(
            # if given a nested list of data, multiple axes are required
            any(isinstance(x, IterableType) for x in inputs)
            # if data are of different types, default to separate
            or not all(type(x) is type(inputs[0]) for x in inputs),
        )

    # build list of lists
    out = []
    for x in inputs:
        if isinstance(x, dict):  # unwrap dict
            x = list(x.values())  # noqa: PLW2901

        # new group from iterable, notes:
        #     the iterable is presumed to be a list of independent data
        #     structures, unless its a list of scalars in which case we
        #     should plot them all as one
        if (
            isinstance(x, KeysView | ValuesView)
            or (
                isinstance(x, list | tuple)
                and (
                    not x
                    or not numpy.isscalar(x[0])
                )
            )
        ):
            out.append(x)

        # dataset starts a new group
        elif separate or not out:
            out.append([x])

        # dataset joins current group
        else:  # append input to most recent group
            out[-1].append(x)

    if flat:
        return [s for group in out for s in group]

    return out


def _common_axis_unit(data: IterableType, axis: str = "x") -> UnitBase | None:
    """Parse a common unit for the given ``axis`` for these ``data``."""
    units = set()
    uname = f"{axis}unit"
    for x in data:
        units.add(getattr(x, uname, None))
    if len(units) == 1:
        return units.pop()
    return None


def _parse_xscale(data: IterableType) -> str | None:
    """Parse the x-axis scale that should be applied to these ``data``."""
    unit = _common_axis_unit(data, axis="x")
    if unit is None:
        return None
    if unit.physical_type == "time":
        return "auto-gps"
    return None
