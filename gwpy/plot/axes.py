# Copyright (c) 2018-2025 Cardiff University
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

"""Extension of `~matplotlib.axes.Axes` for gwpy."""

from __future__ import annotations

import contextlib
from functools import wraps
from math import log
from numbers import Number
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from astropy.time import Time
from matplotlib import (
    _docstring,
    rcParams,
)
from matplotlib.artist import allow_rasterization
from matplotlib.axes import Axes as _Axes
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection

from ..time import to_gps
from .colorbar import colorbar
from .colors import format_norm
from .gps import GPS_SCALES
from .legend import HandlerLine2D

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )
    from typing import Literal

    import PIL.Image
    from matplotlib.artists import Artist
    from matplotlib.backend_bases import RendererBase
    from matplotlib.collections import (
        Collection,
        PathCollection,
        QuadMesh,
    )
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import (
        Colormap,
        Normalize,
    )
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.legend import Legend
    from matplotlib.patches import Polygon
    from matplotlib.transforms import Bbox
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

    from gwpy.plot.gps import GPSTransform
    from gwpy.time import SupportsToGps
    from gwpy.types import (
        Array2D,
        Series,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _sortby(
    sortby: ArrayLike,
    *arrays: ArrayLike,
) -> Iterator[numpy.ndarray]:
    """Sort a set of arrays by the first one (including the first one)."""
    # try and sort the colour array by value
    sortidx = numpy.asanyarray(sortby, dtype=float).argsort()

    def _sort(arr: ArrayLike) -> numpy.ndarray:
        if arr is None or isinstance(arr, Number):
            return arr
        return numpy.asarray(arr)[sortidx]

    # apply the sorting to each data array, and scatter
    for arr in (sortby, *arrays):
        yield _sort(arr)


def _poly_ll(x: float, y: float, w: float, h: float) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return polygon vertices for a rectangle anchored at the lower-left point."""
    return ((x, y), (x, y+h), (x+w, y+h), (x+w, y))


def _poly_lr(x: float, y: float, w: float, h: float) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return polygon vertices for a rectangle anchored at the lower-right point."""
    return ((x-w, y), (x-w, y+h), (x, y+h), (x, y))


def _poly_ul(x: float, y: float, w: float, h: float) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return polygon vertices for a rectangle anchored at the upper-left point."""
    return ((x, y-h), (x, y), (x+w, y), (x+w, y-h))


def _poly_ur(x: float, y: float, w: float, h: float) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return polygon vertices for a rectangle anchored at the upper-right point."""
    return ((x-w, y-h), (x-w, y), (x, y), (x, y-h))


def _poly_center(x: float, y: float, w: float, h: float) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return polygon vertices for a rectangle anchored at the centre point."""
    return (
        (x-w/2., y-h/2.),
        (x-w/2., y+h/2.),
        (x+w/2., y+h/2.),
        (x+w/2., y-h/2.),
    )


# -- new Axes ------------------------

class Axes(_Axes):
    """GWpy customised `~matplotlib.axes.Axes`."""

    def __init__(
        self,
        fig: Figure,
        *args: tuple[float, float, float, float] | Bbox | int,
        **kwargs,
    ) -> None:
        """Initialise a new `Axes`."""
        super().__init__(fig, *args, **kwargs)

        # handle Series in `ax.plot()`
        self._get_lines = PlotArgsProcessor()

        # reset data formatters (for interactive plots) to support
        # GPS time display
        self.fmt_xdata = self._fmt_xdata
        self.fmt_ydata = self._fmt_ydata

    @wraps(_Axes.draw)
    @allow_rasterization
    def draw(self, renderer: RendererBase) -> None:
        """Draw these `Axes` using the given renderer."""
        # Set the default GPS labels for each axis
        labels = {}
        for ax in (self.xaxis, self.yaxis):
            if ax.get_scale() in GPS_SCALES and ax.isDefault_label:
                labels[ax] = ax.get_label_text()
                trans = cast("GPSTransform", ax.get_transform())
                epoch = float(trans.get_epoch())
                unit = trans.get_unit_name()
                iso = Time(epoch, format="gps", scale="utc").iso
                utc = iso.rstrip("0").rstrip(".")
                ax.set_label_text(f"Time [{unit}] from {utc} UTC ({epoch!r})")

        try:
            super().draw(renderer)
        finally:
            for ax in labels:  # reset labels
                ax.isDefault_label = True

    # -- auto-gps helpers ------------

    def _fmt_xdata(self, x: float) -> str:
        """Format a value for display on the X-Axis."""
        if self.get_xscale() in GPS_SCALES:
            return str(to_gps(x))
        return self.xaxis.get_major_formatter().format_data_short(x)

    def _fmt_ydata(self, y: float) -> str:
        """Format a value for display on the Y-Axis."""
        if self.get_yscale() in GPS_SCALES:
            return str(to_gps(y))
        return self.yaxis.get_major_formatter().format_data_short(y)

    @wraps(_Axes.set_xlim)
    def set_xlim(
        self,
        left: SupportsToGps | tuple[SupportsToGps, SupportsToGps] | None = None,
        right: float | SupportsToGps | None = None,
        **kwargs,
    ) -> tuple[float, float]:
        """Set the X-axis view limits."""
        if right is None and numpy.iterable(left):
            left, right = left
        if self.get_xscale() in GPS_SCALES:
            with contextlib.suppress(TypeError):
                left = numpy.longdouble(str(to_gps(left)))
            with contextlib.suppress(TypeError):
                right = numpy.longdouble(str(to_gps(right)))
        return super().set_xlim(left=left, right=right, **kwargs)

    def set_epoch(self, epoch: SupportsToGps) -> None:
        """Set the epoch for the current GPS scale.

        This method will fail if the current X-axis scale isn't one of
        the GPS scales. See :ref:`gwpy-plot-gps` for more details.

        Parameters
        ----------
        epoch : `float`, `str`
            GPS-compatible time or date object, anything parseable by
            :func:`~gwpy.time.to_gps` is fine.
        """
        scale = self.get_xscale()
        return self.set_xscale(scale, epoch=epoch)

    def get_epoch(self) -> float | None:
        """Return the epoch for the current GPS scale/.

        This method will fail if the current X-axis scale isn't one of
        the GPS scales. See :ref:`gwpy-plot-gps` for more details.
        """
        return self.get_xaxis().get_transform().get_epoch()  # type: ignore[attr-defined]

    # -- overloaded plotting methods -

    def scatter(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        s: float | ArrayLike | None = None,
        c: ArrayLike | Sequence[ColorType] | ColorType | None = None,
        **kwargs,
    ) -> PathCollection:
        """Scatter ``y`` vs ``x`` with varying marker size and/or colour."""
        # This method overloads Axes.scatter to enable quick
        # sorting of data by the 'colour' array before scatter
        # plotting.

        if kwargs.pop("sortbycolor", False) and c is not None:
            # try and sort the colour array by value
            try:
                c, x, y, s = _sortby(c, x, y, s)
            except ValueError as exc:
                exc.args = (
                    "Axes.scatter argument 'sortbycolor' can only be used "
                    "with a simple array of floats in the colour array 'c'",
                )
                raise

        return super().scatter(x, y, s=s, c=c, **kwargs)

    scatter.__doc__ = _Axes.scatter.__doc__.replace(  # type: ignore[union-attr]
        "marker :",
        "sortbycolor : `bool`, optional, default: False\n"
        "    Sort scatter points by `c` array value, if given.\n\n"
        "marker :",
    )

    @_docstring.interpd
    def imshow(  # noqa: D417
        self,
        X: ArrayLike | PIL.Image.Image,  # noqa: N803
        cmap: str | Colormap | None = None,
        norm: str | Normalize | None = None,
        **kwargs,
    ) -> AxesImage:
        """Display an image, i.e. data on a 2D regular raster.

        If ``array`` is a :class:`~gwpy.types.Array2D` (e.g. a
        :class:`~gwpy.spectrogram.Spectrogram`), then the defaults are
        _different_ to those in the upstream
        :meth:`~matplotlib.axes.Axes.imshow` method. Namely, the defaults are

        - ``origin='lower'`` (coordinates start in lower-left corner)
        - ``aspect='auto'`` (pixels are not forced to be square)
        - ``interpolation='none'`` (no image interpolation is used)

        In all other usage, the defaults from the upstream matplotlib method
        are unchanged.

        Parameters
        ----------
        X : array-like or PIL image
            The image data.

        %(cmap_doc)s

        %(norm_doc)s

        *args, **kwargs
            All arguments and keywords are passed to the inherited
            :meth:`~matplotlib.axes.Axes.imshow` method.

        See Also
        --------
        matplotlib.axes.Axes.imshow
            for details of the image rendering
        """
        from gwpy.types import Array2D

        # handle log normalisation
        norm, kwargs = format_norm(kwargs | {"norm": norm})

        # handle Array2D as a special case
        if isinstance(X, Array2D):
            return self._imshow_array2d(X, cmap=cmap, norm=norm, **kwargs)

        # otherwise call back to MPL's Axes.imshow
        image = super().imshow(X, cmap=cmap, norm=norm, **kwargs)
        self.autoscale(enable=None, axis="both", tight=None)
        return image

    def _imshow_array2d(
        self,
        array: Array2D,
        origin: Literal["upper", "lower"] | None = "lower",
        interpolation: str | None = "none",
        aspect: Literal["equal", "auto"] | float | None = "auto",
        **kwargs,
    ) -> AxesImage:
        """Render an `~gwpy.types.Array2D` using `Axes.imshow`."""
        # NOTE: If you change the defaults for this method, please update
        #       the docstring for `imshow` above.

        # calculate extent
        extent = tuple(array.xspan) + tuple(array.yspan)
        if self.get_xscale() == "log" and extent[0] == 0.:
            extent = (1e-300, *extent[1:])
        if self.get_yscale() == "log" and extent[2] == 0.:
            extent = (*extent[:2], 1e-300, *extent[3:])
        kwargs.setdefault("extent", extent)

        return self.imshow(
            array.value.T,
            origin=origin,
            aspect=aspect,
            interpolation=interpolation,
            **kwargs,
        )

    def pcolormesh(
        self,
        *args: ArrayLike,
        **kwargs,
    ) -> QuadMesh:
        """Create a pseudocolor plot with a non-regular rectangular grid.

        When using GWpy, this method can be called with a single argument
        that is an :class:`~gwpy.types.Array2D`, for which the ``X`` and ``Y``
        coordinate arrays will be determined from the indexing.

        In all other usage, all ``args`` and ``kwargs`` are passed directly
        to :meth:`~matplotlib.axes.Axes.pcolormesh`.

        Notes
        -----
        Unlike the upstream :meth:`matplotlib.axes.Axes.pcolormesh`,
        this method respects the current grid settings.

        See Also
        --------
        matplotlib.axes.Axes.pcolormesh
        """
        # handle log normalisation
        norm, kwargs = format_norm(kwargs)

        # handle Array2D as a special case
        if len(args) == 1 and hasattr(args[0], "yindex"):
            array = cast("Array2D", args[0])
            return self._pcolormesh_array2d(array, norm=norm, **kwargs)

        return super().pcolormesh(*args, norm=norm, **kwargs)

    def _pcolormesh_array2d(
        self,
        array: Array2D,
        **kwargs,
    ) -> QuadMesh:
        """Render an `~gwpy.types.Array2D` using `Axes.pcolormesh`."""
        x = numpy.concatenate((array.xindex.value, array.xspan[-1:]))
        y = numpy.concatenate((array.yindex.value, array.yspan[-1:]))
        xcoord, ycoord = numpy.meshgrid(x, y, copy=False, sparse=True)
        return self.pcolormesh(
            xcoord,
            ycoord,
            array.value.T,
            **kwargs,
        )

    def hist(
        self,
        x: ArrayLike | Sequence[ArrayLike],
        bins: int | Sequence[float] | str | None = None,
        **kwargs,
    ) -> tuple[
        numpy.ndarray | list[numpy.ndarray],
        numpy.ndarray,
        BarContainer | Polygon | list[BarContainer | Polygon],
    ]:
        """Draw a histogram of some data."""
        x = numpy.asarray(x)

        # re-format weights as array if given as float
        weights = kwargs.get("weights")
        if isinstance(weights, Number):
            kwargs["weights"] = numpy.ones_like(x) * weights

        # calculate log-spaced bins on-the-fly
        if (
            # note: this needs to be first to ensure pop()
            kwargs.pop("logbins", False)
            and isinstance(bins, int | None)
        ):
            nbins = int(bins or rcParams.get("hist.bins", 30))
            # get range
            hrange = kwargs.pop("range", None)
            if hrange is None:
                try:
                    hrange = numpy.min(x), numpy.max(x)
                except ValueError as exc:
                    if str(exc).startswith("zero-size array"):  # no data
                        exc.args = (
                            "cannot generate log-spaced histogram bins for zero-"
                            "size array, please pass `bins` or `range` manually",
                        )
                    raise
            # log-scale the axis and extract the base
            if kwargs.get("orientation") == "horizontal":
                self.set_yscale("log", nonpositive="clip")
                logbase = self.yaxis._scale.base
            else:
                self.set_xscale("log", nonpositive="clip")
                logbase = self.xaxis._scale.base
            # generate the bins
            bins = numpy.logspace(
                log(hrange[0], logbase),
                log(hrange[1], logbase),
                nbins + 1,
                endpoint=True,
            )

        return super().hist(x, bins=bins, **kwargs)

    hist.__doc__ = _Axes.hist.__doc__.replace(  # type: ignore[union-attr]
        "color :",
        "logbins : boolean, optional\n"
        "    If ``True``, use logarithmically-spaced histogram bins.\n\n"
        "    Default is ``False``\n\n"
        "color :")

    # -- new plotting methods --------

    def plot_mmm(
        self,
        data: Series,
        lower: Series | None = None,
        upper: Series | None = None,
        **kwargs,
    ) -> list[Artist]:
        """Plot a `Series` as a line, with a shaded region around it.

        The ``data`` `Series` is drawn, while the ``lower`` and ``upper``
        `Series` are plotted lightly below and above, with a fill
        between them and the ``data``.

        All three `Series` should have the same `~Series.index` array.

        Parameters
        ----------
        data : `~gwpy.types.Series`
            Data to plot normally.

        lower : `~gwpy.types.Series`
            Lower boundary (on Y-axis) for shade.

        upper : `~gwpy.types.Series`
            Upper boundary (on Y-axis) for shade.

        **kwargs
            Any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`.

        Returns
        -------
        artists : `tuple`
            All of the drawn artists:

            - `~matplotlib.lines.Line2d` for ``data``,
            - `~matplotlib.lines.Line2D` for ``lower``, if given
            - `~matplotlib.lines.Line2D` for ``upper``, if given
            - `~matplitlib.collections.PolyCollection` for shading

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        alpha = kwargs.pop("alpha", .1)

        # plot mean
        line, = self.plot(data, **kwargs)
        out: list[Artist] = [line]

        # modify keywords for shading
        kwargs.update({
            "label": "",
            "linewidth": line.get_linewidth() / 2,
            "color": line.get_color(),
            "alpha": alpha * 2,
        })

        # plot lower and upper Series
        fill = [data.xindex.value, data.value, data.value]
        for i, bound in enumerate((lower, upper)):
            if bound is not None:
                out.extend(self.plot(bound, **kwargs))
                fill[i+1] = bound.value

        # fill between
        out.append(self.fill_between(
            *fill,
            alpha=alpha,
            color=kwargs["color"],
            rasterized=kwargs.get("rasterized", True),
        ))

        return out

    def tile(
        self,
        x: ArrayLike,
        y: ArrayLike,
        w: ArrayLike,
        h: ArrayLike,
        color: ArrayLike | None = None,
        anchor: Literal["center", "ll", "lr", "ul", "ur"] = "center",
        edgecolors: ColorType | Sequence[ColorType] | None = "face",
        linewidth: float | Sequence[float] | None = 0.8,
        **kwargs,
    ) -> PolyCollection:
        """Plot rectanguler tiles based onto these `Axes`.

        ``x`` and ``y`` give the anchor point for each tile, with
        ``w`` and ``h`` giving the extent in the X and Y axis respectively.

        Parameters
        ----------
        x, y, w, h : `array_like`, shape (n, )
            Input data

        color : `array_like`, shape (n, )
            Array of amplitudes for tile color

        anchor : `str`, optional
            Anchor point for tiles relative to ``(x, y)`` coordinates, one of

            - ``'center'`` - center tile on ``(x, y)``
            - ``'ll'`` - ``(x, y)`` defines lower-left corner of tile
            - ``'lr'`` - ``(x, y)`` defines lower-right corner of tile
            - ``'ul'`` - ``(x, y)`` defines upper-left corner of tile
            - ``'ur'`` - ``(x, y)`` defines upper-right corner of tile

        edgecolors : colour or list of colors, optional
            Edge colour for each tile.
            Default is the special value ``"face"`` which matches the edgecolor
            to the facecolor.

        linewidth : `float`, array of `float`, optional
            Line width for each tile.

        **kwargs
            Other keywords are passed to
            :meth:`~matplotlib.collections.PolyCollection`.

        Returns
        -------
        collection : `~matplotlib.collections.PolyCollection`
            The collection of tiles drawn.

        Examples
        --------
        >>> import numpy
        >>> from matplotlib import pyplot
        >>> import gwpy.plot  # to get gwpy's Axes

        >>> x = numpy.arange(10)
        >>> y = numpy.arange(x.size)
        >>> w = numpy.ones_like(x) * .8
        >>> h = numpy.ones_like(x) * .8

        >>> fig = pyplot.figure()
        >>> ax = fig.gca()
        >>> ax.tile(x, y, w, h, anchor='ll')
        >>> pyplot.show()
        """
        if kwargs.pop("sortbycolor", False) and color is not None:
            # try and sort the colour array by value
            try:
                color, x, y, w, h = _sortby(color, x, y, w, h)
            except ValueError as exc:
                exc.args = (
                    "Axes.tile argument 'sortbycolor' can only be used "
                    "with a simple array of floats in the `color` array",
                )
                raise

        # define how to make a polygon for each tile
        try:
            _poly = globals()[f"_poly_{anchor}"]
        except KeyError:
            msg = f"unrecognised tile anchor '{anchor}'"
            raise ValueError(msg) from None

        # build collection
        cmap = kwargs.pop("cmap", rcParams["image.cmap"])
        coll = PolyCollection(
            numpy.fromiter(
                (
                    _poly(*tile) for tile in
                    zip(x, y, w, h, strict=True)  # type: ignore[arg-type]
                ),
                dtype=(float, (4, 2)),
                count=numpy.shape(x)[0],
            ),
            edgecolors=edgecolors,
            linewidth=linewidth,
            **kwargs,
        )
        if color is not None:
            coll.set_array(color)
            coll.set_cmap(cmap)

        self.add_collection(coll)
        self.autoscale_view()
        return coll

    # -- overloaded auxiliary methods

    def legend(
        self,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> Legend:
        """Generate a `~matplotlib.legend.Legend` for these `Axes`.

        This custom method just inserts our custom `HandlerLine2D` handler into the
        default ``handler_map``.
        """
        # build custom handler to render thick lines by default
        handler_map = kwargs.setdefault("handler_map", {})
        if isinstance(handler_map, dict):
            handler_map.setdefault(Line2D, HandlerLine2D(6))

        # create legend
        return super().legend(*args, **kwargs)

    legend.__doc__ = _Axes.legend.__doc__.replace(  # type: ignore[union-attr]
        "Call signatures",
        """.. note::

   This method uses a custom default legend handler for
   `~matplotlib.lines.Line2D` objects, with increased linewidth relative
   to the upstream :meth:`~matplotlib.axes.Axes.legend` method.
   To disable this, pass ``handler_map=None``, or create and pass your
   own handler class.  See :ref:`gwpy-plot-legend` for more details.

Call signatures""",
    )

    def colorbar(
        self,
        mappable: AxesImage | Collection | None = None,
        fraction: float = 0.,
        **kwargs,
    ) -> Colorbar:
        """Add a `~matplotlib.colorbar.Colorbar` to these `Axes`.

        Parameters
        ----------
        mappable : matplotlib data collection, optional
            Collection against which to map the colouring, default will
            be the last added mappable artist (collection or image).

        fraction : `float`, optional
            Fraction of space to steal from these `Axes` to make space
            for the new axes, default is ``0.``.
            Use ``fraction=.15`` to match the upstream matplotlib default.

        **kwargs
            other keyword arguments to be passed to the
            :meth:`Plot.colorbar` generator

        Returns
        -------
        cbar : `~matplotlib.colorbar.Colorbar`
            the newly added `Colorbar`

        See Also
        --------
        Plot.colorbar
        """
        return colorbar(
            self.get_figure(),
            ax=self,
            mappable=mappable,
            fraction=fraction,
            **kwargs,
        )


# override default Axes with this one by registering a projection with the
# same name

register_projection(Axes)


# -- overload Axes.plot() to handle Series

class PlotArgsProcessor(_process_plot_var_args):
    """`Axes.plot()` argument processor."""

    def __call__(
        self,
        *args: float | ArrayLike,
        **kwargs,
    ) -> list[Artist]:
        """Find `Series` data in `plot()` args and unwrap."""
        from gwpy.types import Series

        newargs: list[float | ArrayLike] = []

        # matplotlib 3.8.0 includes the Axes object up-front
        if args and isinstance(args[0], Axes):
            newargs.append(args[0])
            args = args[1:]

        while args:
            # strip first argument
            this, args = args[:1], args[1:]
            # it its a 1-D Series, then parse it as (xindex, value)
            if isinstance(this[0], Series) and this[0].ndim == 1:
                this = (this[0].xindex.value, this[0].value)
            # otherwise treat as normal (must be a second argument)
            else:
                this += args[:1]
                args = args[1:]
            # allow colour specs
            if args and isinstance(args[0], str):
                this += (args[0],)
                args = args[1:]
            newargs.extend(this)

        return super().__call__(*newargs, **kwargs)
