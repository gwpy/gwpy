# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2022)
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

"""Extension of `~matplotlib.axes.Axes` for gwpy
"""

import warnings
from functools import wraps
from math import log
from numbers import Number

import numpy

from astropy.time import Time

from matplotlib import (
    __version__ as matplotlib_version,
    rcParams,
)
from matplotlib.artist import allow_rasterization
from matplotlib.axes import Axes as _Axes
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection

from . import (Plot, colorbar as gcbar)
from .colors import format_norm
from .gps import GPS_SCALES
from .legend import HandlerLine2D
from ..time import to_gps

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def log_norm(func):
    """Wrap ``func`` to handle custom gwpy keywords for a LogNorm colouring
    """
    @wraps(func)
    def decorated_func(*args, **kwargs):
        norm, kwargs = format_norm(kwargs)
        kwargs['norm'] = norm
        return func(*args, **kwargs)
    return decorated_func


def xlim_as_gps(func):
    """Wrap ``func`` to handle pass limit inputs through `gwpy.time.to_gps`
    """
    @wraps(func)
    def wrapped_func(self, left=None, right=None, **kw):
        if right is None and numpy.iterable(left):
            left, right = left
        kw['left'] = left
        kw['right'] = right
        gpsscale = self.get_xscale() in GPS_SCALES
        for key in ('left', 'right'):
            if gpsscale:
                try:
                    kw[key] = numpy.longdouble(str(to_gps(kw[key])))
                except TypeError:
                    pass
        return func(self, **kw)
    return wrapped_func


def restore_grid(func):
    """Wrap ``func`` to preserve the Axes current grid settings.

    Prior to matplotlib 3.7.0 (unreleased ATOW) pcolor() and pcolormesh()
    automatically removed a grid on a set of Axes. This decorator just
    undoes that.
    """
    if matplotlib_version >= "3.7.0":
        return func

    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        try:
            grid = (
                self.xaxis._minor_tick_kw["gridOn"],
                self.xaxis._major_tick_kw["gridOn"],
                self.yaxis._minor_tick_kw["gridOn"],
                self.yaxis._major_tick_kw["gridOn"],
            )
        except KeyError:  # matplotlib < 3.3.3
            grid = (self.xaxis._gridOnMinor, self.xaxis._gridOnMajor,
                    self.yaxis._gridOnMinor, self.yaxis._gridOnMajor)
        # matplotlib >=3.5.0,<3.7.0 presents a warning if you have a grid
        # that it won't be automatically removed, so we forcibly remove it
        # ahead of time, knowing that if we had one, we will restore it
        # in the 'finally' block below.
        self.grid(False)
        try:
            return func(self, *args, **kwargs)
        finally:
            # reset grid
            self.xaxis.grid(grid[0], which="minor")
            self.xaxis.grid(grid[1], which="major")
            self.yaxis.grid(grid[2], which="minor")
            self.yaxis.grid(grid[3], which="major")
    return wrapped_func


def deprecate_c_sort(func):
    """Wrap ``func`` to replace the deprecated ``c_sort`` keyword.

    This was renamed ``sortbycolor``.
    """
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        if "c_sort" in kwargs:
            warnings.warn(
                f"the `c_sort` keyword for {func.__name__} was "
                "renamed `sortbycolor`, this warning will result "
                "in an error in future versions of GWpy",
                DeprecationWarning,
            )
            kwargs.setdefault(
                "sortbycolor",
                kwargs.pop("c_sort"),
            )
        return func(self, *args, **kwargs)
    return wrapped


def _sortby(sortby, *arrays):
    """Sort a set of arrays by the first one (including the first one)
    """
    # try and sort the colour array by value
    sortidx = numpy.asanyarray(sortby, dtype=float).argsort()

    def _sort(arr):
        if arr is None or isinstance(arr, Number):
            return arr
        return numpy.asarray(arr)[sortidx]

    # apply the sorting to each data array, and scatter
    for arr in (sortby,) + arrays:
        yield _sort(arr)


# -- new Axes -----------------------------------------------------------------

class Axes(_Axes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # handle Series in `ax.plot()`
        self._get_lines = PlotArgsProcessor(self)

        # reset data formatters (for interactive plots) to support
        # GPS time display
        self.fmt_xdata = self._fmt_xdata
        self.fmt_ydata = self._fmt_ydata

    @allow_rasterization
    def draw(self, *args, **kwargs):
        labels = {}

        for ax in (self.xaxis, self.yaxis):
            if ax.get_scale() in GPS_SCALES and ax.isDefault_label:
                labels[ax] = ax.get_label_text()
                trans = ax.get_transform()
                epoch = float(trans.get_epoch())
                unit = trans.get_unit_name()
                iso = Time(epoch, format='gps', scale='utc').iso
                utc = iso.rstrip('0').rstrip('.')
                ax.set_label_text(f"Time [{unit}] from {utc} UTC ({epoch!r})")

        try:
            super().draw(*args, **kwargs)
        finally:
            for ax in labels:  # reset labels
                ax.isDefault_label = True

    # -- auto-gps helpers -----------------------

    def _fmt_xdata(self, x):
        if self.get_xscale() in GPS_SCALES:
            return str(to_gps(x))
        return self.xaxis.get_major_formatter().format_data_short(x)

    def _fmt_ydata(self, y):
        if self.get_yscale() in GPS_SCALES:
            return str(to_gps(y))
        return self.yaxis.get_major_formatter().format_data_short(y)

    set_xlim = xlim_as_gps(_Axes.set_xlim)

    def set_epoch(self, epoch):
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

    def get_epoch(self):
        """Return the epoch for the current GPS scale/

        This method will fail if the current X-axis scale isn't one of
        the GPS scales. See :ref:`gwpy-plot-gps` for more details.
        """
        return self.get_xaxis().get_transform().get_epoch()

    # -- overloaded plotting methods ------------

    @deprecate_c_sort
    def scatter(self, x, y, s=None, c=None, **kwargs):
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

    scatter.__doc__ = _Axes.scatter.__doc__.replace(
        'marker :',
        'sortbycolor : `bool`, optional, default: False\n'
        '    Sort scatter points by `c` array value, if given.\n\n'
        'marker :',
    )

    @log_norm
    def imshow(self, array, *args, **kwargs):
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
        array : array-like or PIL image
            The image data.

        *args, **kwargs
            All arguments and keywords are passed to the inherited
            :meth:`~matplotlib.axes.Axes.imshow` method.

        See also
        --------
        matplotlib.axes.Axes.imshow
            for details of the image rendering
        """
        if hasattr(array, "yspan"):  # Array2D
            return self._imshow_array2d(array, *args, **kwargs)

        image = super().imshow(array, *args, **kwargs)
        self.autoscale(enable=None, axis='both', tight=None)
        return image

    def _imshow_array2d(self, array, origin='lower', interpolation='none',
                        aspect='auto', **kwargs):
        """Render an `~gwpy.types.Array2D` using `Axes.imshow`
        """
        # NOTE: If you change the defaults for this method, please update
        #       the docstring for `imshow` above.

        # calculate extent
        extent = tuple(array.xspan) + tuple(array.yspan)
        if self.get_xscale() == 'log' and extent[0] == 0.:
            extent = (1e-300,) + extent[1:]
        if self.get_yscale() == 'log' and extent[2] == 0.:
            extent = extent[:2] + (1e-300,) + extent[3:]
        kwargs.setdefault('extent', extent)

        return self.imshow(array.value.T, origin=origin, aspect=aspect,
                           interpolation=interpolation, **kwargs)

    @restore_grid
    @log_norm
    def pcolormesh(self, *args, **kwargs):
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

        See also
        --------
        matplotlib.axes.Axes.pcolormesh
        """
        if len(args) == 1 and hasattr(args[0], "yindex"):  # Array2D
            return self._pcolormesh_array2d(*args, **kwargs)
        return super().pcolormesh(*args, **kwargs)

    def _pcolormesh_array2d(self, array, *args, **kwargs):
        """Render an `~gwpy.types.Array2D` using `Axes.pcolormesh`
        """
        x = numpy.concatenate((array.xindex.value, array.xspan[-1:]))
        y = numpy.concatenate((array.yindex.value, array.yspan[-1:]))
        xcoord, ycoord = numpy.meshgrid(x, y, copy=False, sparse=True)
        return self.pcolormesh(xcoord, ycoord, array.value.T, *args, **kwargs)

    def hist(self, x, *args, **kwargs):
        x = numpy.asarray(x)

        # re-format weights as array if given as float
        weights = kwargs.get('weights', None)
        if isinstance(weights, Number):
            kwargs['weights'] = numpy.ones_like(x) * weights

        # calculate log-spaced bins on-the-fly
        if (
            kwargs.pop('logbins', False)
            and not numpy.iterable(kwargs.get('bins', None))
        ):
            nbins = kwargs.get('bins', None) or rcParams.get('hist.bins', 30)
            # get range
            hrange = kwargs.pop('range', None)
            if hrange is None:
                try:
                    hrange = numpy.min(x), numpy.max(x)
                except ValueError as exc:
                    if str(exc).startswith('zero-size array'):  # no data
                        exc.args = ('cannot generate log-spaced histogram '
                                    'bins for zero-size array, '
                                    'please pass `bins` or `range` manually',)
                    raise
            # log-scale the axis and extract the base
            if kwargs.get('orientation') == 'horizontal':
                self.set_yscale('log', nonpositive='clip')
                logbase = self.yaxis._scale.base
            else:
                self.set_xscale('log', nonpositive='clip')
                logbase = self.xaxis._scale.base
            # generate the bins
            kwargs['bins'] = numpy.logspace(
                log(hrange[0], logbase), log(hrange[1], logbase),
                nbins+1, endpoint=True)

        return super().hist(x, *args, **kwargs)

    hist.__doc__ = _Axes.hist.__doc__.replace(
        'color :',
        'logbins : boolean, optional\n'
        '    If ``True``, use logarithmically-spaced histogram bins.\n\n'
        '    Default is ``False``\n\n'
        'color :')

    # -- new plotting methods -------------------

    def plot_mmm(self, data, lower=None, upper=None, **kwargs):
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

        See also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        alpha = kwargs.pop('alpha', .1)

        # plot mean
        line, = self.plot(data, **kwargs)
        out = [line]

        # modify keywords for shading
        kwargs.update({
            'label': '',
            'linewidth': line.get_linewidth() / 2,
            'color': line.get_color(),
            'alpha': alpha * 2,
        })

        # plot lower and upper Series
        fill = [data.xindex.value, data.value, data.value]
        for i, bound in enumerate((lower, upper)):
            if bound is not None:
                out.extend(self.plot(bound, **kwargs))
                fill[i+1] = bound.value

        # fill between
        out.append(self.fill_between(
            *fill, alpha=alpha, color=kwargs['color'],
            rasterized=kwargs.get('rasterized', True)))

        return out

    @deprecate_c_sort
    def tile(self, x, y, w, h, color=None,
             anchor='center', edgecolors='face', linewidth=0.8,
             **kwargs):
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

        **kwargs
            Other keywords are passed to
            :meth:`~matplotlib.collections.PolyCollection`

        Returns
        -------
        collection : `~matplotlib.collections.PolyCollection`
            the collection of tiles drawn

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
        if anchor == 'll':
            def _poly(x, y, w, h):
                return ((x, y), (x, y+h), (x+w, y+h), (x+w, y))
        elif anchor == 'lr':
            def _poly(x, y, w, h):
                return ((x-w, y), (x-w, y+h), (x, y+h), (x, y))
        elif anchor == 'ul':
            def _poly(x, y, w, h):
                return ((x, y-h), (x, y), (x+w, y), (x+w, y-h))
        elif anchor == 'ur':
            def _poly(x, y, w, h):
                return ((x-w, y-h), (x-w, y), (x, y), (x, y-h))
        elif anchor == 'center':
            def _poly(x, y, w, h):
                return ((x-w/2., y-h/2.), (x-w/2., y+h/2.),
                        (x+w/2., y+h/2.), (x+w/2., y-h/2.))
        else:
            raise ValueError(f"Unrecognised tile anchor '{anchor}'")

        # build collection
        cmap = kwargs.pop('cmap', rcParams['image.cmap'])
        coll = PolyCollection((_poly(*tile) for tile in zip(x, y, w, h)),
                              edgecolors=edgecolors, linewidth=linewidth,
                              **kwargs)
        if color is not None:
            coll.set_array(color)
            coll.set_cmap(cmap)

        out = self.add_collection(coll)
        self.autoscale_view()
        return out

    # -- overloaded auxiliary methods -----------

    def legend(self, *args, **kwargs):
        # build custom handler to render thick lines by default
        handler_map = kwargs.setdefault("handler_map", dict())
        if isinstance(handler_map, dict):
            handler_map.setdefault(Line2D, HandlerLine2D(6))

        # create legend
        return super().legend(*args, **kwargs)

    legend.__doc__ = _Axes.legend.__doc__.replace(
        "Call signatures",
        """.. note::

   This method uses a custom default legend handler for
   `~matplotlib.lines.Line2D` objects, with increased linewidth relative
   to the upstream :meth:`~matplotlib.axes.Axes.legend` method.
   To disable this, pass ``handler_map=None``, or create and pass your
   own handler class.  See :ref:`gwpy-plot-legend` for more details.

Call signatures""",
    )

    def colorbar(self, mappable=None, **kwargs):
        """Add a `~matplotlib.colorbar.Colorbar` to these `Axes`

        Parameters
        ----------
        mappable : matplotlib data collection, optional
            collection against which to map the colouring, default will
            be the last added mappable artist (collection or image)

        fraction : `float`, optional
            fraction of space to steal from these `Axes` to make space
            for the new axes, default is ``0.`` if ``use_axesgrid=True``
            is given (default), otherwise default is ``.15`` to match
            the upstream matplotlib default.

        **kwargs
            other keyword arguments to be passed to the
            :meth:`Plot.colorbar` generator

        Returns
        -------
        cbar : `~matplotlib.colorbar.Colorbar`
            the newly added `Colorbar`

        See also
        --------
        Plot.colorbar
        """
        fig = self.get_figure()
        if kwargs.get('use_axesgrid', True):
            kwargs.setdefault('fraction', 0.)
        if kwargs.get('fraction', 0.) == 0.:
            kwargs.setdefault('use_axesgrid', True)
        mappable, kwargs = gcbar.process_colorbar_kwargs(
            fig, mappable=mappable, ax=self, **kwargs)
        if isinstance(fig, Plot):
            # either we have created colorbar Axes using axesgrid1, or
            # the user already gave use_axesgrid=False, so we forcefully
            # disable axesgrid here in case fraction == 0., which causes
            # gridspec colorbars to fail.
            kwargs['use_axesgrid'] = False
        return fig.colorbar(mappable, **kwargs)


# override default Axes with this one by registering a projection with the
# same name

register_projection(Axes)


# -- overload Axes.plot() to handle Series ------------------------------------

class PlotArgsProcessor(_process_plot_var_args):
    """This class controls how ax.plot() works
    """
    def __call__(self, *args, **kwargs):
        """Find `Series` data in `plot()` args and unwrap
        """
        newargs = []
        while args:
            # strip first argument
            this, args = args[:1], args[1:]
            # it its a 1-D Series, then parse it as (xindex, value)
            if hasattr(this[0], "xindex") and this[0].ndim == 1:
                this = (this[0].xindex.value, this[0].value)
            # otherwise treat as normal (must be a second argument)
            else:
                this += args[:1]
                args = args[1:]
            # allow colour specs
            if args and isinstance(args[0], str):
                this += args[0],
                args = args[1:]
            newargs.extend(this)

        return super().__call__(*newargs, **kwargs)
