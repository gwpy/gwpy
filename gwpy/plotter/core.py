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

"""Extension of the basic matplotlib Figure for GWpy
"""

import warnings

import numpy

from matplotlib import (backends, figure, pyplot, colors as mcolors,
                        _pylab_helpers)
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.axes import SubplotBase
from matplotlib.cbook import iterable
from matplotlib.ticker import LogLocator

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    from mpl_toolkits.axes_grid import make_axes_locatable

from . import (axes, utils)
from .axes import Axes
from .log import CombinedLogFormatterMathtext
from .decorators import (auto_refresh, axes_method)

__all__ = ['Plot']

try:
    __IPYTHON__
except NameError:
    IPYTHON = False
else:
    IPYTHON = True


def interactive_backend():
    return pyplot.get_backend() in backends.interactive_bk


class Plot(figure.Figure):
    """An extension of the core matplotlib :class:`~matplotlib.figure.Figure`.

    The `Plot` provides a number of methods to simplify generating
    figures from GWpy data objects, and modifying them on-the-fly in
    interactive mode.
    """
    _DefaultAxesClass = Axes

    def __init__(self, *args, **kwargs):
        # pull non-standard keyword arguments
        auto_refresh = kwargs.pop('auto_refresh', False)

        # generated figure, with associated interactivity from pyplot
        super(Plot, self).__init__(*args, **kwargs)
        backend_mod, _, draw_if_interactive, _ = backends.pylab_setup()
        try:
            manager = backend_mod.new_figure_manager_given_figure(1, self)
        except AttributeError:
            canvas = backend_mod.FigureCanvas(self)
            manager = FigureManagerBase(canvas, 1)
        cid = manager.canvas.mpl_connect(
            'button_press_event',
            lambda ev: _pylab_helpers.Gcf.set_active(manager))
        manager._cidgcf = cid
        _pylab_helpers.Gcf.set_active(manager)
        draw_if_interactive()

        # finalise
        self.set_auto_refresh(auto_refresh)
        self.colorbars = []
        self._coloraxes = []

    # -----------------------------------------------
    # core plot operations

    def get_auto_refresh(self):
        """Return this `Plot`s auto-refresh setting
        """
        return self._auto_refresh

    def set_auto_refresh(self, b):
        """Set this `Plot`s auto-refresh setting

        With auto_refresh set to `True`, all modifications of the underlying
        `Axes` will trigger the plot to be re-drawn

        Parameters
        ----------
        b : `True` or `False`
        """
        self._auto_refresh = bool(b)

    def refresh(self):
        """Refresh the current figure
        """
        for cbar in self.colorbars:
            cbar.draw_all()
        self.canvas.draw()

    def show(self, block=None, warn=True):
        """Display the current figure (if possible)

        Parameters
        ----------
        block : `bool`, default: `None`
            open the figure and block until the figure is closed, otherwise
            open the figure as a detached window. If `block=None`, GWpy
            will block if using an interactive backend and not in an
            ipython session.
        warn : `bool`, default: `True`
            if `block=False` is given, print a warning if matplotlib is
            not running in an interactive backend and cannot display the
            figure.

        Notes
        -----
        If blocking is employed, this method calls the
        :meth:`pyplot.show <matplotlib.pyplot.show>` function, otherwise
        the :meth:`~matplotlib.figure.Figure.show` method of this
        `~matplotlib.figure.Figure` is used.
        """
        # if told to block, or using an interactive backend,
        # but not using ipython
        if block or (block is None and interactive_backend() and not IPYTHON):
            return pyplot.show(block=True)
        # otherwise, don't block and just show normally
        else:
            return super(Plot, self).show(warn=warn)

    def save(self, *args, **kwargs):
        """Save the figure to disk.

        All `args` and `kwargs` are passed directly to the savefig
        method of the underlying `matplotlib.figure.Figure`
        self.fig.savefig(*args, **kwargs)
        """
        self.savefig(*args, **kwargs)

    def close(self):
        """Close the plot and release its memory.
        """
        for ax in self.axes:
            ax.cla()
        pyplot.close(self)

    # -----------------------------------------------
    # colour-bar

    @auto_refresh
    def add_colorbar(self, mappable=None, ax=None, location='right',
                     width=0.2, pad=0.1, log=None, label="", clim=None,
                     cmap=None, clip=None, visible=True, axes_class=axes.Axes,
                     **kwargs):
        """Add a colorbar to the current `Axes`

        Parameters
        ----------
        mappable : matplotlib data collection
            collection against which to map the colouring
        ax : :class:`~matplotlib.axes.Axes`
            axes from which to steal space for the colour-bar
        location : `str`, optional, default: 'right'
            position of the colorbar
        width : `float`, optional default: 0.2
            width of the colorbar as a fraction of the axes
        pad : `float`, optional, default: 0.1
            gap between the axes and the colorbar as a fraction of the axes
        log : `bool`, optional, default: `False`
            display the colorbar with a logarithmic scale
        label : `str`, optional, default: '' (no label)
            label for the colorbar
        clim : pair of floats, optional
            (lower, upper) limits for the colorbar scale, values outside
            of these limits will be clipped to the edges
        visible : `bool`, optional, default: `True`
            make the colobar visible on the figure, this is useful to
            make two plots, each with and without a colorbar, but
            guarantee that the axes will be the same size
        **kwargs
            other keyword arguments to be passed to the
            :meth:`~matplotlib.figure.Figure.colorbar` generator

        Returns
        -------
        Colorbar
            the :class:`~matplotlib.colorbar.Colorbar` added to this plot
        """
        # find default layer
        if mappable is None and ax is not None and len(ax.collections):
            mappable = ax.collections[-1]
        elif mappable is None and ax is not None and len(ax.images):
            mappable = ax.images[-1]
        elif (visible is False and mappable is None and
              ax is not None and len(ax.lines)):
            mappable = ax.lines[-1]
        elif mappable is None and ax is None:
            for ax in self.axes[::-1]:
                if hasattr(ax, 'collections') and len(ax.collections):
                    mappable = ax.collections[-1]
                    break
                elif hasattr(ax, 'images') and len(ax.images):
                    mappable = ax.images[-1]
                    break
                elif visible is False and len(ax.lines):
                    mappable = ax.lines[-1]
                    break
        if visible and mappable is None:
            raise ValueError("Cannot determine mappable layer for this "
                             "colorbar")
        elif ax is None:
            raise ValueError("Cannot determine an anchor Axes for this "
                             "colorbar")

        # find default axes
        if not ax:
            ax = mappable.axes

        mappables = ax.collections + ax.images

        # get new colour axis
        divider = make_axes_locatable(ax)
        if location not in ['right', 'top']:
            raise ValueError("'left' and 'bottom' colorbars have not "
                             "been implemented")
        cax = divider.append_axes(location, width, pad=pad,
                                  add_to_figure=visible, axes_class=axes_class)
        self._coloraxes.append(cax)
        if visible:
            self.sca(ax)
        else:
            return

        # set limits
        if not clim:
            clim = mappable.get_clim()
        if log is None:
            log = isinstance(mappable.norm, mcolors.LogNorm)
        if log and clim[0] <= 0.0:
            cdata = mappable.get_array()
            try:
                clim = (cdata[cdata > 0.0].min(), clim[1])
            except ValueError:
                pass
        for m in mappables:
            m.set_clim(clim)

        # set map
        if cmap is not None:
            mappable.set_cmap(cmap)

        # set normalisation
        norm = mappable.norm
        if clip is None:
            clip = norm.clip
        for m in mappables:
            if log and not isinstance(norm, mcolors.LogNorm):
                m.set_norm(mcolors.LogNorm(*mappable.get_clim()))
            elif not log:
                m.set_norm(mcolors.Normalize(*mappable.get_clim()))
            m.norm.clip = clip

        # set log ticks
        if log:
            kwargs.setdefault('ticks', LogLocator(subs=numpy.arange(1, 11)))
            kwargs.setdefault('format', CombinedLogFormatterMathtext())

        # make colour bar
        colorbar = self.colorbar(mappable, cax=cax, ax=ax, **kwargs)

        # set label
        if label:
            colorbar.set_label(label)
        colorbar.draw_all()

        self.colorbars.append(colorbar)
        return colorbar

    # -------------------------------------------
    # GWpy data adding
    #
    # These methods try to guess which axes to add to, otherwise generate
    # a new one

    def add_subplot(self, *args, **kwargs):
        kwargs.setdefault('projection', self._DefaultAxesClass.name)
        return super(Plot, self).add_subplot(*args, **kwargs)
    add_subplot.__doc__ = figure.Figure.add_subplot.__doc__

    def get_axes(self, projection=None):
        """Find all `Axes`, optionally matching the given projection

        Parameters
        ----------
        projection : `str`
            name of axes types to return
        """
        if projection is None:
            return self.axes
        else:
            return [ax for ax in self.axes if ax.name == projection.lower()]

    def _find_axes(self, projection=None):
        """Find the most recently added axes for the given projection

        Raises
        ------
        IndexError
            if no axes for the projection are found
        """
        try:
            return self.get_axes(projection)[-1]
        except IndexError:
            if projection:
                raise IndexError("No '%s' Axes found in this Plot"
                                 % projection)
            else:
                raise IndexError("No Axes found in this Plot")

    def _increment_geometry(self):
        """Try to determine the geometry to use for a new Axes

        Raises
        ------
        ValueError
            if geometry is ambiguous
        """
        if not len(self.axes):
            return (1, 1, 1)
        # get bottom axes
        try:
            ca = [ax for ax in self.axes if isinstance(ax, SubplotBase)][-1]
        except IndexError:
            ca = self.gca()
        current = ca.get_geometry()
        shape = current[:2]
        pos = current[2]
        num = shape[0] * shape[1]
        # if space left in this set
        if pos < num:
            return (shape[0], shape[1], pos+1)
        # or add a new column
        elif shape[1] > 1 and shape[0] == 1:
            return (1, shape[1] + 1, pos+1)
        # otherwise add a new row
        else:
            return (shape[0] + 1, 1, pos+1)

    def _add_new_axes(self, projection, **kwargs):
        # get new geomtry
        geometry = self._increment_geometry()
        # make new axes
        newax = self.add_subplot(*geometry, projection=projection, **kwargs)
        # update geometry for previous axes
        nrows = geometry[0]
        ncols = geometry[1]
        i = 0
        for ax in self.axes[:-1]:
            if isinstance(ax, SubplotBase):
                i += 1
                ax.change_geometry(nrows, ncols, i)
        return newax

    @auto_refresh
    def _plot(self, x, y, *args, **kwargs):
        """Add a line to the current plot

        Parameters
        ----------
        x : array-like
            x positions of the line points (in axis coordinates)
        y : array-like
            y positions of the line points (in axis coordinates)
        projection : `str`, optional, default: `'timeseries'`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            additional keyword arguments passed directly on to
            the axes :meth:`~matplotlib.axes.Axes.plot` method.

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        # get axes options
        projection = kwargs.pop('projection', None)
        ax = kwargs.pop('ax', None)
        newax = kwargs.pop('newax', False)
        sharex = kwargs.pop('sharex', None)
        sharey = kwargs.pop('sharey', None)

        # set kwargs
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("markersize", 0)

        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection,
                                    sharex=sharex, sharey=sharey)
        # plot on axes
        return ax.plot(numpy.asarray(x), numpy.asarray(y), **kwargs)[0]

    @auto_refresh
    def _scatter(self, x, y, projection=None, ax=None, newax=False,
                 **kwargs):
        """Internal `Plot` method to scatter onto the most
        favourable `Axes`

        Parameters
        ----------
        x : array-like
            x positions of the line points (in axis coordinates)
        y : array-like
            y positions of the line points (in axis coordinates)
        projection : `str`, optional, default: `None`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs.
            other keyword arguments passed to the
            :meth:`matplotlib.axes.Axes.scatter` function

        Returns
        -------
        Collection
            the :class:`~matplotlib.collections.Collection` for this
            scatter layer
        """
        # set kwargs
        kwargs.setdefault("s", 20)
        sharex = kwargs.pop('sharex', None)
        sharey = kwargs.pop('sharey', None)

        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection,
                                    sharex=sharex, sharey=sharey)
        # plot on axes
        return ax.scatter(numpy.asarray(x), numpy.asarray(y), **kwargs)

    @auto_refresh
    def _imshow(self, image, projection=None, ax=None, newax=False, **kwargs):
        """Internal `Plot` method to imshow onto the most
        favourable `Axes`

        Parameters
        ----------
        x : array-like
            x positions of the line points (in axis coordinates)
        y : array-like
            y positions of the line points (in axis coordinates)
        projection : `str`, optional, default: `None`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs.
            other keyword arguments passed to the
            :meth:`matplotlib.axes.Axes.imshow` function

        Returns
        -------
        Collection
            the :class:`~matplotlib.image.AxesImage` for this image
        """
        sharex = kwargs.pop('sharex', None)
        sharey = kwargs.pop('sharey', None)
        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection,
                                    sharex=sharex, sharey=sharey)
        # plot on axes
        return ax.imshow(image, **kwargs)

    @auto_refresh
    def add_line(self, x, y, *args, **kwargs):
        """Add a line to the current plot

        Parameters
        ----------
        x : array-like
            x positions of the line points (in axis coordinates)
        y : array-like
            y positions of the line points (in axis coordinates)
        projection : `str`, optional, default: `None`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            additional keyword arguments passed directly on to
            the axes :meth:`~matplotlib.axes.Axes.plot` method.

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        return self._plot(x, y, *args, **kwargs)

    @auto_refresh
    def add_scatter(self, x, y, **kwargs):
        """Add a set or points to the current plot

        Parameters
        ----------
        x : array-like
            x-axis data points
        y : array-like
            y-axis data points
        projection : `str`, optional, default: `None`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs.
            other keyword arguments passed to the
            :meth:`matplotlib.axes.Axes.scatter` function

        Returns
        -------
        Collection
            the :class:`~matplotlib.collections.Collection` for this
            scatter layer
        """
        return self._scatter(x, y, **kwargs)

    @auto_refresh
    def add_image(self, image, projection=None, ax=None, newax=False,
                  **kwargs):
        """Add a 2-D image to this plot

        Parameters
        ----------
        image : `numpy.ndarray`
            2-D array of data for the image
        **kwargs
            other keyword arguments are passed to the
            :meth:`matplotlib.axes.Axes.imshow` function

        Returns
        -------
        image : :class:`~matplotlib.image.AxesImage`
        """
        return self._imshow(image, projection=projection, ax=ax, newax=newax,
                            **kwargs)

    @auto_refresh
    def add_timeseries(self, timeseries, projection='timeseries',
                       ax=None, newax=False, sharex=None, sharey=None,
                       **kwargs):
        """Add a :class:`~gwpy.timeseries.TimeSeries` trace to this plot

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.TimeSeries`
            the TimeSeries to display
        projection : `str`, optional, default: `'timeseries'`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        return self.add_array(timeseries, 'timeseries', ax=ax, newax=newax,
                              sharex=sharex, sharey=sharey, **kwargs)

    @auto_refresh
    def add_frequencyseries(self, spectrum, projection='spectrum', ax=None,
                            newax=False, sharex=None, sharey=None, **kwargs):
        """Add a `~gwpy.frequencyseries.FrequencySeries` trace to this plot

        Parameters
        ----------
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            the `FrequencySeries` to display
        projection : `str`, optional, default: `'frequencyseries'`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        return self.add_array(spectrum, 'frequencyseries', ax=ax,
                              newax=newax, sharex=sharex, sharey=sharey,
                              **kwargs)

    @auto_refresh
    def add_spectrum(self, *args, **kwargs):
        warnings.warn("This method was renamed add_frequencyseries()",
                      DeprecationWarning)
        return self.add_frequencyseries(*args, **kwargs)
    add_spectrum.__doc__ = add_frequencyseries.__doc__

    @auto_refresh
    def add_spectrogram(self, spectrogram, projection='timeseries',
                        ax=None, newax=False, sharex=None, sharey=None,
                        **kwargs):
        """Add a :class:`~gwpy.spectrogram.core.Spectrogram` trace to
        this plot

        Parameters
        ----------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            the `Spectrogram` to display
        projection : `str`, optional, default: `timeseries`
            name of the Axes projection on which to plot
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        return self.add_array(spectrogram, 'timeseries', ax=ax, newax=newax,
                              sharex=sharex, sharey=sharey, **kwargs)

    @auto_refresh
    def add_array(self, array, projection, ax=None, newax=False,
                  sharex=None, sharey=None, **kwargs):
        """Add a :class:`~gwpy.data.array.Array` to this plot

        Parameters
        ----------
        array : :class:`~gwpy.data.array.Array`
            the `Array` to display
        projection : `str`
        ax : :class:`~gwpy.plotter.Axes`
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Artist : :class:`~matplotlib.artist.Artist`
            the layer return from the relevant plotting function
        """
        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection,
                                    sharex=sharex, sharey=sharey)
        # plot on axes
        return ax.plot(array, **kwargs)

    def add_dataqualityflag(self, flag, projection=None, ax=None, newax=False,
                            sharex=None, sharey=None, **kwargs):
        """Add a :class:`~gwpy.segments.flag.DataQualityFlag` to this plot

        Parameters
        ----------
        flag : :class:`~gwpy.segments.flag.DataQualityFlag`
            the `DataQualityFlag` to display

        """
        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection,
                                    sharex=sharex, sharey=sharey)
        # plot on axes
        return ax.plot(flag, **kwargs)

    # -------------------------------------------
    # Plot legend

    @auto_refresh
    def add_legend(self, *args, **kwargs):
        """Add a legend to this `Plot` on the most favourable `Axes`

        All non-keyword `args` and `kwargs` are passed directly to the
        :meth:`~matplotlib.axes.Axes.legend` generator

        Returns
        -------
        Legend
            the :class:`~matplotlib.legend.Legend` for this plot
        """
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = self._find_axes()
        return ax.legend(*args, **kwargs)

    # -------------------------------------------
    # Convenience methods for single-axes plots
    #
    # The majority of methods in this section are decorated to call the
    # equivalent method of the current Axes, and so contain no actual code

    @axes_method
    def get_xlim(self):
        pass
    get_xlim.__doc__ = axes.Axes.get_xlim.__doc__

    @auto_refresh
    @axes_method
    def set_xlim(self, *args, **kwargs):
        pass
    set_xlim.__doc__ = axes.Axes.set_xlim.__doc__

    xlim = property(fget=get_xlim, fset=set_xlim,
                    doc='x-axis limits for the current axes')

    @axes_method
    def get_ylim(self):
        pass
    get_ylim.__doc__ = axes.Axes.get_ylim.__doc__

    @auto_refresh
    @axes_method
    def set_ylim(self, *args, **kwargs):
        pass
    set_ylim.__doc__ = axes.Axes.set_ylim.__doc__

    ylim = property(fget=get_ylim, fset=set_ylim,
                    doc='y-axis limits for the current axes')

    @axes_method
    def get_xlabel(self):
        pass
    get_xlabel.__doc__ = axes.Axes.get_xlabel.__doc__

    @axes_method
    @auto_refresh
    def set_xlabel(self, *args, **kwargs):
        pass
    set_xlabel.__doc__ = axes.Axes.set_xlabel.__doc__

    xlabel = property(fget=get_xlabel, fset=set_xlabel,
                      doc='x-axis label for the current axes')

    @axes_method
    def get_ylabel(self):
        pass
    get_ylabel.__doc__ = axes.Axes.get_ylabel.__doc__

    @auto_refresh
    @axes_method
    def set_ylabel(self, *args, **kwargs):
        pass
    set_ylabel.__doc__ = axes.Axes.set_ylabel.__doc__

    ylabel = property(fget=get_ylabel, fset=set_ylabel,
                      doc='y-axis label for the current axes')

    @axes_method
    def get_title(self):
        pass
    get_title.__doc__ = axes.Axes.get_title.__doc__

    @auto_refresh
    @axes_method
    def set_title(self, *args, **kwargs):
        pass
    set_title.__doc__ = axes.Axes.set_title.__doc__

    title = property(fget=get_title, fset=set_title,
                     doc='title for the current axes')

    @axes_method
    def get_xscale(self):
        pass
    get_xscale.__doc__ = axes.Axes.get_xscale.__doc__

    @auto_refresh
    @axes_method
    def set_xscale(self, *args, **kwargs):
        pass
    set_xscale.__doc__ = axes.Axes.set_xscale.__doc__

    @property
    def logx(self):
        """View x-axis in logarithmic scale
        """
        return self.get_xscale() == 'log'

    @logx.setter
    @auto_refresh
    def logx(self, log):
        if not self.logx and bool(log):
            self.set_xscale('log')
        elif not self.logx and bool(log):
            self.set_xscale('linear')

    @axes_method
    def get_yscale(self):
        pass
    get_yscale.__doc__ = axes.Axes.get_yscale.__doc__

    @auto_refresh
    @axes_method
    def set_yscale(self, *args, **kwargs):
        pass
    set_yscale.__doc__ = axes.Axes.set_yscale.__doc__

    @property
    def logy(self):
        """View y-axis in logarithmic scale
        """
        return self.get_yscale() == 'log'

    @logy.setter
    @auto_refresh
    def logy(self, log):
        if not self.logy and bool(log):
            self.set_yscale('log')
        elif not self.logy and bool(log):
            self.set_yscale('linear')

    @axes_method
    def html_map(self, filename, data=None, **kwargs):
        pass
    html_map.__doc__ = axes.Axes.html_map.__doc__

    # -----------------------------------------------
    # utilies

    @staticmethod
    def _get_axes_data(inputs, sep=False):
        """Determine the number of axes from the input args to this `Plot`

        Parameters
        ----------
        inputs : `list` of array-like data sets
            a list of data arrays, or a list of lists of data sets
        sep : `bool`, optional
            plot each set of data on a separate `Axes`

        Returns
        -------
        axesdata : `list` of lists of array-like data
            a `list` with one element per required `Axes` containing the
            array-like data sets for those `Axes`

        Notes
        -----
        The logic for this method is as follows:

        - if a `list` of data arrays are given, and `sep=False`, use 1 `Axes`
        - if a `list` of data arrays are given, and `sep=True`, use N `Axes,
          one for each data array
        - if a nested `list` of data arrays are given, ignore `sep` and
          use one `Axes` for each element in the top list.

        For example:

            >>> Plot._get_naxes([data1, data2], sep=False)
            [[data1, data2]]
            >>> Plot._get_naxes([data1, data2], sep=True)
            [[data1], [data2]]
            >>> Plot._get_naxes([[data1, data2], data3])
            [[data1, data2], [data3]]
        """
        # if not given list, default to 1
        if not iterable(inputs):
            return [inputs]
        elif not len(inputs):
            return []
        # if given a nested list of data, multiple axes are required
        if any([isinstance(x, (list, tuple, dict)) for x in inputs]):
            sep = True
        # build list of lists
        out = []
        if not sep:
            out.append([])
        for x in inputs:
            # if not sep, each element of inputs is a data set
            if not sep:
                out[0].append(x)
            # otherwise if this element is a list already, that's fine
            elif isinstance(x, (list, tuple)):
                out.append(x)
            elif isinstance(x, dict):
                out.append(x.values())
            else:
                out.append([x])
        return out

    @staticmethod
    def _parse_kwargs(kwargs):
        """Separate input kwargs dict into axes, and artist params

        The assumption is that all remaining kwargs should be passed to
        the underlying `Plot` constructor

        Parameters
        ----------
        kwargs : `dict`
            `dict` of input keyword arguments for `Plot`

        Returns
        -------
        axargs, artistargs : `list` of `dict`
            separated kwarg `dict` for the the axes and the artists
        """
        separatedargs = []
        for arglist in [utils.AXES_PARAMS, utils.ARTIST_PARAMS]:
            separatedargs.append(dict())
            for key in arglist:
                if key in kwargs:
                    separatedargs[-1][key] = kwargs.pop(key)
        return separatedargs
