# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Core classes for building plots in GWpy
"""

import numpy

from matplotlib import (colors as mcolors, ticker as mticker, pyplot as mpl)
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    from mpl_toolkits.axes_grid import make_axes_locatable

from . import (tex, ticks, axis, layer, IS_INTERACTIVE)
from .decorators import auto_refresh

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Nick Fotopoulos"
__version__ = version.version


class Plot(object):
    """A basic class to describe any plot you might want to make.

    Any instance of Plot will have the following attributes

        - `fig`: the `matplotlib.figure.Figure` instance
        - `ax`: the `matplotlib.axes.Axes` instance representing
                the plottable area.

    A new plot can be geneated as follows::

    >>> plot = Plot()
    >>> plot.add_line([1,2,3,4,5,6,7,8], [3,4,3,2,3,4,3,2])
    >>> plot.show()

    Parameters
    ----------
    figure : :class:`~matplotlib.figure.Figure`, optional, default: `None`
        the parent figure for this new plot
    auto_refresh : `bool`, optional, default: `True`
        update the interactive plot whenever any changes are made to
        layers, labels, or limits
    **kwargs
        other keyword arguments are passed to the generator for a new
        :class:`~matplotlib.figure.Figure`

    Returns
    -------
    plot
        a new `Plot`

    Attributes
    ----------
    xlabel
    ylabel
    colorlabel
    title
    subtitle
    xlim
    ylim
    colorlim
    logx
    logy
    logcolor

    Methods
    -------
    add_bars
    add_colorbar
    add_image
    add_legend
    add_line
    add_patch
    add_rectangle
    add_scatter
    add_spectrogram
    add_spectrum
    add_timeseries
    close
    refresh
    save
    show
    """
    __slots__ = ['xlabel', 'ylabel', 'colorlabel', 'title', 'subtitle',
                 'xlim', 'ylim', 'colorlim', 'logx', 'logy', 'logcolor',
                 '_figure', '_axes', '_layers', '_auto_refresh', 'colorbar']

    def __init__(self, figure=None, auto_refresh=IS_INTERACTIVE, **kwargs):
        """Initialise a new plot.
        """
        # generate figure
        if figure:
            self._figure = figure
        else:
            self._figure = mpl.figure(**kwargs)

        # generate the axes
        self._axes = self.figure.gca()

        # record layers
        self._layers = layer.LayerCollection()

        # set auto refresh
        self._auto_refresh = auto_refresh
        if self._auto_refresh:
            self.show()

    # -----------------------------------------------
    # basic properties

    @property
    def figure(self):
        """This plot's underlying :class:`~matplotlib.figure.Figure`
        """
        return self._figure

    @property
    def axes(self):
        """This plot's :class:`~matplotlib.axes.Axes`
        """
        return self._axes

    @property
    def legend(self):
        """This plot's :class:`~matplotlib.legend.Legend`
        """
        return self.axes.legend_

    # -----------------------------------------------
    # text properties

    # x-axis label
    @property
    def xlabel(self):
        """Label for the x-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.axes.xaxis.label
    @xlabel.setter
    @auto_refresh
    def xlabel(self, text):
        if isinstance(text, basestring):
            self.axes.set_xlabel(text)
        else:
            self.axes.xaxis.label = text
    @xlabel.deleter
    @auto_refresh
    def xlabel(self):
        self.axes.set_xlabel("")

    # y-axis label
    @property
    def ylabel(self):
        """Label for the y-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.axes.yaxis.label
    @ylabel.setter
    @auto_refresh
    def ylabel(self, text):
        if isinstance(text, basestring):
            self.axes.set_ylabel(text)
        else:
            self.axes.yaxis.label = text
    @ylabel.deleter
    @auto_refresh
    def ylabel(self):
        self.axes.set_ylabel("")

    # colour-bar label
    @property
    def colorlabel(self):
        """Label for the colour-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self._colorbar.ax.yaxis.label
    @colorlabel.setter
    @auto_refresh
    def colorlabel(self, text):
        self._colorbar.ax.set_ylabel(text)
    @colorlabel.deleter
    @auto_refresh
    def colorlabel(self):
        self.colorlabel = ""

    # title
    def set_title(self, text, **kwargs):
        self.figure.suptitle(text, **kwargs)
    set_title.__doc__ = mpl.Figure.suptitle.__doc__

    @property
    def title(self):
        """Title for this figure

        :type: :class:`~matplotlib.text.Text`

        Notes
        -----
        The :attr:`Plot.title` attribute is set as the title for the
        enclosing :class:`~matplotlib.figure.Figure` and is centred on that
        frame, rather than the :class:`~matplotlib.axes.AxesSubPlot`
        """
        return self.figure._suptitle

    @title.setter
    @auto_refresh
    def title(self, text):
        self.set_title(text)

    @title.deleter
    @auto_refresh
    def title(self):
        self.title = ""

    # sub-title
    def set_subtitle(self, text, **kwargs):
        self.axes.set_title(text, **kwargs)
    set_subtitle.__doc__ = mpl.Axes.set_title.__doc__

    @property
    def subtitle(self):
        """Sub-title for this figure

        :type: :class:`~matplotlib.text.Text`

        Notes
        -----
        The :attr:`Plot.subtitle` attribute is set as the title for the
        enclosing :class:`~matplotlib.axes.AxesSubPlot` and is centred on that
        frame, rather than the parent :class:`~matplotlib.figure.Figure`
        """
        return self.axes.title
    @subtitle.setter
    @auto_refresh
    def subtitle(self, text):
        if isinstance(text, basestring):
            self.set_subtitle(text)
        else:
            self.axes.title = text
    @subtitle.deleter
    @auto_refresh
    def subtitle(self):
        self.subtitle = ""

    # -----------------------------------------------
    # limit properties

    @property
    def xlim(self):
        """Limits for the x-axis

        :type: `tuple`
        """
        return self.axes.get_xlim()
    @xlim.setter
    @auto_refresh
    def xlim(self, limits):
        self.axes.set_xlim(*limits)
    @xlim.deleter
    @auto_refresh
    def xlim(self):
        self.axes.relim()
        self.axes.autoscale_view(scalex=True, scaley=False)

    @property
    def ylim(self):
        """Limits for the y-axis

        :type: `tuple`
        """
        return self.axes.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, limits):
        self.axes.set_ylim(*limits)
    @ylim.deleter
    def ylim(self):
        self.axes.relim()
        self.axes.autoscale_view(scalex=False, scaley=True)

    @property
    def colorlim(self):
        """Limits for the colour-axis

        :type: `tuple`
        """
        return self._colorbar.get_clim()
    @colorlim.setter
    @auto_refresh
    def colorlim(self, limits):
        self._colorbar.set_clim(*limits)
        self._colorbar.draw_all()

    # -----------------------------------------------
    # scale properties

    @property
    def logx(self):
        """Display the x-axis with a logarithmic scale

        :type: `bool`
        """
        return self.axes.get_xscale() == "log"
    @logx.setter
    @auto_refresh
    def logx(self, log):
        self.axes.set_xscale(log and "log" or "linear")

    @property
    def logy(self):
        """Display the y-axis with a logarithmic scale

        :type: `bool`
        """
        return self.axes.get_yscale() == "log"
    @logy.setter
    @auto_refresh
    def logy(self, log):
        self.axes.set_yscale(log and "log" or "linear")

    @property
    def logcolor(self):
        """Display the colour-axis with a logarithmic scale

        :type: `bool`
        """
        return self._logcolor
    @logcolor.setter
    @auto_refresh
    def logcolor(self, log):
        if hasattr(self, '_logcolor') and self._logcolor == log:
            return
        if hasattr(self, '_colorbar'):
            clim = self._colorbar.get_clim()
            self.figure.delaxes(self._colorbar.ax)
            del self._colorbar
            self.add_colorbar(log=log, clim=clim)
        else:
            self._logcolor = log
            for mappable in self._layers.values():
                if hasattr(mappable, 'norm'):
                    if log:
                        mappable.set_norm(mcolors.LogNorm(*mappable.get_clim()))
                    else:
                        mappable.set_norm(
                            mcolors.Normalize(*mappable.get_clim()))

    # -----------------------------------------------
    # core plot operations

    def refresh(self):
        """Refresh the current figure
        """
        self.figure.canvas.draw()

    def show(self):
        """Display the current figure
        """
        self.figure.patch.set_alpha(0.0)
        self.figure.show()

    def save(self, *args, **kwargs):
        """Save the figure to disk.

        All `args` and `kwargs` are passed directly to the savefig
        method of the underlying `matplotlib.figure.Figure`
        self.fig.savefig(*args, **kwargs)
        """
        self.figure.savefig(*args, **kwargs)

    def close(self):
        """Close the plot and release its memory.
        """
        mpl.close(self.figure)

    # -----------------------------------------------
    # add layers

    @auto_refresh
    def add_line(self, x, y, **kwargs):
        """Add a line to the current plot

        Parameters
        ----------
        x : array-like
            x positions of the line points (in axis coordinates)
        y : array-like
            y positions of the line points (in axis coordinates)
        **kwargs
            additional keyword arguments passed directly on to
            the axes :meth:`~matplotlib.axes.Axes.plot` method.

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        # set kwargs
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("linewidth", 1)
        kwargs.setdefault("markersize", 0)
        # generate layer
        l = self.axes.plot(numpy.asarray(x), numpy.asarray(y), **kwargs)[0]
        self._layers.add(l)
        return l

    @auto_refresh
    def add_scatter(self, x, y, **kwargs):
        """Add a set or points to the current plot

        Parameters
        ----------
        x : array-like
            x-axis data points
        y : array-like
            y-axis data points
        **kwargs.
            other keyword arguments passed to the
            :meth:`matplotlib.axes.Axes.scatter` function

        Returns
        -------
        Collection
            the :class:`~matplotlib.collections.Collection` for this
            scatter layer
        """
        kwargs.setdefault("s", 20)
        l = self.axes.scatter(x, y, **kwargs)
        self._layers.add(l)
        return l

    @auto_refresh
    def add_bars(self, left, height, width=0.8, bottom=None, **kwargs):
        """Add a set of bars to the current plot

        Parameters
        ----------
        left : array-like
            left edge points for each bar
        height : array-like
            height of each bar
        width : `float` or array-like, optional, default: 0.8
            the width(s) of the bars
        bottom : scalar of array-like, optional, default: `None`
            the bottom(s) of the bars
        **kwargs
            other keyword arguments are passed to the
            :meth:`matplotlib.axes.Axes.bar` function

        Returns
        -------
        Container
            :class:`~matplotlib.container.BarContainer` containing a
            sequence of :class:`~matplotlib.patches.Rectange` bars
        """
        l = self.axes.bar(left, height, width=width, bottom=None, **kwargs)
        self._layers.add(l)
        return l

    @auto_refresh
    def add_rectangle(self, xll, yll, width, height, **kwargs):
        """Add a rectangle to the current plot

        Parameters
        ----------
        xll : `float`
            lower-left x-coordinate for the rectangle
        yll : `float`
            lower-left y-coordinate for the rectangle
        width : `float`
            rectangle width
        height : `float`
            rectangle height
        **kwargs
            other keyword arguments passed directly to the
            :class:`matplotlib.patches.Rectangle` constructor

        Returns
        -------
        Patch
            the :class:`~matplotlib.patches.Patch` for this rectangle
        """
        p = mpl.Rectangle((xll, yll), width, height, **kwargs)
        l = self.add_patch(p, layer=layer, **kwargs)
        self._layers.add(l)
        return l

    @auto_refresh
    def add_patch(self, patch, **kwargs):
        """Add a patch to the current plot

        Parameters
        ----------
        patch : :class:`~matplotlib.patches.Patch`
            patch to add to this figure
        **kwargs
            keyword arguments passed directly to the
            :meth:`~matplotlib.axes.Axes.add_patch` function

        Returns
        -------
        Patch
            the input :class:`~matplotlib.patches.Patch`
        """
        l = self.axes.add_patch(patch, **kwargs)
        self._layers.add(l)
        return l

    @auto_refresh
    def add_image(self, image, **kwargs):
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
        im = self.axes.imshow(image, **kwargs)
        self._layers.add(im)
        return im

    # -----------------------------------------------
    # Legend

    @auto_refresh
    def add_legend(self, *args, **kwargs):
        """Add a legend to the figure

        All non-keyword `args` and `kwargs` are passed directly to the
        :meth:`~matplotlib.axes.Axes.legend` generator

        Returns
        -------
        Legend
            the :class:`~matplotlib.legend.Legend` for this plot
        """
        alpha = kwargs.pop("alpha", 0.8)
        linewidth = kwargs.pop("linewidth", 8)
        legend = self.axes.legend(*args, **kwargs)
        legend.set_alpha(alpha)
        [l.set_linewidth(linewidth) for l in legend.get_lines()]
        return self.legend

    # -----------------------------------------------
    # colour-bar

    @auto_refresh
    def add_colorbar(self, layer=None, location='right', width=0.2, pad=0.1,
                     log=False, label="", clim=None, visible=True, **kwargs):
        """Add a colorbar to the current plot

        The colorbar will be added to modify an existing layer on the
        figure.

        Parameters
        ----------
        layer : `str`, optional
            name of layer onto which to attach the colorbar, defaults to
            the best match
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
        # remove old colorbar
        if hasattr(self, 'colorbar'):
            self.figure.delaxes(self.colorbar.ax)
            del self.colorbar

        # find default layer
        if layer:
            mappable = self._layers[layer]
        else:
            mappable = self._layers.colorlayer

        # get new colour axis
        divider = make_axes_locatable(self.axes)
        if location == 'right':
            cax = divider.new_horizontal(size=width, pad=pad)
        elif location == 'top':
            cax = divider.new_vertical(size=width, pad=pad)
        else:
            raise ValueError("'left' and 'bottom' colorbars have not "
                             "been implemented")
        if visible:
            divider._fig.add_axes(cax)
        else:
            return

        # set limits
        if not clim:
            clim = mappable.get_clim()
        if log and clim[0] <= 0.0:
            cdata = mappable.get_array()
            try:
                clim = (cdata[cdata>0.0].min(), clim[1])
            except ValueError:
                pass
        mappable.set_clim(clim)

        # set tick format (including sub-ticks for log scales)
        if mpl.rcParams["text.usetex"]:
            if log and abs(float.__sub__(*numpy.log10(clim))) >= 2:
                func = lambda x,pos: (mticker.is_decade(x) and
                                  '$%s$' % tex.float_to_latex(x, '%.4g') or ' ')
            else:
                func = lambda x,pos: '$%s$' % tex.float_to_latex(x, '% .4g')
            kwargs.setdefault('format', mticker.FuncFormatter(func))

        # set log scale
        if log:
            mappable.set_norm(mcolors.LogNorm(*mappable.get_clim()))
        else:
            mappable.set_norm(mcolors.Normalize(*mappable.get_clim()))
        self.logcolor = log

        # set tick locator
        if log:
            kwargs.setdefault('ticks',
                              mticker.LogLocator(subs=numpy.arange(1,11)))

        # make colour bar
        self.colorbar = self.figure.colorbar(mappable, cax=cax, **kwargs)

        # set label
        if label:
            self.colorbar.set_label(label)
        self.colorbar.draw_all()

        return self.colorbar

    # -----------------------------------------------
    # shortcuts for data-types

    def add_timeseries(self, timeseries, **kwargs):
        """Add a :class:`~gwpy.timeseries.core.TimeSeries` trace to this plot

        Parameters
        ----------
        timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
            the TimeSeries to display
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        kwargs.setdefault('label', timeseries.name)
        return self.add_line(timeseries.times, timeseries, **kwargs)

    def add_spectrum(self, spectrum, **kwargs):
        """Add a :class:`~gwpy.spectrum.core.Spectrum` trace to this plot

        Parameters
        ----------
        spectum : :class:`~gwpy.spectrum.core.Spectrum`
            the Spectrum to display
        **kwargs
            other keyword arguments for the `Plot.add_line` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        kwargs.setdefault('label', spectrum.name)
        return self.add_line(spectrum.frequencies, spectrum, **kwargs)

    def add_spectrogram(self, spectrogram, **kwargs):
        """Add a :class:`~gwpy.spectrogram.core.Spectrogram` trace
        to this plot

        Parameters
        ----------
        spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
            the Spectrogram to display
        **kwargs
            other keyword arguments for the `Plot.add_image` function

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer
        """
        kwargs.setdefault('label', spectrogram.name)
        im = spectrogram.data.T
        nf, nt = im.shape
        freqs = spectrogram.get_frequencies()
        extent = (kwargs.pop('extent', None) or 
                  [spectrogram.epoch.gps, (spectrogram.epoch.gps +
                                           float(nt*spectrogram.dt)),
                   freqs.data.min(), freqs.data.max()])
        self.add_image(im, extent=extent, **kwargs)

    def transform_axis(self, axis, func):
        for layer in self._layers.values():
            ticks.transform(layer, axis, func)
        if axis == "x":
            self.axes.set_xlim(map(func, self.axes.get_xlim()))
            self.axes.xaxis.set_data_interval(
                *map(func, self.axes.xaxis.get_data_interval()), ignore=True)
        else:
            self.axes.set_ylim(map(func, self.axes.get_ylim()))
            self.axes.yaxis.set_data_interval(
                *map(func, self.axes.yaxis.get_data_interval()), ignore=True)

    # -----------------------------------------------
    # utilities

    @auto_refresh
    def add_label_unit(self, unit, axis="x"):
        attr = "%slabel" % axis
        label = getattr(self, attr).get_text()
        if not label:
            label = unit.__doc__
        if mpl.rcParams.get("text.usetex", False):
            unitstr = tex.unit_to_latex(unit)
        else:
            unitstr = unit.to_string()
        if label:
            setattr(self, attr, "%s (%s)" % (label, unitstr))
        else:
            setattr(self, attr, unitstr)

    @auto_refresh
    def set_xaxis_format(self, format_, **kwargs):
        axis.set_axis_format(self.axes.xaxis, format_, **kwargs)

    @auto_refresh
    def set_yaxis_format(self, format_, **kwargs):
        axis.set_axis_format(self.axes.yaxis, format_, **kwargs)
