# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Core classes for building plots in GWpy
"""

import numpy

from matplotlib import (colors, ticker, pyplot as mpl)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lal import (git_version, gpstime)

from . import (tex, ticks, axis)
from .decorators import auto_refresh

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Nick Fotopoulos"
__version__ = version.version


class BasicPlot(object):
    """A basic class to describe any plot you might want to make.

    It provides basic initialization, a savefig method, and a close
    method.
    It is up to developers to subclass BasicPlot and fill in the
    add_content() and finalize() methods.

    Any instance of BasicPlot will have the following attributes

        - `fig`: the `matplotlib.figure.Figure` instance
        - `ax`: the `matplotlig.axes.AxesSubplot` instance representing
                the plottable area.
    """
    __slots__ = ['xlim', 'ylim', 'colorlim', 'logx', 'logy', 'logcolor',
                 'xlabel', 'ylabel', 'colorlabel']

    def __init__(self, figure=None, subplot=False, auto_refresh=False,
                 **kwargs):
        """Initialise a new plot.

        All other keyword arguments are passed to
        `matplotlib.pyplot.figure`.

        @returns a new BasicPlot instance
        """
        # generate figure
        if figure:
            self._figure = figure
        else:
            self._figure = mpl.figure(**kwargs)

        # generate the axes
        self._ax = self._figure.gca()

        # set instance attributes
        self._xformat = None
        self._yformat = None
        self._logcolor = False

        # set layers
        self._layers = dict()
        self._line_layers = []
        self._scatter_layers = []
        self._patch_layers = []
        self._bar_layers = []
        self._image_layers = []

        self._auto_refresh = auto_refresh
        if self._auto_refresh:
            self.show()

    # -----------------------------------------------
    # property definitions

    @property
    def xlabel(self):
        return self._ax.get_xlabel()
    @xlabel.setter
    @auto_refresh
    def xlabel(self, text):
        self._ax.set_xlabel(text)
    @xlabel.deleter
    @auto_refresh
    def xlabel(self):
        self._ax.set_xlabel("")

    @property
    def ylabel(self):
        return self._ax.get_ylabel()
    @ylabel.setter
    @auto_refresh
    def ylabel(self, text):
        self._ax.set_ylabel(text)
    @ylabel.deleter
    @auto_refresh
    def ylabel(self):
        self._ax.set_ylabel("")

    @property
    def colorlabel(self):
        return self._colorbar.ax.get_ylabel()
    @colorlabel.setter
    @auto_refresh
    def colorlabel(self, text):
        self._colorbar.ax.set_ylabel(text)
    @colorlabel.deleter
    @auto_refresh
    def colorlabel(self):
        self.colorlabel = ""

    @property
    def xlim(self):
        return self._ax.get_xlim()
    @xlim.setter
    @auto_refresh
    def xlim(self, limits):
        self._ax.set_xlim(*limits)
    @xlim.deleter
    @auto_refresh
    def xlim(self):
        self._ax.relim()
        self._ax.autoscale_view(scalex=True, scaley=False)

    @property
    def ylim(self):
        return self._ax.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, limits):
        self._ax.set_ylim(*limits)
    @ylim.deleter
    def ylim(self):
        self._ax.relim()
        self._ax.autoscale_view(scalex=False, scaley=True)

    @property
    def colorlim(self):
        try:
            return self._colorbar.get_clim()
        except AttributeError:
            layer = self._get_color_layer()
            return self._layers[layer].get_clim()
    @colorlim.setter
    @auto_refresh
    def colorlim(self, limits):
        try:
            self._colorbar.set_clim(*limits)
            self._colorbar.draw_all()
        except AttributeError:
            layer = self._get_color_layer()
            self._layers[layer].set_clim(*limits)

    @property
    def logx(self):
        return self._ax.get_xscale() == "log"
    @logx.setter
    @auto_refresh
    def logx(self, log):
        self._ax.set_xscale(log and "log" or "linear")

    @property
    def logy(self):
        return self._ax.get_yscale() == "log"
    @logy.setter
    @auto_refresh
    def logy(self, log):
        self._ax.set_yscale(log and "log" or "linear")

    @property
    def logcolor(self):
        return self._logcolor
    @logcolor.setter
    @auto_refresh
    def logcolor(self, log):
        if self._logcolor == log:
            return
        if hasattr(self, '_colorbar'):
            clim = self._colorbar.get_clim()
            self._figure.delaxes(self._colorbar.ax)
            del self._colorbar
            self.add_colorbar(log=log, clim=clim)
        else:
            self._logcolor = log
            for mappable in self._layers.values():
                if hasattr(mappable, 'norm'):
                    if log:
                        mappable.set_norm(colors.LogNorm(*mappable.get_clim()))
                    else:
                        mappable.set_norm(
                            colors.Normalize(*mappable.get_clim()))

    # -----------------------------------------------

    @auto_refresh
    def add_line(self, x, y, layer=None, **kwargs):
        """Add a line to the current plot

        @param x
            x positions of the line points (in axis coordinates)
        @param y
            y positions of the line points (in axis coordinates)
        @param layer
            name for this line
        @param kwargs
            additional keyword arguments passed directly on to
            matplotlib's `matplotlib.axes.Axes.plot` method.
        """
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("linewidth", 2)
        kwargs.setdefault("markersize", 0)
        l = self._ax.plot(x, y, **kwargs)[0]
        if not layer:
            layer = kwargs.get("label", None)
        if not layer:
            layer = "line_%d" % len(self._line_layers)
        self._line_layers.append(layer)
        self._layers[layer] = l
        self.add_legend()

    @auto_refresh
    def add_markers(self, x, y, layer=None, **kwargs):
        """Add a set or points to the current plot

        @param x
            x positions of the line points (in axis coordinates)
        @param y
            y positions of the line points (in axis coordinates)
        @param layer
            name for this marker set
        @param kwargs
            additional keyword arguments passed directly on to
            matplotlib's `matplotlib.axes.Axes.scatter` method.
        """
        if not "c" in kwargs:
            kwargs.setdefault("edgecolor", "black")
            #kwargs.setdefault("facecolor", "none")
        kwargs.setdefault("s", 20)
        s = self._ax.scatter(x, y, **kwargs)
        if not layer:
            layer = "marker_set_%d" % len(self._scatter_layers)
        self._scatter_layers.append(layer)
        self._layers[layer] = s
        self.add_legend()

    @auto_refresh
    def add_bars(self, left, height, width=0.8, bottom=None, layer=None,
                 **kwargs):
        """Add a set of bars to the current plot
        """
        bar = self._ax.bar(left, height, width=width, bottom=None, **kwargs)
        if not layer:
            layer = "bars_%d" % self._bar_layers
        self._bar_layers.append(layer)
        self._layers[layer] = bar
        self.add_legend()

    @auto_refresh
    def add_rectangle(self, xll, yll, width, height, layer=None, **kwargs):
        """Add a rectangle to the current plot
        """
        p = mpl.Rectangle((xll, yll), width, height, **kwargs)
        self.add_patch(p, layer=layer, **kwargs)
        self.add_legend()

    @auto_refresh
    def add_patch(self, patch, layer=None, **kwargs):
        """Add a patch to the current plot
        """
        patch = self._ax.add_patch(patch, **kwargs)
        if not layer:
            layer = "patch_%d" % self._patch_layers
        self._patch_layers.append(layer)
        self._layers[layer] = patch
        self.add_legend()

    @auto_refresh
    def add_image(self, image, layer=None, **kwargs):
        im = self._ax.imshow(image, **kwargs)
        if not layer:
            layer = 'image_%s' % self._image_layers
        self._image_layers.append(layer)
        self._layers[layer] = im
        self.add_legend()
        return im

    @auto_refresh
    def add_legend(self, *args, **kwargs):
        """Add a legend to the figure
        """
        empty = not len(self._ax.get_legend_handles_labels()[0])
        if empty:
            self.legend = None
        else:
            alpha = kwargs.pop("alpha", 0.8)
            linewidth = kwargs.pop("linewidth", 8)
            self.legend = self._ax.legend(*args, **kwargs)
            self.legend.set_alpha(alpha)
            [l.set_linewidth(linewidth) for l in self.legend.get_lines()]

    @auto_refresh
    def add_colorbar(self, layer=None, location='right', width=0.2, pad=0.1,
                     log=None, label="", clim=None, visible=True, **kwargs):
        if hasattr(self, '_colorbar'):
            self._figure.delaxes(self._colorbar.ax)
            del self._colorbar
        # find default layer
        layer = layer or self._get_color_layer()
        mappable = self._layers[layer]

        # get new colour axis
        divider = make_axes_locatable(self._ax)
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
        mappable.set_clim(clim)

        # set tick format
        if mpl.rcParams["text.usetex"]:
            kwargs.setdefault('format',
                              ticker.FuncFormatter(lambda x,pos:
                                               '$%s$' % tex.float_to_latex(x, '%.3g')))

        # set log scale
        if log is None:
            log = self._logcolor
        if log:
            mappable.set_norm(colors.LogNorm(*mappable.get_clim()))
        else:
            mappable.set_norm(colors.Normalize(*mappable.get_clim()))
        self._logcolor = log

        # make colour bar
        self._colorbar = self._figure.colorbar(mappable, cax=cax, **kwargs)


        # set label
        if label:
            self._colorbar.set_label(label)
        self._colorbar.draw_all()

        # set ticks
        if len(self._colorbar.ax.get_yticks()) < 4:
            limits = self._colorbar.get_clim()
            if log:
                cticks = numpy.logspace(numpy.log10(limits[0]),
                                        numpy.log10(limits[1]), num=5)
            else:
                cticks = numpy.linspace(*limits, num=5)
            self._colorbar.set_ticks(cticks)

    def add_timeseries(self, timeseries, **kwargs):
        kwargs.setdefault('layer', timeseries.name)
        x = timeseries.get_times()
        y = timeseries.data
        self.add_line(x, y, **kwargs)

    def add_spectrum(self, spectrum, **kwargs):
        kwargs.setdefault('layer', spectrum.name)
        x = spectrum.get_frequencies()
        y = spectrum.data
        self.add_line(x, y, **kwargs)

    def add_spectrogram(self, spectrogram, **kwargs):
        kwargs.setdefault('layer', spectrogram.name)
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
            self._ax.set_xlim(map(func, self._ax.get_xlim()))
            self._ax.xaxis.set_data_interval(
                *map(func, self._ax.xaxis.get_data_interval()), ignore=True)
        else:
            self._ax.set_ylim(map(func, self._ax.get_ylim()))
            self._ax.yaxis.set_data_interval(
                *map(func, self._ax.yaxis.get_data_interval()), ignore=True)

    def refresh(self):
        """Refresh the current figure
        """
        self._figure.canvas.draw()

    def show(self):
        """Display the current figure
        """
        self._figure.patch.set_alpha(0.0)
        self._figure.show()

    def save(self, *args, **kwargs):
        """Save the figure to disk.

        All `args` and `kwargs` are passed directly to the savefig
        method of the underlying `matplotlib.figure.Figure`
        self.fig.savefig(*args, **kwargs)
        """
        self._figure.savefig(*args, **kwargs)

    def close(self):
        """Close the plot and relase its memory.
        """
        pyplot.close(self._figure)

    def _get_color_layer(self):
        # find default layer
        if len(self._image_layers) > 0:
            layer = self._image_layers[-1]
        elif len(self._patch_layers) > 0:
            layer = self._patch_layers[-1]
        elif len(self._scatter_layers) > 0:
            layer = self._scatter_layers[-1]
        else:
            raise AttributeError("No color-mappable layers found in this Plot")
        return layer

    @auto_refresh
    def add_label_unit(self, unit, axis="x"):
        attr = "%slabel" % axis
        label = getattr(self, attr)
        if not label and unit.physical_type != u'unknown':
            label = unit.physical_type.title()
        elif not label and hasattr(unit, "name"):
            label = unit.name
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
        axis.set_axis_format(self._ax.xaxis, format_, **kwargs)

    @auto_refresh
    def set_yaxis_format(self, format_, **kwargs):
        axis.set_axis_format(self._ax.yaxis, format_, **kwargs)
