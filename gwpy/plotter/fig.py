
"""Extension of the basic matplotlib Figure for GWpy
"""

import inspect
import numpy

from matplotlib import (figure, pyplot, colors as mcolors,
                        ticker as mticker)
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    from mpl_toolkits.axes_grid import make_axes_locatable

from . import IS_INTERACTIVE
from . import tex, axes
from .decorators import auto_refresh


class Plot(figure.Figure):
    """An extension of the matplotib :class:`~matplotlib.figure.Figure`
    object for GWpy
    """
    def __init__(self, auto_refresh=IS_INTERACTIVE, **kwargs):
        # call pyplot.figure to get a better object, but protect against
        # recursion
        super(Plot, self).__init__(**kwargs)
        self._auto_refresh = auto_refresh

    def add_timeseries(self, timeseries, ax=None, **kwargs):
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
        # find relevant axes
        if ax is None and self.axes:
            from .timeseries import TimeSeriesAxes
            tsaxes = [a for a in self.axes if isinstance(a, TimeSeriesAxes)]
            if tsaxes:
                ax = tsaxes[0]
        if ax is None:
            ax = self.add_subplot(111, projection='timeseries')
        # plot on axes
        kwargs.setdefault('label', timeseries.name)
        return ax.plot(timeseries, **kwargs)

    # -----------------------------------------------
    # core plot operations

    def refresh(self):
        """Refresh the current figure
        """
        self.canvas.draw()

    def show(self):
        """Display the current figure
        """
        self.patch.set_alpha(0.0)
        super(Plot, self).show()

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
        pyplot.close(self)

    # -----------------------------------------------
    # colour-bar

    @auto_refresh
    def add_colorbar(self, mappable=None, ax=None, location='right',
                     width=0.2, pad=0.1, log=False, label="", clim=None,
                     visible=True, axes_class=axes.Axes, **kwargs):
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
        elif mappable is None and ax is None:
            for ax in self.axes[::-1]:
                if hasattr(self, 'collections') and len(self.collections):
                    mappable = self.collections[-1]
                    break
        if not mappable:
            raise ValueError("Cannot determine mappable layer for this "
                             "colorbar")

        # find default axes
        if not ax:
            ax = mappable.axes

        # get new colour axis
        divider = make_axes_locatable(ax)
        if location == 'right':
            cax = divider.new_horizontal(size=width, pad=pad,
                                         axes_class=axes_class)
        elif location == 'top':
            cax = divider.new_vertical(size=width, pad=pad,
                                       axes_class=axes_class)
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
        if pyplot.rcParams["text.usetex"]:
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

        # set tick locator
        if log:
            kwargs.setdefault('ticks',
                              mticker.LogLocator(subs=numpy.arange(1,11)))

        # make colour bar
        colorbar = self.colorbar(mappable, cax=cax, **kwargs)

        # set label
        if label:
            colorbar.set_label(label)
        colorbar.draw_all()

        return colorbar


