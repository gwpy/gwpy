
"""Extension of the basic matplotlib Figure for GWpy
"""

import inspect

from matplotlib import (figure, pyplot)

from . import IS_INTERACTIVE


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

