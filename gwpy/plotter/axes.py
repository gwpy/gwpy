
"""Extension of the :class:`~matplotlib.axes.Axes` class with
user-friendly attributes
"""

from matplotlib.axes import Axes as _Axes

from .decorators import auto_refresh

from ..version import version as __version__
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

class Axes(_Axes):
    """An extension of the standard matplotlib
    :class:`~matplotlib.axes.Axes` object with simpler attribute
    accessors, and GWpy class plotting methods

    Notes
    -----
    A new set of `Axes` should be constructed via::

        >>> plot.add_subplots(111, projection='xxx')

    where plot is a :class:`~gwpy.plotter.fig.Plot` figure, and ``'xxx'``
    is the name of the `Axes` you want to add.

    Attributes
    ----------
    name
    xlabel
    ylabel
    xlim
    ylim
    logx
    logy

    Methods
    -------
    plot
    plot_dqflag
    plot_segmentlist
    plot_segmentlistdict
    plot_timeseries
    """
    # -----------------------------------------------
    # text properties

    # x-axis label
    @property
    def xlabel(self):
        """Label for the x-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.xaxis.label
    @xlabel.setter
    @auto_refresh
    def xlabel(self, text):
        if isinstance(text, basestring):
            self.set_xlabel(text)
        else:
            self.xaxis.label = text
    @xlabel.deleter
    @auto_refresh
    def xlabel(self):
        self.set_xlabel("")

    # y-axis label
    @property
    def ylabel(self):
        """Label for the y-axis

        :type: :class:`~matplotlib.text.Text`
        """
        return self.yaxis.label
    @ylabel.setter
    @auto_refresh
    def ylabel(self, text):
        if isinstance(text, basestring):
            self.set_ylabel(text)
        else:
            self.yaxis.label = text
    @ylabel.deleter
    @auto_refresh
    def ylabel(self):
        self.set_ylabel("")

    # -----------------------------------------------
    # limit properties

    @property
    def xlim(self):
        """Limits for the x-axis

        :type: `tuple`
        """
        return self.get_xlim()
    @xlim.setter
    @auto_refresh
    def xlim(self, limits):
        self.set_xlim(*limits)
    @xlim.deleter
    @auto_refresh
    def xlim(self):
        self.relim()
        self.autoscale_view(scalex=True, scaley=False)

    @property
    def ylim(self):
        """Limits for the y-axis

        :type: `tuple`
        """
        return self.get_ylim()
    @ylim.setter
    @auto_refresh
    def ylim(self, limits):
        self.set_ylim(*limits)
    @ylim.deleter
    def ylim(self):
        self.relim()
        self.autoscale_view(scalex=False, scaley=True)

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

    # -------------------------------------------
    # Axes methods

    @auto_refresh
    def resize(self, pos, which='both'):
        """Set the axes position with::

            pos = [left, bottom, width, height]

        in relative 0,1 coords, or *pos* can be a
        :class:`~matplotlib.transforms.Bbox`

        There are two position variables: one which is ultimately
        used, but which may be modified by :meth:`apply_aspect`, and a
        second which is the starting point for :meth:`apply_aspect`.
        """
        return super(Axes, self).set_position(pos, which='both')
