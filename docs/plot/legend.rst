.. currentmodule:: gwpy.plot

.. _gwpy-plot-legend:

######################
Custom legends in GWpy
######################

GWpy overrides the default :class:`~matplotlib.axes.Axes` class with one that
uses a different default legend handler for line plots.
This means that, by default, lines in a legend will be thicker than on a
standard matplotlib figure:

.. plot::
    :include-source:
    :context: reset

    >>> import gwpy  # <- import anything from gwpy
    >>> from matplotlib import pyplot
    >>> fig = pyplot.figure()
    >>> ax = fig.gca()
    >>> ax.plot(range(10), label='My data')
    >>> ax.legend()
    >>> fig.show()

This can be disabled by passing an empty ``handler_map`` to the
:meth:`~matplotlib.axes.Axes.legend` method:

.. plot::
    :include-source:
    :context: close-figs

    >>> fig = pyplot.figure()
    >>> ax = fig.gca()
    >>> ax.plot(range(10), label='My data')
    >>> ax.legend(handler_map=None)
    >>> fig.show()

Similarly, you can implement your own custom legend handler and overwrite
things yourself.
Below is a simple example, but for more details see
:ref:`matplotlib:legend_guide`.

.. plot::
    :include-source:
    :context: close-figs

    >>> from matplotlib.legend_handler import HandlerLine2D
    >>> from matplotlib.lines import Line2D
    >>> class MyHandler(HandlerLine2D):
    ...     def create_artists(self, *args, **kwargs):
    ...         line, = super().create_artists(*args, **kwargs)
    ...         line.set_linewidth(4.)
    ...         line.set_linestyle('--')
    ...         return [line]
    >>> fig = pyplot.figure()
    >>> ax = fig.gca()
    >>> ax.plot(range(10), label='My data')
    >>> ax.legend(handler_map={Line2D: MyHandler()}, handlelength=10)
    >>> fig.show()
