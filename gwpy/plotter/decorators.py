
"""Decorator for GWpy plotting
"""

import threading

from matplotlib.figure import Figure

from .decorator import decorator

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

mydata = threading.local()


@decorator
def auto_refresh(f, *args, **kwargs):
    refresh = kwargs.pop('refresh', True)
    # The following is necessary rather than using mydata.nesting = 0 at the
    # start of the file, because doing the latter caused issues with the Django
    # development server.
    mydata.nesting = getattr(mydata, 'nesting', 0) + 1
    try:
        return f(*args, **kwargs)
    finally:
        mydata.nesting -= 1
        if hasattr(args[0], 'figure') and args[0].figure is not None:
            if refresh and mydata.nesting == 0 and args[0].figure._auto_refresh:
                args[0].figure.canvas.draw()
        elif isinstance(args[0], Figure):
            if refresh and mydata.nesting == 0 and args[0]._auto_refresh:
                args[0].canvas.draw()

@decorator
def axes_method(f, *args, **kwargs):
    figure = args[0]
    axes = [ax for ax in figure.axes if ax not in figure.coloraxes]
    if len(axes) == 0:
        raise RuntimeError("No axes found for which '%s' is applicable"
                           % f.__name__)
    if len(axes) != 1:
        raise RuntimeError("{0} only applicable for a Plot with a single set "
                           "of data axes. With multiple data axes, you should "
                           "access the {0} method of the relevant Axes (stored "
                           "in ``Plot.axes``) directly".format(f.__name__))
    axesf = getattr(axes[0], f.__name__)
    return axesf(*args[1:], **kwargs)
