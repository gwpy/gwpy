.. _currentmodule: pydischarge.plot

.. _pydischarge-plot-colors:

################################################
The Gravitational-Wave Observatory colour scheme
################################################

In order to simplify visual identification of a specific gravitational-wave observatory (GWO) on a figure where many of them are plotted (e.g. amplitude spectral densities, or filtered strain time-series), the GWO standard colour scheme should be used:

.. plot::
    :include-source: False

    import numpy
    from matplotlib import (pyplot, rcParams)
    from matplotlib.colors import to_hex
    from pydischarge.plot import colors

    rcParams.update({
        'text.usetex': False,
        'font.size': 15
    })

    th = numpy.linspace(0, 2*numpy.pi, 512)
    names = [
         'pydischarge:geo600',
         'pydischarge:kagra',
         'pydischarge:ligo-hanford',
         'pydischarge:ligo-india',
         'pydischarge:ligo-livingston',
         'pydischarge:virgo',
    ]

    fig = pyplot.figure(figsize=(5, 2))
    ax = fig.gca()
    ax.axis('off')

    for j, name in enumerate(sorted(names)):
        c = str(to_hex(name))
        v_offset = -(j / len(names))
        ax.plot(th, .1*numpy.sin(th) + v_offset, color=c)
        ax.annotate("{!r}".format(name), (0, v_offset), xytext=(-1.5, 0),
                    ha='right', va='center', color=c,
                    textcoords='offset points', family='monospace')
        ax.annotate("{!r}".format(c), (2*numpy.pi, v_offset), xytext=(1.5, 0),
                    ha='left', va='center', color=c,
                    textcoords='offset points', family='monospace')

    fig.subplots_adjust(**{'bottom': 0.0, 'left': 0.54,
                           'right': 0.78, 'top': 1})
    pyplot.show()

For example:

.. plot::

    from pydischarge.timeseries import TimeSeries
    from pydischarge.plot import Plot
    h1 = TimeSeries.fetch_open_data('H1', 1126259457, 1126259467)
    h1b = h1.bandpass(50, 250).notch(60).notch(120)
    l1 = TimeSeries.fetch_open_data('L1', 1126259457, 1126259467)
    l1b = l1.bandpass(50, 250).notch(60).notch(120)
    plot = Plot(figsize=(12, 4.8))
    ax = plot.add_subplot(xscale='auto-gps')
    ax.plot(h1b, color='pydischarge:ligo-hanford', label='LIGO-Hanford')
    ax.plot(l1b, color='pydischarge:ligo-livingston', label='LIGO-Livingston')
    ax.set_epoch(1126259462.427)
    ax.set_xlim(1126259462, 1126259462.6)
    ax.set_ylim(-1e-21, 1e-21)
    ax.set_ylabel('Strain noise')
    ax.legend()
    plot.show()

The above code was adapted from the example :ref:`pydischarge-example-signal-gw150914`.

The colours can also be specified using the interferometer prefix (e.g. ``'H1'``) via the `pydischarge.plot.colors.GW_OBSERVATORY_COLORS` object:

.. plot::

    from matplotlib import pyplot
    from pydischarge.plot.colors import GW_OBSERVATORY_COLORS
    fig = pyplot.figure()
    ax = fig.gca()
    ax.plot([1, 2, 3, 4, 5], color=GW_OBSERVATORY_COLORS['L1'])
    fig.show()

.. note::

   The ``'pydischarge:<>'`` colours will not be available until `pydischarge`
   has been imported.
