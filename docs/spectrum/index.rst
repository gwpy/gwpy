.. currentmodule:: gwpy.frequencyseries

.. _gwpy-frequencyseries:

##################################
The :class:`FrequencySeries` class
##################################

While gravitational-wave detectors are time-domain instruments, their sensitivity is frequency dependent and so is often measured as a power-spectral-density over the range of interesting gravitational-wave frequencies (~10-10,000 Hz).
Additionally, the auxiliary `channels <../detector/channel>`_ used to sense and control instrumental operations each have their own frequency-domain characteristics, contributing to the overall sensitivity spectrum.

The :class:`FrequencySeries` object is used to represent any frequency series, including the power-spectral (and amplitude-spectral) density series describing instrument performance.

Analogously to the :class:`~gwpy.timeseries.TimeSeries`, a new `FrequencySeries` can be generated from any data sequence along with the minimal :attr:`~FrequencySeries.f0` and :attr:`~FrequencySeries.df` metadata::

    >>> from gwpy.frequencyseries import FrequencySeries
    >>> spec = FrequencySeries([1,2,3,4,5,6,7,8,9,10], f0=0, df=1)
    >>> print(spec)
    FrequencySeries([ 1  2  3  4  5  6  7  8  9 10],
                    name: None,
                    unit: None,
                    epoch: None,
                    channel: None,
                    f0: 0 Hz,
                    df: 1 Hz,
                    logf: False)

The full set of metadata that can be provided is as follows:

.. autosummary::

    ~FrequencySeries.name
    ~FrequencySeries.unit
    ~FrequencySeries.epoch
    ~FrequencySeries.f0
    ~FrequencySeries.df

==========================================================================
Generating a `FrequencySeries` from a :class:`~gwpy.timeseries.TimeSeries`
==========================================================================

.. currentmodule:: gwpy.timeseries

The frequency-spectrum of a :class:`TimeSeries` can be calculated using either of the following methods:

.. autosummary::
    :nosignatures:

    TimeSeries.psd
    TimeSeries.asd

In this example we expand upon plotting a :class:`~gwpy.timeseries.TimeSeries`, by calculating the amplitude-spectral density of the gravitational-wave strain data from LHO:

.. plot::
    :context: reset
    :include-source:
    :nofigs:

    >>> from gwpy.timeseries import TimeSeries
    >>> gwdata = TimeSeries.get(
    ...     "H1",
    ...     "Sep 14 2015 09:45",
    ...     "Sep 14 2015 09:55",
    ... )
    >>> spectrum = gwdata.asd(8, 4)

where the result is an average spectrum calculated using the
`Welch method <https://en.wikipedia.org/wiki/Welch_method>`_.

=====================================
Reading/writing frequency-domain data
=====================================

.. toctree::
    :maxdepth: 2

    io


.. _gwpy-frequencyseries-plot:

============================
Plotting a `FrequencySeries`
============================

.. currentmodule:: gwpy.frequencyseries

Similarly to the :class:`~gwpy.timeseries.TimeSeries`, the `FrequencySeries` object comes with its own :meth:`~FrequencySeries.plot` method, which will quickly construct a :class:`~gwpy.plot.Plot`:

.. plot::
    :context:
    :include-source:

    >>> plot = spectrum.plot()
    >>> ax = plot.gca()
    >>> ax.set_xlim(10, 2000)
    >>> ax.set_ylabel(r'GW strain ASD [strain$/\sqrt{\mathrm{Hz}}$]')
    >>> ax.set_ylim(5e-24, 1e-20)
    >>> plot.show()

==============================
`FrequencySeries` applications
==============================

.. toctree::
    :titlesonly:

    filtering
