***************
Getting started
***************

Importing GWpy
==============

GWpy is a large package with a number sub-packages, so importing the root package via::

    >>> import gwpy

isn't going to be very useful. Instead, it is best to import the desired sub-package as::

    >>> from gwpy import subpackage

Simple examples
===============

Below are a small set of simple examples to get going with using GWpy for studying data. They are by no means complete, but should reference to a deeper set of documentation as relevant.

#1: Reading data from a GW frame
--------------------------------

The `gwpy.data` module provides objects that represent the basic data products from a time-domain instrument, namely the `TimeSeries`, `Spectrum`, and `Spectrogram`.

All data channels recorded by the instrument are stored in `TimeSeries` within Gravitational Wave Frame (GWF) files.
If you wanted to read data from a `GWF` file for a given data channel, you can import the `TimeSeris` object and fill it with data::

    >>> from gwpy.data import TimeSeries
    >>> data = TimeSeries.read('myframe.gwf', 'G1:DER_DATA_H')

where `G1:DER_DATA_H` is the name of the data stream for calibrated gravitational wave strain recorded by the GEO600 instrument.

The returned `data` object is a `~gwpy.data.TimeSeries` object.

#2: Plotting data from a GW frame
--------------------------------

Given the above example to read data from a frame, it's a simple step further to get these data into an image for viewing::

    >>> from gwpy.data import TimeSeries
    >>> data = TimeSeries.read('myframe.gwf', 'G1:DER_DATA_H')
    >>> plot = data.plot()
    >>> plot.show()
