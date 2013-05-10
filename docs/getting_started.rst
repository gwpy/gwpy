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

The `gwpy.io` module provides modules for reading and writing data from a number of formats. If you wanted to read data from a `GWF` file for a given data channel, you can import the `gwf` module and read the data::

    >>> from gwpy.io import gwf
    >>> data = gwf.read('myframe.gwf', 'G1:DER_DATA_H')

where `G1:DER_DATA_H` is the name of the data stream for calibrated gravitational wave strain recorded by the GEO600 instrument.

The returned `data` object is a `~gwpy.types.timeseries.TimeSeries` object.
