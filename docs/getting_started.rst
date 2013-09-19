***************
Getting started
***************
.. replace:: 

==============
Importing GWpy
==============

GWpy is a large package with a number sub-packages, so importing the root package via::

    >>> import gwpy

isn't going to be very useful. Instead, it is best to import the desired sub-package as::

    >>> from gwpy import subpackage

Even better is to import the classes you need directly::

   >>> from gwpy.timeseries import TimeSeries

===========================
Object-oriented programming
===========================

GWpy is designed to be an object-oriented programming package, that is, data objected are the central focus of the package. Each data object is represented as an instance of a class, describing its properties and the data it holds.

In order to generate a new instance of any class, you should use the standard constructor, or any of the `classmethod` functions. For example, a `TimeSeries` can be generated from an existing data array::

    >>> from gwpy.timeseries import TimeSeries
    >>> mydata = TimeSeries([1,2,3,4,5,6,7,8,9,10], sample_rate=1, epoch=0)

or by downloading it from the relevant network data server::

    >>> from gwpy.timeseries import TimeSeries
    >>> mydata = TimeSeries.fetch('H1:LDAS-STRAIN', start=964656015, 964656615)

From there, anything you might want to do with the TimeSeries can be done directly from the object itself, rather than passing it into a separate function. For example, if you want to calculate the amplitude spectral density of that series::

    >>> spectrum = mydata.asd(4)

where the ``4`` argument tells the :meth:`~gwpy.timeseries.core.TimeSeries.asd` method to generate a Welch average spectrum (by default) with non-overlapping, 4-second Fourier transforms.

Then you have a new object, a :class:`~gwpy.spectrum.core.Spectrum`, with its own methods and properties.

============
Core objects
============

There are a small number of core objects provided by GWpy, each representing the standard data products of a gravitational-wave interferometer, or their derivatives. These are

=========================================== =================================
:class:`~gwpy.timeseries.core.TimeSeries`   A data series defined with a starting :attr:`~gwpy.timeseries.core.TimeSeries.epoch` and a :attr:`~gwpy.timeseries.core.TimeSeries.sample_rate`
:class:`~gwpy.spectrum.core.Spectrum`       A data series defined with a base frequency (:attr:`~gwpy.spectrum.core.Spectrum.f0`) and a frequency spacing (:attr:`~gwpy.spectrum.core.Spectrum.df`)
:class:`~gwpy.spectrogram.core.Spectrogram` A 2-dimensional array combining the properties of a :class:`~gwpy.timeseries.core.TimeSeries` and a :class:`~gwpy.spectrum.core.Spectrum`
=========================================== =================================
