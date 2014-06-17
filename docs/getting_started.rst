***************
Getting started
***************

==============
Importing GWpy
==============

GWpy is a large package with a number sub-packages, so importing the root package via::

    >>> import gwpy

isn't going to be very useful. Instead, it is best to import the classes you need directly, for example::

   >>> from gwpy.timeseries import TimeSeries

===========================
Object-oriented programming
===========================

GWpy is designed to be an object-oriented programming package, that is, data objects are the central focus of the package. Each data object is represented as an instance of a class, describing its properties and the data it holds.

In order to generate a new instance of any class, you should use the standard constructor, or any of the `classmethod` functions. For example, a :class:`~gwpy.timeseries.TimeSeries` can be generated from an existing data array::

    >>> from gwpy.timeseries import TimeSeries
    >>> mydata = TimeSeries([1,2,3,4,5,6,7,8,9,10], sample_rate=1, epoch=0)

or by downloading it from the relevant network data server::

    >>> from gwpy.timeseries import TimeSeries
    >>> mydata = TimeSeries.fetch('H1:LDAS-STRAIN', 964656015, 964656615)

From there, anything you might want to do with the :class:`~gwpy.timeseries.TimeSeries` can be done directly from the object itself, rather than passing it into a separate function. For example, if you want to calculate the amplitude spectral density of that series::

    >>> spectrum = mydata.asd(4)

where the ``4`` argument tells the :meth:`~gwpy.timeseries.TimeSeries.asd` method to generate a Welch average spectrum (by default) with non-overlapping, 4-second Fourier transforms.

Then you have a new object, a :class:`~gwpy.spectrum.Spectrum`, with its own methods and properties.

============
Core objects
============

There are a small number of core objects provided by GWpy, each representing the standard data products of a gravitational-wave interferometer, or their derivatives. These are

.. autosummary::
   :nosignatures:

   ~gwpy.timeseries.TimeSeries
   ~gwpy.spectrum.Spectrum
   ~gwpy.spectrogram.Spectrogram
   ~gwpy.segments.DataQualityFlag

The following pages in this documentation give full descriptions of how to read and manipulate data, and access data segments, amongst other things.
The remainder of this page outlines a few more key concepts surrounding these core objects.

------------
Input/Output
------------

Each of these objects comes with a standard input method ``read()``, which will accept any of a set of registered file formats for the respective `class`.
For example, you can read a :class:`~gwpy.timeseries.TimeSeries` from a GWF-format file as follows::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.read('/archive/frames/A6/L0/LLO/L-R-10670/L-R-1067042880-32.gwf', 'L1:PSL-ODC_CHANNEL_OUT_DQ')

Similary, each `class` has a standard output method ``write()``, again accepting a number of recognised formats.

-------------
Visualisation
-------------

Analogous to the unified input/output system, each of the standard objects comes with a ``plot()`` method, display that object on a figure using the :mod:`matplotlib` display library.
Following from the above example, the :class:`~gwpy.timeseries.TimeSeries` ``data`` can be displayed via::

    >>> plot = data.plot()

If you have an interactive `backend <http://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`_, you can immediately show the figure on your screen via::

    >>> plot.show()

