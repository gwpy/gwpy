.. _timeseries:

.. currentmodule:: gwpy.timeseries

#############################
Accessing interferometer data
#############################

Gravitational-wave detectors are time-domain instruments, attempting to record gravitational-wave amplitude as a differential change in the lengths of each of the interferometer arms.
The primary output of these detectors is a single time-stream of gravitational-wave strain.

Alongside these data, thousands of auxiliary instrumental control and error signals and environmental monitors are recorded in real-time and recorded to disk and archived for off-line study.
The data are archived in ``.gwf`` files, a custom binary format that efficiently stores the time streams and all necessary metadata, for more details about this particular data format, take a look at the specification document `LIGO-T970130 <https://dcc.ligo.org/LIGO-T970130/public>`_.

==================
Remote data access
==================

GWpy provides ways to download these data, one for authorised collaborators, and another for public data releases:

==================================  ===========  =========================
Method                              Restricted?  Description
==================================  ===========  =========================
:meth:`TimeSeries.get`              LIGO.ORG     Fetch data via NDS2 or
                                                 `gw_data_find`
:meth:`TimeSeries.fetch_open_data`  public       Fetch data from LIGO Open
                                                 Science Center (LOSC)
==================================  ===========  =========================

Public
------

For example, to fetch 32 seconds of strain data around event `GW150914 <http://dx.doi.org/10.1103/PhysRevLett.116.061102>`_

.. plot::
   :context: reset
   :include-source:

   from gwpy.timeseries import TimeSeries
   data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)

Here the ``'L1'`` refers to the `LIGO Livingston Observatory <https://www.ligo.caltech.edu/LA>`_ (``'H1'`` would be the `LIGO Hanford Observatory <https://www.ligo.caltech.edu/WA>`_), and the two 10-digit numbers are GPS times (for more details on GPS timing, see :ref:`time`). You could just as easily use a more readable format::

    data = TimeSeries.fetch_open_data('L1', 'Sep 14 2015 09:50:29', 'Sep 14 2015 09:51:01')

Now that you have the data, making a plot is as easy as calling the :meth:`~TimeSeries.Plot` method:

.. plot::
   :context:

   plot = data.plot()
   plot.show()

.. note::

   These data are part of small snippet of LIGO data provided by the `LIGO Open Science Center <https://losc.ligo.org>`_, and are freely available to the public, see the LOSC website for full details on which data are available.

LIGO.ORG
--------

If you have LIGO.ORG credentials, meaning you're a member of the LIGO Scientific Collaboration or the Virgo Collaboration, you need to specify the exact name of the :ref:`channel <channel>` you want.
For Advanced LIGO the calibrated strain channel is called ``GDS-CALIB_STRAIN``, so to fetch H1 strain data for the same period as above::

    data = TimeSeries.get('H1:GDS-CALIB_STRAIN', 1126259446, 1126259478)

With authenticated access, you also have access to the thousands of auxiliary channels mentioned above, if you run.::

    gnd = TimeSeries.get('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

you'll get back one hour of data from the vertical-ground-motion seismometer located near the ITMY vacuum enclosure at LIGO Livingston.

The :meth:`TimeSeries.get` method tries direct file access (using :mod:`glue.datafind` for file discovery) first, then falls back to using the Network Data Server (NDS2) for remote access. If you want to manually use NDS2 for remote access you can instead use the :meth:`TimeSeries.fetch` method.

=================
Local data access
=================

If you have direct access to one or more files of data, either on the LIGO Data Grid (where the ``gwf`` files are stored) or you have some files of your own, you can use the `TimeSeries.read` method:

This method is an example of the unified input/output system provided by `astropy <https://astropy.org>`_, so should respond in the same way whether you give it a ``gwf`` file, or an `''hdf5''` file, or a simple `''txt''` file.::

    data = TimeSeries.read('mydata.gwf', 'L1:GDS-CALIB_STRAIN')

.. note::

   The input options to `TimeSeries.read` depend on what format the source data are in, but the first argument will always be the path of the file to read. Please refer to the function documentation for details and examples

If you are on the LIGO Data Grid, but you don't know where the files are, you can use the :meth:`TimeSeries.find` method to automatically locate and read data for a given channel::

    data = TimeSeries.find('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

This method will search through all available data to find the correct files to read, so this may take a while. If you know the frametype - the tag associated with files containing your data, you can pass that to significantly speed up the search::

    data = TimeSeries.find('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00', frametype='L1_R')

Passing ``frametype`` will help most for accessing auxiliary data, rather than calibrated strain.

----------
Frametypes
----------

Data recorded by LIGO are identified by a frametype tag, which identifies which data are contained in a given ``gwf`` file.
The following table is an incomplete, but probably OK, reference to which frametype you want to use for auxiliary data access:

=========  ==========================================================================
Frametype  Description
=========  ==========================================================================
``H1_R``   All auxiliary channels, stored at the native sampling rate
``H1_T``   Second trends of all channels, including ``.mean``, ``.min``, and ``.max``
``H1_M``   Minute trends of all channels, including ``.mean``, ``.min``, and ``.max``
=========  ==========================================================================

The above frametypes refer to the ``H1`` (LIGO-Hanford) instrument, the same are available for LIGO-Livingston by substituting the ``L1`` prefix.

============================================
Accessing data for multiple channels at once
============================================

Because typically each ``gwf`` file contains data for a large number of channels, it is inefficient to ask for data for each channel in turn, since that would meaning opening, decoding, and closing each file every time.
Instead you can read all of the data you want in a single step, using the `TimeSeriesDict` object:

.. plot::
   :context: reset
   :include-source:
   :nofigs:

   from gwpy.timeseries import TimeSeriesDict
   alldata = TimeSeriesDict.get(['H1:PSL-PWR_PMC_TRANS_OUT16','H1:IMC-PWR_IN_OUT16'], 'Feb 1 00:00', 'Feb 1 02:00')

[These two channels represent the power generated by the Pre-Stabilized Laser at the input to interferometer, and the power actually entering the Input Mode Cleaner.]

=======================
Plotting a `TimeSeries`
=======================

As we have seem above, the `TimeSeries` object comes with its own :meth:`~TimeSeries.plot` method, which will quickly construct a `~gwpy.plotter.TimeSeriesPlot`.
The same is true of the `TimeSeriesDict`, so we can extend the above snippet to plot multiple channels:

.. plot::
   :context:
   :include-source:

   plot = alldata.plot()
   ax = plot.gca()
   ax.set_ylabel('Power [W]')
   ax.set_title('Available vs requested input power for H1')
   plot.show()

.. rubric:: What's next

The next documentation topic is :ref:`signal-processing`.

=============
Reference/API
=============

The above documentation references the following objects:

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   TimeSeries
   TimeSeriesDict
   TimeSeriesList
