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

============================  ===========  ===============================================
Method                        Restricted?  Description
============================  ===========  ===============================================
`TimeSeries.get`              LIGO.ORG     Fetch data via NDS2 or `gw_data_find`
`TimeSeries.fetch_open_data`  public       Fetch data from LIGO Open Science Center (LOSC)
============================  ===========  ===============================================

Public
------

For example, to fetch 32 seconds of strain data around event `GW150914 <http://dx.doi.org/10.1103/PhysRevLett.116.061102>`_::

    from gwpy.timeseries import TimeSeries
    data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)

Here the ``'L1'`` refers to the `LIGO Livingston Observatory <https://www.ligo.caltech.edu/LA>`_ (``'H1'`` would be the `LIGO Hanford Observatory <https://www.ligo.caltech.edu/WA>`_), and the two 10-digit numbers are GPS times (for more details on GPS timing, see :ref:`time`). You could just as easily use a more readable format::

    data = TimeSeries.fetch_open_data('L1', 'Sep 14 2015 09:50:29', 'Sep 14 2015 09:51:01')

.. note::

   These data are part of small snippet of LIGO data provided by the `LIGO Open Science Center <https://losc.ligo.org>`_, and are freely available to the public, see the LOSC website for full details on which data are available.

LIGO.ORG
--------

If you have LIGO.ORG credentials, meaning you're a member of the LIGO Scientific Collaboration or the Virgo Collaboration, you need to specify the exact name of the :ref:`channel <channel>` you want::

    data = TimeSeries.get('H1:GDS-CALIB_STRAIN', 1126259446, 1126259478)

With authenticated access, you also have access to the thousands of auxiliary channels mentioned above, if you run.::

    gnd = TimeSeries.get('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

you'll get back one hour of data from the vertical-ground-motion seismometer located near the ITMY vacuum enclosure at LIGO Livingston.

The `TimeSeries.get` method tries direct file access (using `glue.datafind` for file discovery) first, then falls back to using the Network Data Server (NDS2) for remote access. If you want to manually use NDS2 for remote access you can instead use the `TimeSeries.fetch` method.

=================
Local data access
=================

If you have direct access to one or more files of data, either on the LIGO Data Grid (where the ``gwf`` files are stored) or you have some files of your own, you can use the `TimeSeries.read` method:

This method is an example of the unified input/output system provided by `astropy <https://astropy.org>`_, so should respond in the same way whether you give it a ``gwf`` file, or an `''hdf5''` file, or a simple `''txt''` file.:

    data = TimeSeries.read('mydata.gwf', 'L1:GDS-CALIB_STRAIN')

.. note::

   The input options to `TimeSeries.read` depend on what format the source data are in, please refer to the function documentation for details and examples

If you are on the LIGO Data Grid, but you don't know where the files are, you can use the `TimeSeries.find` method to automatically locate and read data for a given channel::

    data = TimeSeries.find('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

============================================
Accessing data for multiple channels at once
============================================

Because typically each ``gwf`` file contains data for a large number of channels, it is inefficient to ask for data for each channel in turn, since that would meaning opening, decoding, and closing each file every time.
Instead you can read all of the data you want in a single step, using the `TimeSeriesDict` object, either for remote access::

    from gwpy.timeseries import TimeSeriesDict
    alldata = TimeSeriesDict.fetch(['H1:ISI-GND_STS_ITMX_X_DQ', 'H1:SUS-ITMX_L2_WIT_L_DQ', 'Feb 1 09:00', 'Feb 1 09:15')

or from ``gwf`` files::

    alldata = TimeSeriesDict.get(['H1:PSL-PWR_PMC_TRANS_OUT16','H1:IMC-PWR_IN_OUT16'], 'Feb 1 00:00', 'Feb 1 02:00')  # fetch the data via NDS

[These two channels represent the power generated by the Pre-Stabilized Laser at the input to interferometer, and the power actually entering the Input Mode Cleaner.]

=======================
Plotting a `TimeSeries`
=======================

The `TimeSeries` object comes with its own :meth:`~TimeSeries.plot` method, which will quickly construct a `~gwpy.plotter.timeseries.TimeSeriesPlot`.
In the following example, we download ten seconds of gravitational-wave strain data from the LIGO Hanford Observatory, and display it:

.. plot::

   from gwpy.timeseries import TimeSeries  # load the class
   gwdata = TimeSeries.fetch_open_data('H1', 968654552, 968654562)  # fetch data from LOSC
   plot = gwdata.plot()  # make a plot
   ax = plot.gca()  # extract the Axes object
   ax.set_ylabel('Gravitational-wave amplitude [strain]')  # set the label for the Y-axis
   ax.set_title('LIGO Hanford Observatory data')  # set the title
   plot.show()  # show me the plot

|

And similarly for the `TimeSeriesDict` (using `~TimeSeriesDict.plot`):

.. plot::

   from gwpy.timeseries import TimeSeriesDict  # load the class
   alldata = TimeSeriesDict.get(['H1:PSL-PWR_PMC_TRANS_OUT16','H1:IMC-PWR_IN_OUT16'], 'Feb 1 00:00', 'Feb 1 02:00')  # fetch the data via NDS
   plot = alldata.plot()  # make a plot
   ax = plot.gca()  # extract the Axes object
   ax.set_ylabel('Power [W]')  # set the label for the Y-axis
   ax.set_title('Available vs requested input power for H1')  # set the title
   plot.show()  # show me the plot

|

=============
Reference/API
=============

The `gwpy.timeseries` module provides the following `class` objects for handling instrumental data:

.. autosummary::
   :nosignatures:

   TimeSeries
   TimeSeriesList
   TimeSeriesDict

.. autoclass:: TimeSeries

.. autoclass:: TimeSeriesList
   :no-inherited-members:

.. autoclass:: TimeSeriesDict
   :no-inherited-members:

