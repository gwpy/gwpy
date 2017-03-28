.. currentmodule:: gwpy.timeseries

.. _gwpy-timeseries-datafind:

####################################
Data-discovery on the LIGO Data Grid
####################################

.. _gwpy-timeseries-datafind-discovery:

Data discovery
--------------

The LIGO Data Grid consists of a trio of large-scale high-throughput computing (HTC) facilities operated by the LIGO Laboratory, and serves as the primary host of the data recorded from the observatories.

The observatories record ~30 MBytes/second of 'raw' data that are continuously written to GWF-format files (see :ref:`gwpy-timeseries-io-gwf` for more details on the GWF format), as well as the processed strain data produced in real-time for low-latency gravitational-wave searches.
A data discovery service named `datafind` is available allowing any authenticated user to query for the locations of GWF files by specifying the observatory, the frame type, and a time interval.

The `TimeSeries.find` method leverages that discovery service to enable users to automatically locate and read data from these files in a convenient manner.
For example::

    >>> data = TimeSeries.find('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

This method will search through all available data to find the correct files to read, so this may take a while. If you know the frametype - the tag associated with files containing your data - you can pass that via the ``frametype`` keyword argument to significantly speed up the search::

    >>> data = TimeSeries.find('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00', frametype='L1_R')

.. _gwpy-timeseries-datafind-frametypes:

Frametypes
----------

All data recorded by LIGO are identified by a frametype tag, which identifies which data are contained in a given ``gwf`` file.
The following table is an incomplete, but probably OK, reference to which frametype you want to use for auxiliary data access:

===============  ==========================================================
Frametype        Description
===============  ==========================================================
``H1_R``         All auxiliary channels, stored at the native sampling rate
``H1_T``         Second trends of all channels, including ``.mean``,
                 ``.min``, and ``.max``
``H1_M``         Minute trends of all channels, including ``.mean``,
                 ``.min``, and ``.max``
``H1_HOFT_C00``  Strain *h(t)* and metadata generated using the real-time
                 calibration pipeline
``H1_HOFT_CXY``  Strain *h(t)* and metadata generated using the off-line
                 calibration pipeline at version ``XY``
===============  ==========================================================

The above frametypes refer to the ``H1`` (LIGO-Hanford) instrument, the same are available for LIGO-Livingston by substituting the ``L1`` prefix.
