.. currentmodule:: gwpy.timeseries
.. sectionauthor:: Duncan Macleod <macleoddm@cardiff.ac.uk>

.. _gwpy-timeseries-get:

##############
Data Discovery
##############

.. _gwpy-timeseries-data:

*******************
GW Observatory Data
*******************

======================
Overview of Data Types
======================

Each GW observatory produces several distinct types of data,
each serving different purposes in the data analysis process:

- Raw Instrumental Data - Direct output from detector monitoring and control systems
- Calibrated Strain Data *h(t)* - Dimensionless strain time series
- Analysis-Ready Data - Quality-controlled data prepared for analysis

The final 'Analysis-Ready Data' are what are typically used for
gravitational-wave searches and parameter estimation, and are repackaged
and distributed by |GWOSC|_.

===========================
Data Collection and Storage
===========================

--------
Raw data
--------

The observatories record ~30 MBytes/second of 'raw' data that are continuously
written to :ref:`GWF <gwpy-timeseries-io-gwf>` files, encompassing thousands of
channels monitoring the detector state, environmental conditions, and the
gravitational-wave channel itself.

The raw data include:

- Auxiliary channels: Environmental and instrumental monitoring
- Control signals: Feedback and control system data
- Calibration data: Information needed for strain reconstruction

These data are available to collaboration members via direct access at the
observatory sites and computing centres, and may be available remotely using
:ref:`gwpy-external-nds2`.

----------------------
Calibrated strain data
----------------------

The raw data are processed in real time to produce the calibrated strain data
*h(t)*, which is the primary data product used in gravitational-wave searches.

These data are stored in :ref:`GWF <gwpy-timeseries-io-gwf>` files, and are
made available to collaboration members through |OSDFl|_.

-------------------
Analysis-ready data
-------------------

The final analysis-ready data are produced by zeroing out data outside of observing
times.

These data are stored in :ref:`GWF <gwpy-timeseries-io-gwf>` files, and are
made available to collaboration members through |OSDFl|_.

-----------
Public data
-----------

|GWOSCl|_ is the public archive for gravitational-wave data from LIGO, Virgo,
and KAGRA.
It provides open access to strain data, data quality information, and
software tools for analysis.
The data are available to the public after a proprietary period following
each observing run.

GWpy's automatic data discovery includes support for searching and downloading
data from |GWOSC|_, see :ref:`gwpy-timeseries-get-gwosc`.

******************
`TimeSeries.get()`
******************

The :meth:`~Timeseries.get` class method provides a simple interface to automatically discover and
download gravitational-wave strain data and auxiliary channels from a variety of sources.

The method is designed to be a one-stop shop for downloading time series data, with the user only
needing to specify the channel name, start time, and end time. The method will then automatically
determine the best source for the data, download it, and return it as a :class:`~Timeseries` object.

The method supports a variety of data sources, including:

- :ref:`gwpy-timeseries-get-gwdatafind` - for the widest range of strain data and auxiliary channels
- :ref:`gwpy-timeseries-get-gwosc` - for publicly-released gravitational-wave strain data through the |GWOSC|_ API
- :ref:`gwpy-timeseries-get-nds2` - for remote access to real-time and archived data from the LIGO detectors

=====
Usage
=====

------------------
Required arguments
------------------

:meth:`TimeSeries.get` takes the following parameters:

- **channel** (*str*) - The name of the channel to download; this can also be a simple detector prefix such as 'H1' or 'L1' to download the default strain channel for that detector from GWOSC.
- **start** - The start time of the data to download; this can be a GPS time (as a float or int), or a human-readable string (e.g. '2015-09-14 09:50:29').
- **end** - The end time of the data to download; this can be a GPS time (as a float or int), or a human-readable string.

With these inputs, :meth:`TimeSeries.get` will construct a list of possible data sources to query, possibly
trying each one multiple times in different configurations, and will return the first successful result.

.. admonition:: Regarding channel names

    To use :meth:`TimeSeries.get` for anything other than public strain data from |GWOSC|_,
    you need to know the full name of the data channel you want, which is often not obvious.

    The `Observatory Data Sets <https://gwosc.org/data/>`_ page on the GWOSC website
    includes links to channel lists for each public auxiliary data release.
    For the full proprietary data set, the IGWN Detector Characterisation working group maintains
    a record of the most relevant channels for studying a given interferometer subsystem.

    The proprietary strain data channels are documented internally within the collaborations,
    and while the names are often similar to the public channels, they are not guaranteed to be the same.

.. admonition:: Debugging data retrieval issues

    If :meth:`TimeSeries.get` is unable to find or download the requested data, it will raise a
    a `gwpy.io.registry.GetExceptionGroup` that contains the errors raised by each source attempt.
    To catch this exception, or specific sub-errors, you should use the new `except*` syntax introduced
    in Python 3.11, as shown in the example below:

    .. code-block:: python

        import warnings
        from gwpy.timeseries import TimeSeries
        try:
            data = TimeSeries.get("H1", "2015-09-14 09:50:30", "2015-09-14 09:51:00")
        except* ValueError as e:
            warnings.warn("Something went wrong")
            raise

    If the error message is not sufficiently informative, you can enable debug logging to see more details
    of the data retrieval process, including which data sources were tried and what errors were encountered.
    See :doc:`/logging` for details of how to enable debug logging.

    Note that `except*` will only catch the exceptions you tell it to, any other exception types raised
    by any source will propagate as normal.
    To catch all exceptions raised by all sources, you can use `except ExceptionGroup`.

------------------
Optional arguments
------------------

See :meth:`TimeSeries.get` for the full list of optional arguments that can be used to
customise and optimise the data retrieval process.

============
Data sources
============

.. _gwpy-timeseries-get-gwdatafind:

----------
GWDataFind
----------

GWDataFind is a service provided by the LIGO Scientific Collaboration to facilitate
the discovery and retrieval of gravitational-wave data, both strain data and auxiliary channels.
It provides a unified interface to search for data across multiple observatories and data archives.

The :meth:`TimeSeries.get` method uses the :doc:`gwdatafind <gwdatafind:index>` Python package
to query the GWDataFind service for the requested channel(s) and time range,
and will read or download the data from the best available source on-the-fly.

For remote data access, :meth:`TimeSeries.get` will typically retrieve data from |OSDF|_, which
provides a distributed network of data caches to ensure high availability and low latency access
to gravitational-wave data.

.. _gwpy-timeseries-get-gwosc:

-----
GWOSC
-----

|GWOSCl|_ manages public data releases of gravitational-wave strain and auxiliary data from the
LIGO, Virgo, and KAGRA observatories.
These data are freely accessible to any user.

The :meth:`TimeSeries.get` method uses the :doc:`gwosc <gwosc:index>` Python
package to query the GWOSC service for the requested channel(s) and time range,
and will download the data from GWOSC if it is available.

.. _gwpy-timeseries-get-nds2:

----
NDS2
----

|nds2l|_ is a networked data service that provides access to real-time and archived
gravitational-wave data from the LIGO detectors.
It is primarily intended for use within the LIGO Scientific Collaboration, but a public
NDS2 server is operated by GWOSC to provide access to a subset of the public data.

The :meth:`TimeSeries.get` method uses the :ref:`gwpy-external-nds2` Python package to connect
to an NDS2 server and retrieve the requested channel(s) and time range, if available.

========
Examples
========

See :doc:`/examples/index` for examples of using :meth:`TimeSeries.get` to retrieve
gravitational-wave strain data and auxiliary channels from various sources.

------------------------
Download data from GWOSC
------------------------

For example, to download 30 seconds of LIGO-Livingston strain data around the first ever
gravitational-wave detection (|GW150914|_), you can use :meth:`TimeSeries.get` as follows:

.. code-block:: python

    from gwpy.timeseries import TimeSeries
    data = TimeSeries.get("L1", "2015-09-14 09:50:30", "2015-09-14 09:51:00")

--------------------------------
Download proprietary strain data
--------------------------------

If you have access to the proprietary LIGO data, you can use :meth:`TimeSeries.get` to
download strain data for specific times (this downloads one minute of data from the start
of the current UTC day):

.. code-block:: python

    from gwpy.timeseries import TimeSeries
    data = TimeSeries.get("H1:GDS-CALIB_STRAIN", "00:00", "00:01")

**********************
`TimeSeriesDict.get()`
**********************

The :meth:`TimeSeriesDict.get` class method provides a similar interface to
:meth:`TimeSeries.get`, but allows multiple channels to be specified and returned
as a :class:`~gwpy.timeseries.TimeSeriesDict` object.

The only difference in usage is that the ``channel`` argument should be a list of
channel names, rather than a single channel name.

For example, to download 30 seconds of strain data from both LIGO detectors around
the first ever gravitational-wave detection (|GW150914|_):

.. code-block:: python

    from gwpy.timeseries import TimeSeriesDict
    data = TimeSeriesDict.get(["H1", "L1"], "2015-09-14 09:50:30", "2015-09-14 09:51:00")

*******************
`StateVector.get()`
*******************

The :meth:`StateVector.get` class method provides a similar interface to
:meth:`TimeSeries.get`, but is used to retrieve state vector data, which are
discrete time series that represent the state of a system or subsystem.

The usage is similar to :meth:`TimeSeries.get`, with the same required and optional
arguments.
However, simple detector prefixes such as 'H1' or 'L1' trigger a download of the
relevant default state vector channel for that detector, rather than the strain channel.

For example, to download 30 seconds of state vector data from the LIGO-Livingston
detector around the first ever gravitational-wave detection (|GW150914|_):

.. code-block:: python

    from gwpy.timeseries import StateVector
    data = StateVector.get("L1", "2015-09-14 09:50:30", "2015-09-14 09:51:00")
