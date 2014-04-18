.. currentmodule:: gwpy.timeseries.core

#####################################
Reading a `TimeSeries` from GWF files
#####################################

Gravitational-wave frame (GWF) files are archived on disk by the `LIGO Data Grid <https://www.lsc-group.phys.uwm.edu/lscdatagrid/>`_, providing direct access to collaboration members at a number of shared computing centres.
These files are indexed and accessible using the :mod:`~glue.datafind` service:

===================
Finding data frames
===================

.. warning::

   Finding data frames requires the :mod:`glue` python package be installed on your system

The :mod:`glue.datafind` module provides a interface to the indexing service used to record the location on disk of all GWF files.
This package is complemented by the command-line tool ``gw_data_find``.

For any user wishing to read detector data from files on disk, they can login to a shared computing centre and run the following to locate the files::

    >>> from glue import datafind
    >>> connection = datafind.GWDataFindHTTPConnection()
    >>> cache = connection.find_frame_urls('L', 'R', 1067042880, 1067042900, urltype='file')

i.e. open a connection to the server, and query for a set of frame URLs.
This query required the following arguments:

==================  =======================================================================================================
``L``               Single-character observatory identifier, ``L`` for the LIGO Livingston Observatory.
``R``               Single-character frame data type, ``R`` refers to the 'raw' set of channels whose data are archived.
``1067042880``      GPS start time, any GPS integer is acceptable.
``1067042900``      GPS end time.
``urltype='file'``  File scheme restriction, both ``gsiftp`` and ``file`` scheme paths are returned by default.
==================  =======================================================================================================

and returns a :class:`~glue.lal.Cache` object, a `list` of :class:`~glue.lal.CacheEntry` reprentations of individual frame files::

   >>> for ce in cache:
   >>>     print(ce)
   L R 1067042880 32 file://localhost/archive/frames/A6/L0/LLO/L-R-10670/L-R-1067042880-32.gwf

=============
Frame reading
=============

.. warning::

   Reading data from GWF files requires that either the ``frameCPP`` or ``lalframe`` packages (including SWIG bindings for Python) are installed on your system.

The above :class:`~glue.lal.Cache` can be passed into the :meth:`TimeSeries.read`, to extract data for a specific :class:`~gwpy.detector.channel.Channel`::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.read(cache, 'L1:PSL-ODC_CHANNEL_OUT_DQ')

The :meth:`TimeSeries.read` `classmethod` will accept any of the following as its first argument:

    * a single GWF frame path (as a `str`, or a :class:`~glue.lal.CacheEntry`)
    * a :class:`glue.lal.Cache` object, or a :lalsuite:`LALCache` object

while the second argument should always be a :class:`~gwpy.detector.channel.Channel`, or simply a channel `str` name.
Optional ``start`` and ``end`` keyword arguments can be given to restrict the returned data span.

As part of the unified input/output system, the :meth:`TimeSeries.read` method is documented as follows:

=========================
Reading multiple channels
=========================

Normally, each frame file holds the data for more than one channel over a single GPS epoch.
Any user can read all channels of interest, assuming they all exist in a single GWF file, using the :class:`TimeSeriesDict` object, and its :meth:`~TimeSeriesDict.read` `classmethod`::

    >>> from gwpy.timeseries import TimeSeriesDict
    >>> datadict = TimeSeriesDict.read(cache, ['L1:PSL-ISS_PDA_OUT_DQ', 'L1:PSL-ISS_PDB_OUT_DQ'])
    >>> print(datadict.keys())
    ['L1:PSL-ISS_PDA_OUT_DQ', 'L1:PSL-ISS_PDB_OUT_DQ'])

The output is an :class:`~collections.OrderedDict` of (``name``, `TimeSeries`) pairs as read from the cache.

=========================
A note on frame libraries
=========================

GWpy takes its ability to read GWF-format files from one of the two available GWF I/O libraries:

========  ========================================================================
lalframe  The LALSuite frame I/O library, built on top of the core FrameL library.
frameCPP  A stand-alone C++ library with python wrappings built using `SWIG`.
========  ========================================================================

Each of these provide much the same functionality, with one caveat:

.. note::

   Only when using ``format='framecpp'`` format can GWpy extract multiple `TimeSeries` from a single file without opening it multiple times;
   when using the ``format='lalframe'`` each `TimeSeries` is read by re-opening the given frame-file.
   As a result, when available the ``'framecpp'`` format specifier is the default.
