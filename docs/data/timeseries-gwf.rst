.. currentmodule:: gwpy.timeseries.core

###################################
Reading `TimeSeries` from GWF files
###################################

Gravitational-wave frame (GWF) files are archived on disk by the `LIGO Data Grid <https://www.lsc-group.phys.uwm.edu/lscdatagrid/>`_, providing direct access to collaboration members at a number of shared computing centres.
These files are indexed and accessible using the :mod:`~glue.datafind` service:

========
Datafind
========

The :mod:`glue.datafind` module provides a python interface to the indexing service used to record the location on disk of all GWF files.
This package is complemented by the command-line tool ``gw_data_find``.

For any use wishing to read detector data from files on disk, they can login to a shared computing centre and run the following to locate the files::

    >>> from glue import datafind
    >>> connection = datafind.GWDataFindHTTPConnection()
    >>> cache = connection.find_frame_urls('L', 'R', 1067042880, 1067042900, urltype='file')

This query required the following arguments:

==================  ==================================================================================================
``L``               single-character observatory identifier, ``L`` for the LIGO Livingston Observatory
``R``               single-character frame data type, ``R`` refers to the 'raw' data containing all channels
``1067042880``      GPS start time
``1067042900``      GPS end time
``urltype='file'``  file scheme restriction, this removes remote ``gsiftp`` files that would otherwise match the query
==================  ==================================================================================================

and returns a :class:`~glue.lal.Cache` object, a list of :class:`~glue.lal.CacheEntry` reprentations of individual frame files::

   >>> for ce in cache:
   >>>     print(ce)
   L R 1067042880 32 file://localhost/archive/frames/A6/L0/LLO/L-R-10670/L-R-1067042880-32.gwf

=============
Frame reading
=============

The above :class:`~glue.lal.Cache` can be passed into the :meth:`TimeSeries.read`, to extract data for a specific :class:`~gwpy.detector.channel.Channel`::

    >>> from gwpy.timeseries import TimeSeries
    >>> data = TimeSeries.read(cache, 'L1:PSL-ODC_CHANNEL_OUT_DQ')

The :meth:`TimeSeries.read` method will accept any of the following as its first argument:

    * a single GWF frame path
    * a :class:`glue.lal.Cache` object, or a :lalsuite:`LALCache` object

while the second argument should always be a :class:`~gwpy.detector.channel.Channel`, or simply a channel `str` name.
Optional ``start`` and ``end`` keyword arguments can be given to restrict the returned data span.
