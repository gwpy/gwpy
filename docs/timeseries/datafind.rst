.. currentmodule:: gwpy.timeseries

.. _gwpy-timeseries-datafind:

########################
Automatic data-discovery
########################

.. _gwpy-timeseries-datafind-intro:

============
Introduction
============

The LIGO Scientific Collaboration uses a service colloquially referred to as
'datafind' to index and archive the locations of data files produced at the
observatories, including those from other detectors (e.g. GEO600).
The datafind service continually updates to provide information with a 
typical latency of 10 minutes, often much less.

The |gwdatafind|_ package can be used to execute queries against the
datafind server to discover the URLs of data files (typically ``.gwf`` format).
See the documentation for that package for full details.

Users who have access to the LIGO Data Grid -- the shared computing infrastructure
that supports internal collaboration data analysis -- can use the local datafind
server to query for local files that can be accessed directly.
Other users, including those on the Open Science Grid, can use the server located
at ``datafind.ligo.org:443`` to query for files archived under
`CVMFS <https://cvmfs.readthedocs.io>`__.

.. _gwpy-timeseries-datafind-discovery:

===================================================
Auto-discovery of data using :meth:`TimeSeries.get`
===================================================

**Additional dependencies:** |LDAStools.frameCPP|_

To discover and read data automatically, use :meth:`TimeSeries.get`, and point
it at a datafind ``host``::

   >>> from gwpy.timeseries import TimeSeries
   >>> data = TimeSeries.get("L1:GWOSC-4KHZ_R1_STRAIN", 1187008880, 1187008884,
   ...                       host="datafind.ligo.org:443")
   >>> print(data)
   TimeSeries([-5.98844033e-20, -6.34482794e-20, -6.31740522e-20,
               ...,  6.23573197e-20,  5.54748519e-20,
                5.91121781e-20]
              unit: dimensionless,
              t0: 1187008880.0 s,
              dt: 0.000244140625 s,
              name: L1:GWOSC-4KHZ_R1_STRAIN,
              channel: L1:GWOSC-4KHZ_R1_STRAIN)

This will execute the following series of steps:

- query datafind.ligo.org for the list of datasets it knows about
- for each dataset, determine whether ``'L1:GWOSC-4KHZ_R1_STRAIN'`` is
  contained within a representative file, and pick the most appropriate
  dataset (if multiple)
- query datafind.ligo.org again for the URLs of file paths for the matched
  dataset name
- read each required file and return the data

.. note::

   At the time of writing, all queries to datafind.ligo.org are restricted to
   persons with a valid LIGO.ORG RFC 3820 (X509) credential.

If any of those steps were to fail, :meth:`TimeSeries.get` will automatically
fall back to attempting to use |nds2| to access the data.

By default, as described, this method will search through all available data to
find the correct files to read, so this may take a while if the server has
knowledge of a large number of different datasets.
If you know the dataset name -- the tag associated with files containing your
data -- you can pass that via the ``frametype`` keyword argument to
significantly speed up the search::

    >>> data = TimeSeries.get("L1:GWOSC-4KHZ_R1_STRAIN", "17 August 2017 12:42:02",
    ...                       "17 August 2017 12:42:06", frametype="L1_GWOSC_O2_4KHZ_R1")

.. _gwpy-timeseries-datafind-datasets:

===================
Available datasets
===================

All data recorded by the current generation of detectors are identified
by a dataset tag, which identifies which data are contained in a
given ``gwf`` file.
The following table is an incomplete, but probably OK, reference to which
dataset (``frametype``) you want to use for file-based data access:

.. table:: Datasets available with |gwdatafind|_
   :align: left
   :name: gwdatafind-datasets

   ========================  =====================================================
   Dataset (frametype)       Description
   ========================  =====================================================
   ``H1_R``                  All auxiliary channels, stored at the native sampling
                             rate
   ``H1_T``                  Second trends of all channels, including
                             ``.mean``, ``.min``, and ``.max``
   ``H1_M``                  Minute trends of all channels, including
                             ``.mean``, ``.min``, and ``.max``
   ``H1_HOFT_C00``           Strain *h(t)* and metadata generated using the
                             real-time calibration pipeline
   ``H1_HOFT_CXY``           Strain *h(t)* and metadata generated using the
                             off-line calibration pipeline at version ``XY``
   ``H1_GWOSC_O2_4KHZ_R1``   4k Hz Strain *h(t)* and metadata as released by
                             |GWOSC|_ for the O2 data release
   ``H1_GWOSC_O2_16KHZ_R1``  16k Hz Strain *h(t)* and metadata as released by
                             |GWOSC|_ for the O2 data release
   ========================  =====================================================

The above datasets refer to the ``H1`` (LIGO-Hanford) instrument, the same are
available for LIGO-Livingston by substituting the ``L1`` prefix.

.. note::

   Not all datasets are available from all datafind servers.  Each LIGO Lab-operated
   computing centre has its own datafind server with a subset of the available
   datasets.
