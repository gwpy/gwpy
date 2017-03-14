.. currentmodule:: gwpy.segments
.. include:: ../references.txt

.. _gwpy-segments:

#####################
Data-quality segments
#####################

In order to successfully search data for gravitational-wave signals, precise records of when each observatory was operating, and in a particular configuration, are kept to enable search teams to pick the best data to analyse.

Time segments are recorded denoting when each observatory was taking science-quality data and when the calibration was nominal, as well as during times of possible problems - electronics glitches or severe weather, for example.

The international collaboration operates using the GPS time standard (seconds since the GPS epoch of midnight on January 6th 1980), and records such times as semi-open GPS ``[start, stop)`` segments.

GWpy provides a number of classes for generating and manipulating such segments, inherited most functionality from the |glue.segments|_ package.
All credits for their usefulness go to the authors of that package.

These basic objects are as follows:

.. autosummary::
   :nosignatures:

   Segment
   SegmentList
   SegmentListDict

While these objects are key to representing core data segments, they are usually applied to analyses of data as a `DataQualityFlag`.

============================
The :class:`DataQualityFlag`
============================

A `DataQualityFlag` is an annotated set of segments that indicate something about instrumental operation.
Each flag is defined by applying some algorithm on data and generating a :class:`SegmentList` that indicates some good or bad condition has been met during those times.
For example, the times during which the LIGO interferometers are operating under observing conditions are recorded as the 'analysis-ready' flag, which are used by data analysis groups to define periods of data over which to run their pipelines.
Conversely, high seismic noise around the observatory buildings is recorded in a data-quality flag used by analysis groups to veto periods of analysis as a result of sub-standard data.

Each `DataQualityFlag` has some key attributes:

.. autosummary::

   ~DataQualityFlag.name
   ~DataQualityFlag.active
   ~DataQualityFlag.valid

By convention, the :attr:`~DataQualityFlag.name` is typically constructed of three colon-separated components: the :attr:`~DataQualityFlag.ifo`, :attr:`~DataQualityFlag.tag`, and :attr:`~DataQualityFlag.version`, e.g. ``L1:DMT-ANALYSIS_READY:1``.

=====================
The `DataQualityDict`
=====================

Groups of `DataQualityFlags <DataQualityFlag>` can be collected into a `DataQualityDict`, a simple extension of the :class:`~collections.OrderedDict` with methods for operating on a group of flags in bulk.

The most immediate utility of this group class is a bulk query of the segment database, using the :meth:`DataQualityDict.query` `classmethod`.
This method is what is actually called by the :meth:`DataQualityFlag.query` `classmethod` anyway.

=====================
Working with segments
=====================

Reading/writing and querying segments:

.. toctree::
   :maxdepth: 2

   dqsegdb
   io

Generating segments from data:

.. toctree::
   :maxdepth: 1

   thresholding

=============
Reference/API
=============

This reference includes the following `class` entries:

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   DataQualityFlag
   DataQualityDict
   Segment
   SegmentList
   SegmentListDict
