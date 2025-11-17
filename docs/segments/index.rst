.. currentmodule:: gwpy.segments

.. _gwpy-segments:

#####################
Data-quality segments
#####################

In order to successfully search data for gravitational-wave signals, precise
records of when each observatory was operating, and in which configuration,
are kept to enable search teams to pick the best data to analyse.

Time segments are recorded denoting when each observatory was taking
observation-quality data and when the calibration was nominal, as well as
during times of possible problems - electronics glitches or severe weather,
for example.

The international collaboration operates using the GPS time standard
(seconds since the GPS epoch of midnight on January 6th 1980), and records
such times as semi-open GPS ``[start, stop)`` segments.

=====================================================
The :class:`Segment` and :class:`SegmentList` classes
=====================================================

GWpy provides a number of classes for generating and manipulating such
segments, enhancing the functionality provided by the (excellent)
:doc:`igwn-segments <igwn-segments:index>` package.
All credits for their usefulness go to the authors of that package.

These basic objects are as follows:

.. autosummary::
    :nosignatures:

    Segment
    SegmentList

While these objects are key to representing core data segments,
they are usually applied to analyses of data as a `DataQualityFlag`.

==================================
The :class:`DataQualityFlag` class
==================================

A `DataQualityFlag` is an annotated set of segments that indicate something
about instrumental operation.
Each flag is defined by applying some algorithm on data and generating a
:class:`SegmentList` that indicates some good or bad condition has been met
during those times.
For example, the times during which the LIGO interferometers are operating
under observing conditions are recorded as the '*analysis-ready*' flag, which
is used by data analysis groups to define periods of data over which to search
for gravitational-wave signals.
Conversely, high seismic noise around the observatory buildings is recorded
in a data-quality flag used by those groups to veto periods of analysis as
a result of sub-standard data.

Each `DataQualityFlag` has some key attributes:

.. autosummary::

    ~DataQualityFlag.name
    ~DataQualityFlag.known
    ~DataQualityFlag.active

By convention, the :attr:`~DataQualityFlag.name` is typically constructed of
three colon-separated components: the
:attr:`~DataQualityFlag.ifo`,
:attr:`~DataQualityFlag.tag`, and
:attr:`~DataQualityFlag.version`,
e.g. ``L1:DMT-ANALYSIS_READY:1``.

=============================
Combining `DataQualityFlag`\s
=============================

`DataQualityFlag`\s can be combined in a number of ways, using the standard python operators, e.g. `&` and `|`.

.. _gwpy-segments-intersection:

--------------------
Intersection (``&``)
--------------------
::

    >>> a & b

returns the intersection of both the `~DataQualityFlag.known` and
`~DataQualityFlag.active` segment lists, e.g::

    >>> a = DataQualityFlag(known=[(0, 5), (10, 15)], active=[(1, 5), (10, 12)])
    >>> b = DataQualityFlag(known=[(0, 12)], active=[(3, 7), (10, 12)])
    >>> print(a & b)
    <DataQualityFlag(No name,
                    known=[[0 ... 5)
                           [10 ... 12)],
                    active=[[3 ... 5)
                            [10 ... 12)],
                    description=None)>

This new flag represents times when both ``a`` and ``b`` were known and
when both were active.

.. _gwpy-segments-union:

-------------
Union (``|``)
-------------
::

    >>> a | b

returns the intersection of both the `~DataQualityFlag.known` and
`~DataQualityFlag.active` segment lists, e.g::

    >>> print(a | b)
    <DataQualityFlag(No name,
                     known=[[0 ... 15)],
                     active=[[1 ... 7)
                             [10 ... 12)],
                     description=None)>

This new flag represents times when either ``a`` or ``b`` were known and when
either was active.

.. _gwpy-segments-sub:

-------------------
Subtraction (``-``)
-------------------
::

    >>> a - b

returns the union of the `~DataQualityFlag.known` segments, and the difference
of the `~DataQualityFlag.active` segment lists, e.g.::

    >>> print(a - b)
    <DataQualityFlag(No name,
                     known=[[0 ... 5)
                            [10 ... 12)],
                     active=[[1 ... 3)],
                     description=None)>

The new flag represents times when both ``a`` and ``b`` were *known*, but only ``a`` was active.

.. _gwpy-segments-add:

----------------
Addition (``+``)
----------------
::

    >>> a + b

This operation is the same as :ref:`gwpy-segments-union`.

-----------------
Inversion (``~``)
-----------------
::

    >>> ~a

returns the same `~DataQualityFlag.known` segments, and the complement `~DataQualityFlag.active` segment lists, e.g::

    >>> print(~a)
    <DataQualityFlag(No name,
                     known=[[0 ... 5)
                            [10 ... 15)],
                     active=[[0 ... 1)
                             [12 ... 15)],
                     description=None)>

The new flag represents times when the state of ``a`` was known, but it was not active.

--------------------
Exclusive OR (``^``)
--------------------
::

    >>> a ^ b

returns the intersection of `~DataQualityFlag.known` segments and the exclusive OR of `~DataQualityFlag.active` segment lists, e.g::

    >>> print(a ^ b)
    <DataQualityFlag(No name,
                     known=[[0 ... 5)
                            [10 ... 12)],
                     active=[[1 ... 3)
                             [5 ... 7)],
                     description=None)>

The new flag represents times when the state of both ``a`` and ``b`` are known, but exactly one of the flags was active.

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

:mod:`gwpy.segments` provides the following `class` entries:

.. autosummary::
    :nosignatures:

    DataQualityFlag
    DataQualityDict
    Segment
    SegmentList
    SegmentListDict
