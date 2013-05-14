##################
Data quality flags
##################

Getting started
===============

A data quality flag is a named set of segments used to indicate something about the performance of a gravitational wave detector - if the name is well constructed, what that flag indicates should be obvious, for example the flag indicating 'Science' mode operation during initial LIGO was called '`H1:DMT-SCIENCE:1`'. The three components of this name are as follows:

    - `H1`: the prefix of the interferometer for which this flag is defined.
    - `DMT-SCIENCE`: the name of this flag, i.e. `SCIENCE` mode as defined by the `DMT` (Data Monitoring Tool),
    - `1`: version 1 - all flags have a version, used to separate new
      definitions of the same flag in the event of mistakes.

In GWpy, the `~gwpy.segments.flags.DataQualityFlag` class serves to represent these objects in Python, providing the following attributes:

    - `valid`: a `~gwpy.segments.SegmentList` indicating those times during
       which the flag was correctly defined,
    - `active`: a `~gwpy.segments.SegmentList` indicating those times during
       which the flag was active

A `~gwpy.segments.flags.DataQualityFlag` can be defined as follows::

    >>> from gwpy.segments import DataQualityFlag
    >>> science_mode = DataQualityFlag('H1:DMT-SCIENCE:1')

Each of the `valid` and `active` `~gwpy.segments.SegmentList` objects can be defined directly through keyword arguments, or set at any time through the relevant attributes::

    >>> print(science_mode.ifo, science_mode.version)
    ('H1', 1)
    >>> science_mode.valid = [(1, 4), (5, 8)]
    >>> science_mode.active = [(2, 4), (5, 9)]
    >>> print(science_mode.active)
    [Segment(2, 4), Segment(5, 8)]

This example shows a restiction placed on the `DataQualityFlag`: that all `active` `~gwpy.segments.Segments` must fall within a `valid` `~gwpy.segments.Segment`.

LVDB: the 'segment database'
============================

The LIGO and Virgo projects maintain a queriable database of all defined `~gwpy.segments.flags.DataQualityFlags`, to which automatic and manual tools publish flags, and from which users can download the `valid` and `active` segments for a given flag, as follows::

    >>> from gwpy.segments import DataQualityFlag
    >>> science_mode = DataQualityFlag.query(
                           'H1:DMT-SCIENCE:4', 968630415, 968716815,
                           url='https://segdb.ligo.caltech.edu') 
    >>> print science_mode.valid
    [Segment(968630415, 968716815)]
    >>> print science_mode.active
    [Segment(968632249, 968641010),
     Segment(968642871, 968644430),
     Segment(968646220, 968681205),
     Segment(968686357, 968686575),
     Segment(968688760, 968690950),
     Segment(968692881, 968714403)]

In the above example we have recovered the valid and active science-mode segments for the LHO instrument over a day of September 2010. The `url` argument will default to the above address, but can be given to specify different database, as required for access to Science Run 5, or Advanced LIGO data.
