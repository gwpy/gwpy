.. currentmodule:: gwpy.time

.. _gwpy-time:

####################
Times and timestamps
####################

.. seealso::

    - **Reference:** :mod:`gwpy.time`

========
GPS time
========

All gravitational-wave data are recorded with timestamps in the GPS time system (recording the absolute number of seconds since the start of the GPS epoch at midnight on January 6th 1980).

The LIGO Scientific Collaboration stores such GPS times with nanosecond precision using the `~gwpy.time.LIGOTimeGPS` object.

================
Time conversions
================

`Astropy <http://astropy.org>`__ provides the excellent `Time <astropy.time>` object to allow easy conversion between this format and a number of other formats.
For convenience, this object is available in GWpy as ``gwpy.time.Time``.


On top of that, GWpy provides three simple methods to simplify converting between GPS times and Python-standard `datetime` objects, namely:

.. autosummary::
    :nosignatures:

    tconvert
    to_gps
    from_gps
