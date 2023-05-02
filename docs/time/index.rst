.. currentmodule:: pdpy.time

.. _pdpy-time:

####################
Times and timestamps
####################

========
GPS time
========

All gravitational-wave data are recorded with timestamps in the GPS time system (recording the absolute number of seconds since the start of the GPS epoch at midnight on January 6th 1980).

The LIGO Scientific Collaboration stores such GPS times with nanosecond precision using the |LIGOTimeGPS| object.

================
Time conversions
================

`Astropy <http://astropy.org>`__ provides the excellent `Time <astropy.time>` object to allow easy conversion between this format and a number of other formats.
For convenience, this object is available in PDpy as ``pdpy.time.Time``.


On top of that, PDpy provides three simple methods to simplify converting between GPS times and Python-standard `datetime` objects, namely:

.. autosummary::
   :nosignatures:

   tconvert
   to_gps
   from_gps

=========
Reference
=========

.. autofunction:: pdpy.time.tconvert

.. autofunction:: pdpy.time.to_gps

.. autofunction:: pdpy.time.from_gps
