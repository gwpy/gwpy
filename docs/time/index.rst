.. _time:

.. currentmodule:: gwpy.time

####################
Times and timestamps
####################

========
GPS time
========

All gravitational-wave data are recorded with timestamps in the GPS time system (recording the absolute number of seconds since the start of the GPS epoch at midnight on January 6th 1980).

The LIGO Scientific Collaboration stores such GPS times with nanosecond precision using the :class:`~gwpy.time.LIGOTimeGPS` object.

================
Time conversions
================

The `astropy <http://astropy.org>`_ package provides the excellent `Time <astropy.time>` object to allow easy conversion between this format and a number of other formats.
For convenience, this object is available in GWpy as ``gwpy.time.Time``.

On top of that, GWpy provides three simple methods to simplify converting between GPS times and Python-standard :class:`datetime.datetime` objects, namely:

.. autosummary::

   tconvert
   to_gps
   from_gps

=========
Reference
=========

This reference contains the following `class` entries:

.. autosummary::
   :nosignatures:

   LIGOTimeGPS

and the following `function` entries:

.. autosummary::
   :nosignatures:

   tconvert
   to_gps
   from_gps


.. autoclass:: LIGOTimeGPS


.. automethod:: gwpy.time.tconvert


.. automethod:: gwpy.time.to_gps


.. automethod:: gwpy.time.from_gps
