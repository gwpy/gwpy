.. currentmodule:: gwpy.table

.. _gwpy-table:

############
Tabular data
############

Significant events discovered in data from gravitational-wave observatories are
stored in tabular datasets, typically describing astrophysical events, or
noise transients (glitches).

================
The `EventTable`
================

GWpy extends the (excellent) :ref:`astropy-table` from `astropy` by providing
the :class:`EventTable`.

This object extends the functionality of the regular `~astropy.table.Table`
with enhanced I/O to support gravitational-wave data, and utilities for
processing and plotting tables of events that contain GPS timestamps.

See the :any:`astropy-table` documentation for examples of how to create
custom tables, and much more.
The only thing to change if you want to use the `EventTable` instead of
the basic `Table` is the import::

    >>> from gwpy.table import EventTable


============================
Downloading GWOSC catalogues
============================

.. toctree::
   :maxdepth: 2

   gwosc

======================
Reading/writing tables
======================

.. toctree::
   :maxdepth: 2

   io

============
Using tables
============

.. toctree::
   :titlesonly:

   filter
   rate

===============
Plotting tables
===============

.. toctree::
   :maxdepth: 2

   plot
   histogram

===============
Class reference
===============

.. currentmodule:: gwpy.table

This reference includes the following `class` entries:

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   Table
   EventTable
