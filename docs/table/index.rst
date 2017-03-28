.. currentmodule:: gwpy.table

.. _gwpy-table:

############
Tabular data
############

Alongside the :ref:`timeseries <timeseries>` data produced continuously at the
laboratories, a number of different sets of tabular data are generated, typically
recording significant noise transients (glitches) or gravitational-wave events found
in the data.

============================
The `Table` and `EventTable`
============================

GWpy provides two classes for handling tabular data.

.. autosummary::
   :nosignatures:

   gwpy.table.Table
   gwpy.table.EventTable

.. note::

   The `Table` object is just an import of the
   :mod:`astropy.table.Table <astropy.table>` object, provided for
   import convenience, see the Astropy documentation for full details on
   that (excellent) object).

The `EventTable` object extends the functionality of the regular
`~astropy.table.Table` with utilities for processing and plotting tables of
events that contain GPS timestamps.

See the :mod:`astropy.table` documentation for examples of how to create custom tables, and much more.

============================
Reading/writing tabular data
============================

.. toctree::
   :maxdepth: 2

   io.rst

=====================
Plotting tabular data
=====================

.. toctree::
   :maxdepth: 2

   plot.rst

=========================
`EventTable` applications
=========================

.. toctree::
   :titlesonly:

   rate
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
