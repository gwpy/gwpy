.. currentmodule:: gwpy.table

.. _gwpy-table:

###########
Data tables
###########

Significant events discovered in data from gravitational-wave observatories are
stored in tabular datasets, typically describing astrophysical events, or
noise transients (glitches).

.. seealso::

    - **Reference**: :mod:`gwpy.table`

=======================
The :class:`EventTable`
=======================

GWpy extends the (excellent) :class:`~astropy.table.Table` from
`astropy` by providing the :class:`EventTable`.

This object extends the functionality of the regular
:class:`~astropy.table.Table` with enhanced I/O to support
gravitational-wave data, and utilities for processing and plotting tables of
events that contain GPS timestamps.

See the :any:`astropy-table` documentation for examples of how to create
custom tables, and much more.
The only thing to change if you want to use the `EventTable` instead of
the basic `Table` is the import:

.. code-block:: python
    :caption: Importing the `EventTable`
    :name: gwpy-table-import

    >>> from gwpy.table import EventTable

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
