.. currentmodule:: gwpy.table

############
Tabular data
############

Alongside the `timeseries <../timeseries>`_ data produced continuously at the
laboratories, the gravitational-wave community produces a number of different
sets of tabular data, including segments of time indicating interferometer
state, and transient event triggers.

==========================
The ``LIGO_LW`` XML format
==========================

The LIGO Scientific Collaboration uses a custom scheme of XML in which to
store tabular data, called the ``LIGO_LW`` scheme.
Complementing the scheme is a python library - :mod:`glue.ligolw` - which
allows users to read and write all of the different types of tabular data
produced by gravitational-wave searches.

The remainder of this document outlines the small number of extensions that
GWpy provides for the table classes provided by :mod:`glue.ligolw`.

.. note::

   All users should review the original documentation for :mod:`glue.ligolw`
   to get any full sense of how to use these objects.

=========================
Reading ``LIGO_LW`` files
=========================

GWpy annotates all of the :class:`~glue.ligolw.table.Table` subclasses defined
in :mod:`glue.ligolw.lsctables` to make reading those tables from ``LIGO_LW``
XML files a bit easier.

These annotations hook into the unified input/output scheme used for all of
the other core data classes.
For example, you can read a table of single-interferometer burst events as
follows:

.. literalinclude:: ../../examples/table/scatter.py
   :lines: 35,38

For full details, check out the :meth:`~SnglBurstTable.read` documentation.

=======================
Plotting event triggers
=======================

The other annotation GWpy defines provides a simple plotting method for a
number of classes.

We can extend the above example to include plotting:

.. literalinclude:: ../../examples/table/scatter.py
   :append: plot.show()
   :lines: 41-46

.. plot:: ../examples/table/scatter.py

|

These code snippets are part of the GWpy example on
:doc:`plotting event triggers <../examples/table/scatter>`.

====================
Plotting event tiles
====================

Many types of event triggers define a 2-dimensional tile, for example in time and frequency.
These tiles can be plotted in a similar manner to simple triggers.

.. literalinclude:: ../../examples/table/tiles.py
   :append: plot.show()
   :lines: 42-47

.. plot:: ../examples/table/tiles.py

|

These code snippets are part of the GWpy example on
:doc:`plotting events as 2-d tiles <../examples/table/tiles>`.

==================
Table applications
==================

.. toctree::
   :titlesonly:

   rate
   histogram

===============
Class reference
===============

.. currentmodule:: gwpy.table.lsctables

.. note::

   All of the below classes are based on :class:`glue.ligolw.table.Table`.

This reference includes the following `class` entries:

.. autosummary::
   :nosignatures:

   CoincDefTable
   CoincTable
   CoincInspiralTable
   CoincRingdownTable
   ExperimentMapTable
   ExperimentSummaryTable
   ExperimentTable
   FilterTable
   GDSTriggerTable
   MultiBurstTable
   MultiInspiralTable
   ProcessTable
   ProcessParamsTable
   SearchSummaryTable
   SearchSummVarsTable
   SimBurstTable
   SimInspiralTable
   SimRingdownTable
   SnglBurstTable
   SnglInspiralTable
   SnglRingdownTable
   StochasticTable
   StochSummTable
   SummValueTable
   SegmentTable
   SegmentSumTable
   SegmentDefTable
   SummMimeTable
   TimeSlideSegmentMapTable
   TimeSlideTable
   VetoDefTable

.. autoclass:: CoincDefTable
   :no-inherited-members:

.. autoclass:: CoincTable
   :no-inherited-members:

.. autoclass:: CoincInspiralTable
   :no-inherited-members:

.. autoclass:: CoincRingdownTable
   :no-inherited-members:

.. autoclass:: ExperimentMapTable
   :no-inherited-members:

.. autoclass:: ExperimentSummaryTable
   :no-inherited-members:

.. autoclass:: ExperimentTable
   :no-inherited-members:

.. autoclass:: FilterTable
   :no-inherited-members:

.. autoclass:: GDSTriggerTable
   :no-inherited-members:

.. autoclass:: MultiBurstTable
   :no-inherited-members:

.. autoclass:: MultiInspiralTable
   :no-inherited-members:

.. autoclass:: ProcessTable
   :no-inherited-members:

.. autoclass:: ProcessParamsTable
   :no-inherited-members:

.. autoclass:: SearchSummaryTable
   :no-inherited-members:

.. autoclass:: SearchSummVarsTable
   :no-inherited-members:

.. autoclass:: SimBurstTable
   :no-inherited-members:

.. autoclass:: SimInspiralTable
   :no-inherited-members:

.. autoclass:: SimRingdownTable
   :no-inherited-members:

.. autoclass:: SnglBurstTable
   :no-inherited-members:

.. autoclass:: SnglInspiralTable
   :no-inherited-members:

.. autoclass:: SnglRingdownTable
   :no-inherited-members:

.. autoclass:: StochasticTable
   :no-inherited-members:

.. autoclass:: StochSummTable
   :no-inherited-members:

.. autoclass:: SummValueTable
   :no-inherited-members:

.. autoclass:: SegmentTable
   :no-inherited-members:

.. autoclass:: SegmentSumTable
   :no-inherited-members:

.. autoclass:: SegmentDefTable
   :no-inherited-members:

.. autoclass:: SummMimeTable
   :no-inherited-members:

.. autoclass:: TimeSlideSegmentMapTable
   :no-inherited-members:

.. autoclass:: TimeSlideTable
   :no-inherited-members:

.. autoclass:: VetoDefTable
   :no-inherited-members:

