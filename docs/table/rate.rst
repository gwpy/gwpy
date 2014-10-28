.. currentmodule:: gwpy.table.lsctables

##############################
Calculating event trigger rate
##############################

=================
Simple event rate
=================

For discrete data in the form of event triggers, it is often very illuminating to analyse the rate at which these events occur in time - usually a high event rate is indicative of a higher noise level.
One can calculate the event rate of any of the annotated tables using its :meth:`event_rate` method.
For example, as defined for the `SnglBurstTable`:

.. automethod:: SnglBurstTable.event_rate
   :noindex:

We can use the same data as for the example on :doc:`plotting event triggers <../examples/table/scatter>` to demonstrate how to calculate and display the rate versus time of some event triggers:

.. literalinclude:: ../../examples/table/rate.py
   :lines: 36,39,40,43,46-50

.. plot:: ../examples/table/rate.py

|

This code is a snippet of the example on :doc:`plotting event rate <../examples/table/rate>`.

=================
Binned event rate
=================

Following from a simple rate versus time calculation, it is often useful to calculate the event rate for multiple conditions on the same table.
The :meth:`binned_event_rates` method is attached to each :class:`~glue.ligolw.table.Table` subclass for convenience.

.. automethod:: SnglBurstTable.binned_event_rates
   :noindex:

For example, in the following example, we calculate the rate of events with signal-to-noise ratio (SNR) above some threshold, for the same table as above.

.. literalinclude:: ../../examples/table/rate_binned.py
   :lines: 41-42,48-53

.. plot:: ../examples/table/rate_binned.py

|

This code is a snippet of the example on :doc:`plotting event rate <../examples/table/rate_binned>`.
