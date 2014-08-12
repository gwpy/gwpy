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

We can use the same data as for the example on :doc:`plotting event triggers <../examples/gw_ex_plot_event_triggers>` to demonstrate how to calculate and display the rate versus time of some event triggers:

.. literalinclude:: ../../examples/gw_ex_plot_event_rate.py
   :append: plot.show()
   :lines: 39,42-43,46,49-52

.. plot:: ../examples/gw_ex_plot_event_rate.py

=================
Binned event rate
=================

Following from a simple rate versus time calculation, it is often useful to calculate the event rate for multiple conditions on the same table.
The :meth:`binned_event_rates` method is attached to each :class:`~glue.ligolw.table.Table` subclass for convenience.

.. automethod:: SnglBurstTable.binned_event_rates

For example, in the following example, we calculate the rate of events with signal-to-noise ratio (SNR) above some threshold, for the same table as above.

.. literalinclude:: ../../examples/gw_ex_plot_event_rate_binned.py
   :append: plot.show()
   :lines: 46-47,50-54

.. plot:: ../examples/gw_ex_plot_event_rate_binned.py
