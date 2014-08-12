#####################
Data table histograms
#####################

One can easily generate a histogram of a tabular data using the :meth`hist` instance method.
For example, as defined for the `SnglBurstTable`:

.. automethod:: SnglBurstTable.hist

Following the worked example on :doc:`plotting an event table histogram <../examples/gw_ex_plot_event_histogram>` we can study the cumulative distribution of event signal-to-noise ratio:

.. literalinclude:: ../../examples/gw_ex_plot_event_histogram.py
   :lines: 39,42-43,46-50
   :append: plot.show()

.. plot:: ../examples/gw_ex_plot_event_histogram.py

.. note::

   Here we have given ``cumulative=-1`` in order to generate an inverse cumulative histogram (each value of SNR counts the rate of events with that SNR and above).
