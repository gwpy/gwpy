.. currentmodule:: gwpy.table

#####################
Data table histograms
#####################

The `EventTable` object comes with a :meth:`~EventTable.hist` method, allowing
trivial generation of histograms using any column as the counter:

.. automethod:: EventTable.hist

Using the above method we can generate a histogram as follows

.. plot::
   :include-source:

   >>> from gwpy.table import EventTable
   >>> events = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst', columns=['snr'])
   >>> plot = events.hist('snr', weights=1/10., logbins=True, bins=50, histtype='stepfilled')
   >>> ax = plot.gca()
   >>> ax.set_xlabel('Signal-to-noise ratio (SNR)')
   >>> ax.set_ylabel('Rate [Hz]')
   >>> ax.set_title('LHO event triggers for GW100916')
   >>> ax.autoscale(axis='x', tight=True)
   >>> plot.show()

This is a snippet from the example :ref:`example-table-histogram`.
