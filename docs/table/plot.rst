.. currentmodule:: gwpy.table

.. _gwpy-table-plot:

#####################
Plotting tabular data
#####################

.. _gwpy-table-plot-scatter:

=======================
Plotting event triggers
=======================

The `EventTable` class provides a convenience instance method to
:meth:`~EventTable.plot` events by specifying the columns to use for the
x-axis, y-axis, and optionally colouring:

.. plot::
   :context: reset
   :include-source:

   >>> from gwpy.table import EventTable
   >>> events = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst', columns=['time', 'central_freq', 'snr'])
   >>> plot = events.plot('time', 'central_freq', color='snr')
   >>> ax = plot.gca()
   >>> ax.set_epoch(968654552)
   >>> ax.set_yscale('log')
   >>> ax.set_ylabel('Frequency [Hz]')
   >>> plot.add_colorbar(clim=[1, 10], cmap='YlGnBu', log=True, label='Signal-to-noise ratio (SNR)')

.. _gwpy-table-plot-tiles:

====================
Plotting event tiles
====================

Many types of event triggers define a 2-dimensional tile, for example in time and frequency.
These tiles can be plotted in a similar manner to simple triggers.

.. plot::
   :context:
   :include-source:

   >>> events = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst', columns=['time', 'central_freq', 'snr', 'duration', 'bandwidth'])
   >>> plot = events.plot('time', 'central_freq', 'duration', 'bandwidth', color='snr')
   >>> ax = plot.gca()
   >>> ax.set_epoch(968654552)
   >>> ax.set_yscale('log')
   >>> ax.set_ylabel('Frequency [Hz]')
   >>> plot.add_colorbar(clim=[1, 10], cmap='YlGnBu', log=True, label='Signal-to-noise ratio (SNR)')


These code snippets are part of the example :ref:`gwpy-example-table-tiles`.
