.. currentmodule:: gwpy.table

############
Tabular data
############

Alongside the :ref:`timeseries <timeseries>` data produced continuously at the
laboratories, a number of different sets of tabular data are generated, typically
recording significant noise transients (glitches) or gravitational-wave events found
in the data.

========================
`Table` and `EventTable`
========================

GWpy provides two classes for handling tabular data.

.. autosummary::

   gwpy.table.Table
   gwpy.table.EventTable

.. note::

   The `Table` object is just an import of the
   :mod:`astropy.table.Table <astropy.table>` object, provided for import convenience,
   see the Astropy documentation for full details on that (excellent) object).

e.g.::

    >>> from gwpy.table import EventTable

The `EventTable` object extends the functionality of the regular
`~astropy.table.Table` with utilities for processing and plotting tables of
events that contain GPS timestamps. The methods given include

.. autosummary::

   EventTable.plot
   EventTable.hist
   EventTable.event_rate
   EventTable.binned_event_rates

====================
Reading tabular data
====================

Astropy provides an excellent unified input/output system for the
`~astropy.table.Table` object, and GWpy extends upon that to include common
gravitational-wave file types, as well as providing event-specific
input/output registrations for event data.
In the most general case you can read a table of ASCII data as follows::

    >>> from gwpy.table import Table
    >>> table = Table.read('mydata.txt')

See the Astropy documentation page on :mod:`astropy.table` for more details.

----------------------
Reading LIGO_LW tables
----------------------

The LIGO Scientific Collaboration uses a custom scheme of XML in which to
store tabular data, called the ``LIGO_LW`` scheme.
Complementing the scheme is a python library - :mod:`glue.ligolw` - which
allows users to read and write all of the different types of tabular data
produced by gravitational-wave searches.

In GWpy you can read ``LIGO_LW``-format XML files into a `Table` or
`EventTable` by using the relevant `read()` method, while specifying the
table name that you want to read via the ``format`` keyword argument::

   >>> from gwpy.table import Table
   >>> proc = Table.read('mydata.xml', format='ligolw.process')

This will return the ``process`` table from the given ``mydata.xml`` file.

For event tables it is recommended that you read data into an `EventTable`
so that you get access to specialist methods for timestamped events that
aren't available for regular data tables, e.g.

.. code-block:: python

   >>> from gwpy.table import EventTable
   >>> events = EventTable.read('H1-LDAS_STRAIN-968654552-10.xml.gz', format='ligolw.sngl_burst')

By default, this will read all columns available in the file, so you end up with
something like this

.. code-block:: python

   >>> print(events)
   ifo peak_time peak_time_ns start_time ... confidence chisq chisq_dof bandwidth
   --- --------- ------------ ---------- ... ---------- ----- --------- ---------
    H1 968654557    783203126  968654557 ...  16.811825   0.0     512.0     256.0
    H1 968654557    781250001  968654557 ...  16.816761   0.0     512.0     256.0
    H1 968654557    779296876  968654557 ...  16.696106   0.0     512.0     256.0
    H1 968654557    777343751  968654557 ...  16.739489   0.0     512.0     256.0
    H1 968654557    775390626  968654557 ...  16.802326   0.0     512.0     256.0
    H1 968654557    773437501  968654557 ...   16.30731   0.0     512.0     256.0
    H1 968654557    771484376  968654557 ...  16.307253   0.0     512.0     256.0
    H1 968654557    769531251  968654557 ...   16.35647   0.0     512.0     256.0
    H1 968654557    767578126  968654557 ...  16.561176   0.0     512.0     256.0
    H1 968654557    765625001  968654557 ...  16.393112   0.0     512.0     256.0
    H1 968654557    763671876  968654557 ...  16.404041   0.0     512.0     256.0
    H1 968654557    761718751  968654557 ...  16.405825   0.0     512.0     256.0
    H1 968654557    759765626  968654557 ...  16.715092   0.0     512.0     256.0
    H1 968654557    757812501  968654557 ...  17.512749   0.0     512.0     256.0
    H1 968654557    755859376  968654557 ...  17.347675   0.0     512.0     256.0
    H1 968654557    753906251  968654557 ...  17.077478   0.0     512.0     256.0
    H1 968654557    751953126  968654557 ...  16.742907   0.0     512.0     256.0
   ...       ...          ...        ... ...        ...   ...       ...       ...
    H1 968654560    342773438  968654559 ...  11.029792   0.0      16.0       8.0
    H1 968654559    280273438  968654558 ...  12.363036   0.0      16.0       8.0
    H1 968654559    217773438  968654558 ...  13.985101   0.0      16.0       8.0
    H1 968654559    155273438  968654558 ...  14.662391   0.0      16.0       8.0
    H1 968654559     92773438  968654558 ...  15.864924   0.0      16.0       8.0
    H1 968654559     30273438  968654558 ...  16.321821   0.0      16.0       8.0
    H1 968654558    967773438  968654558 ...  16.975931   0.0      16.0       8.0
    H1 968654558    905273438  968654558 ...  19.160393   0.0      16.0       8.0
    H1 968654560    811523438  968654560 ...  11.270205   0.0       8.0       4.0
    H1 968654560    686523438  968654560 ...  15.839205   0.0       8.0       4.0
    H1 968654560    561523438  968654560 ...  15.944695   0.0       8.0       4.0
    H1 968654560    436523438  968654559 ...  14.384432   0.0       8.0       4.0
    H1 968654560    311523438  968654559 ...  14.465309   0.0       8.0       4.0
    H1 968654560    186523438  968654559 ...  13.045853   0.0       8.0       4.0
    H1 968654560    561523438  968654560 ...  11.636543   0.0       4.0       4.0
    H1 968654560    436523438  968654560 ...  15.344837   0.0       4.0       4.0
    H1 968654560    311523438  968654560 ...  11.367717   0.0       4.0       4.0
   Length = 2052 rows

-----------------------
Registered file formats
-----------------------

The full list of registered file formats can be seen by reading the docstring
for the `Table.read` method:

.. automethod:: Table.read

and similarly for the `EventTable`

.. automethod:: EventTable.read


=====================
Plotting tabular data
=====================

-----------------------
Plotting event triggers
-----------------------

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

--------------------
Plotting event tiles
--------------------

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


These code snippets are part of the GWpy example on
:doc:`plotting events as 2-d tiles <../examples/table/tiles>`.

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
