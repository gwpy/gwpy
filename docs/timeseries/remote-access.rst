.. currentmodule:: gwpy.timeseries
.. include:: ../references.txt

.. _gwpy-timeseries-remote:

##################
Remote data access
##################

The LIGO Laboratory archives instrumental data in GWF files hosted on the LIGO Data Grid (see :ref:`gwpy-timeseries-datafind` for more details), however, remote access tools have been developed to simplify loading data.
GWpy provides two methods for remote data access, one for public data releases, and another for authenticated access to the complete data archive:

==================================  ===========  =========================
Method                              Restricted?  Description
==================================  ===========  =========================
:meth:`TimeSeries.fetch_open_data`  public       Fetch data from LIGO Open
                                                 Science Center (LOSC)
:meth:`TimeSeries.get`              LIGO.ORG     Fetch data via local disk
                                                 or NDS2
==================================  ===========  =========================

.. _gwpy-timeseries-remote-public:

==================
Open data releases
==================

**Additional dependencies:** |h5py|_

The `LIGO Open Science Center <https://losc.ligo.org/>`_ hosts a large quantity of open (meaning publicly-available) data from LIGO science runs, including the full strain record for the sixth LIGO science run (S6, 2009-2010) and short extracts of the strain record surrounding published GW observations from Advanced LIGO.

To fetch 32 seconds of strain data around event `GW150914 <http://dx.doi.org/10.1103/PhysRevLett.116.061102>`_, you need to give the prefix of the relevant observatory (``'H1'`` for the LIGO Hanford Observatory, ``'L1'`` for LIGO Livingston), and the start and end times of your query:

.. plot::
   :context: reset
   :include-source:
   :nofigs:

   >>> from gwpy.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)

Above the times are given in GPS format, but you could just as easily use a more readable format::

    >>> data = TimeSeries.fetch_open_data('L1', 'Sep 14 2015 09:50:29', 'Sep 14 2015 09:51:01')

You can then trivially plot these data:

.. plot::
   :include-source:
   :context:

   >>> plot = data.plot()
   >>> plot.show()

For more details on plotting a `TimeSeries`, see :ref:`gwpy-timeseries-plot`.

.. _gwpy-timeseries-remote-nds2:

=============================
Restricted full data archives
=============================

**Additional dependencies:** |nds2|_

Members of the LIGO Scientific Collaboration or the Virgo Collaboration can access the full raw and processed data archived via an authenticated remote protocol called NDS2.
In this case, the :meth:`TimeSeries.get` method can be used to download data for any one of thousands of archived data channels.

.. note::

   Access to data using the `nds2` client requires a `Kerberos <https://web.mit.edu/kerberos/>`_ authentication ticket.
   This can be generated from the command-line using ``kinit``:

   .. code-block:: bash

      $ kinit albert.einstein@LIGO.ORG

   where ``albert.einstein`` should be replaced with your own LIGO.ORG identity.
   If you don't have an active kerberos credential at the time you run `TimeSeries.get`, GWpy will prompt you to create one.

For Advanced LIGO the calibrated strain data channel is called ``GDS-CALIB_STRAIN``, so to fetch H1 strain data for the same period as above::

    >>> data = TimeSeries.get('H1:GDS-CALIB_STRAIN', 1126259446, 1126259478)

Authenticated collaborators also have access to the thousands of auxiliary channels mentioned above, for example running::

    >>> gnd = TimeSeries.get('L1:ISI-GND_STS_ITMY_Z_DQ', 'Jan 1 2016', 'Jan 1 2016 01:00')

will return one hour of data from the vertical-ground-motion seismometer located near the ITMY vacuum enclosure at LIGO Livingston.

The `TimeSeries.get` method tries direct file access (using :mod:`glue.datafind` for file discovery) first, then falls back to using the Network Data Server (NDS2) for remote access.
If you want to manually use NDS2 for remote access you can instead use the `TimeSeries.fetch` method.
