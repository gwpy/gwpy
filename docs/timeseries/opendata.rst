.. currentmodule:: pydischarge.timeseries

.. _pydischarge-timeseries-opendata:

##############################
Accessing Open Data from GWOSC
##############################

******************************************
The Gravitational-Wave Open Science Center
******************************************

|GWOSCl|_ provides strain data from gravitational-wave observatories for
public use, including small datasets around each GW event detection,
and bulk datasets covering entire observing epochs.

.. note::

   For full details of the available data, please see

   https://gwosc.org/data/

.. _pydischarge-timeseries-fetchopendata:

**********************************
:meth:`TimeSeries.fetch_open_data`
**********************************

pyDischarge provides the :meth:`TimeSeries.fetch_open_data` method as an interface
to the GWOSC data archive, requiring users to provide a minimum of
information in order to access data.

For example, to fetch 30 seconds of strain data around the first ever
gravitational-wave detection (|GW150914|_), you need to give the prefix
of the relevant observatory (``'H1'`` for the LIGO Hanford Observatory,
``'L1'`` for LIGO Livingston), and the start and end times of your query.
We can use the |gwosc-mod|_ Python package to query GWOSC itself for the
right central GPS time:

.. plot::
   :context: reset
   :include-source:
   :nofigs:

   >>> from gwosc.datasets import event_gps
   >>> gps = event_gps("GW150914")
   >>> start = int(gps) - 15
   >>> end = int(gps) + 15

Then we can call :meth:`TimeSeries.fetch_open_data` to download the
calibrated GW strain data in that interval:

.. plot::
   :include-source:
   :context:

   >>> from pydischarge.timeseries import TimeSeries
   >>> data = TimeSeries.fetch_open_data('L1', start, end)

We can then trivially plot these data:

.. plot::
   :include-source:
   :context:

   >>> plot = data.plot()
   >>> plot.show()

.. admonition:: Plotting a `TimeSeries`

   For more details on plotting a `TimeSeries`, see
   :ref:`pydischarge-timeseries-plot`.

.. note:: :meth:`TimeSeries.fetch_open_data` keywords

   For more details on all of the available keyword options,
   see the documentation of :meth:`TimeSeries.fetch_open_data`.

**********************************************
Accessing data from Auxiliary Channel Releases
**********************************************

|GWOSC|_ has also published a data set containing instrumental sensor data
in a three-hour window around |GW170814|_ and throughout the entirety of O3.
These data cannot be loaded using :meth:`TimeSeries.fetch_open_data`,
but can be loaded using :meth:`TimeSeries.get`
(or :meth:`TimeSeriesDict.get`), by specifying `host="losc-nds.ligo.org"`.
For more details, see :ref:`pydischarge-timeseries-get`.
