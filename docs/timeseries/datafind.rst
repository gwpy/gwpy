.. currentmodule:: pydischarge.timeseries

.. _pydischarge-timeseries-datafind:

########################
Automatic data-discovery
########################

.. _pydischarge-timeseries-datafind-intro:

The :meth:`TimeSeries.fetch_open_data` method is only able to download
GW strain data from those datasets exposed through the |GWOSC|_ API;
this notably does not include the |GWOSC_AUX_RELEASE|_, or strain data
for any events or observing runs not yet published.

In addition, the GW strain data make up only a tiny fraction of the 'raw'
output of a gravitational-wave detector, which includes in excess of
100,000 different 'channels' (data streams) from sensors and digital control
systems that are used to operate the interferometer and diagnose measure
performance.

The full data set for each detector is archived at the relevant observatory
and is made freely available to all registered collaboration members.
A discovery service called |gwdatafind|_ is provided at each location to
simplify discovering the file(s) that contain the data of interest for any
research.

These data are also made available remotely using :ref:`pydischarge-external-nds2`,
which enables sending data directly over a network to any location.
This is used for both the full proprietary data (which requires an
authorisation credential to access) and also the |GWOSC_AUX_RELEASE|_
(which is freely available).

.. _pydischarge-timeseries-get:

**********************
:meth:`TimeSeries.get`
**********************

**Additional dependencies:** :ref:`pydischarge-external-framecpp` or :ref:`pydischarge-external-nds2`

pyDischarge provides the :meth:`TimeSeries.get` method as a one-stop interface
to all automatically-discoverable data hosted locally at an IGWN
computing centre, or available remotely.

============
How it works
============

Without any customisation, :meth:`TimeSeries.get` will attempt to locate
data 'by any means necessary'; in practice that is

- if the ``GWDATAFIND_SERVER`` environment variable (or legacy
  ``LIGO_DATAFIND_SERVER`` variable) points to a server URL,
  use |gwdatafind|_ to identify which data set includes
  the channel(s) requested, then locate the files for that data set,
  and then read them,
- if that doesn't work (for any reason), loop through the NDS2 servers
  identified in the ``NDSSERVER`` environment variable to see if they
  have the data -- if the ``NDSSERVER`` variable isn't set, guess
  which NDS2 server(s) to use based on the interferometer whose data was
  requested.

.. admonition:: Regarding channel names

   To use :meth:`TimeSeries.get`, you need to know the full name of the
   data channel you want, which is often not obvious.
   The |GWOSC_AUX_RELEASE|_ includes a link to a full listing of all
   included channels.
   For the full proprietary data set, the IGWN Detector
   Characterisation working group maintains a record of the most relevant
   channels for studying a given interferometer subsystem.

.. _pydischarge-timeseries-get-example:

=======
Example
=======

For example, to channel that records the power incident on the
input mode cleaner (IMC) at LIGO-Hanford, is called::

   H1:IMC-PWR_IN_OUT_DQ

and we can use :meth:`TimeSeries.get` to 'get' the data for that
channel by specifying the special GWOSC NDS2 server url using the
``host`` keyword:

.. plot::
   :context: reset

   >>> from gwosc.datasets import event_gps
   >>> from pydischarge.timeseries import TimeSeries
   >>> gps = event_gps("GW170814")
   >>> start = int(gps) - 100
   >>> end = int(gps) + 100
   >>> data = TimeSeries.get("H1:IMC-PWR_IN_OUT_DQ", start, end, host="losc-nds.ligo.org")
   >>> plot = data.plot(ylabel="Power [W]")
   >>> plot.show()

.. _pydischarge-timeseries-datafind-datasets:

********************
Proprietary datasets
********************

All data archived at an IGWN computing centre are identified by a data
set 'tag', which identifies which data are contained in a given ``gwf``
file(s).
By default, as described, :meth:`TimeSeries.get` will search through all
available data to find the correct files to read, so this may take a
while if the server has knowledge of a large number of different datasets.
If you know the dataset name -- the tag associated with files containing your
data -- you can pass that via the ``frametype`` keyword argument to
significantly speed up the search.

The following table is an incomplete, but probably OK, reference to which
dataset (``frametype``) you want to use for file-based data access:

.. tabbed:: GEO-600

   .. table:: GEO-600 datasets available with |gwdatafind|_
      :name: gwdatafind-datasets-geo600

      ========================  =====================================================
      Dataset (frametype)       Description
      ========================  =====================================================
      ``G1_RDS_C01_L3``         The GEO-600 data, including calibrated strain *h(t)*
      ========================  =====================================================

.. tabbed:: LIGO-Hanford

   .. table:: LIGO-Hanford datasets available with |gwdatafind|_
      :name: gwdatafind-datasets-ligo-hanford

      ========================  =====================================================
      Dataset (frametype)       Description
      ========================  =====================================================
      ``H1_R``                  All auxiliary channels, stored at the native sampling
                                rate
      ``H1_T``                  Second trends of all channels, including
                                ``.mean``, ``.min``, and ``.max``
      ``H1_M``                  Minute trends of all channels, including
                                ``.mean``, ``.min``, and ``.max``
      ``H1_HOFT_C00``           Strain *h(t)* and metadata generated using the
                                real-time calibration pipeline
      ``H1_HOFT_CXY``           Strain *h(t)* and metadata generated using the
                                off-line calibration pipeline at version ``XY``
      ``H1_GWOSC_O2_4KHZ_R1``   4k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ``H1_GWOSC_O2_16KHZ_R1``  16k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ========================  =====================================================

.. tabbed:: LIGO-Livingston

   .. table:: LIGO-Livingston datasets available with |gwdatafind|_
      :name: gwdatafind-datasets-ligo-livingston

      ========================  =====================================================
      Dataset (frametype)       Description
      ========================  =====================================================
      ``L1_R``                  All auxiliary channels, stored at the native sampling
                                rate
      ``L1_T``                  Second trends of all channels, including
                                ``.mean``, ``.min``, and ``.max``
      ``L1_M``                  Minute trends of all channels, including
                                ``.mean``, ``.min``, and ``.max``
      ``L1_HOFT_C00``           Strain *h(t)* and metadata generated using the
                                real-time calibration pipeline
      ``L1_HOFT_CXY``           Strain *h(t)* and metadata generated using the
                                off-line calibration pipeline at version ``XY``
      ``L1_GWOSC_O2_4KHZ_R1``   4k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ``L1_GWOSC_O2_16KHZ_R1``  16k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ========================  =====================================================

.. tabbed:: Virgo

   .. table:: Virgo datasets available with |gwdatafind|_
      :name: gwdatafind-datasets-virgo

      ========================  =====================================================
      Dataset (frametype)       Description
      ========================  =====================================================
      ``raw``                   All auxiliary channels, stored at the native sampling
                                rate
      ``V1O2Repro1A``           Strain *h(t)* and metadata for Observing run 3
                                (``O3``) generated off-line using version ``1A``
                                calibration; replace ``O2`` and ``1A`` as appropriate
      ``V1_GWOSC_O2_4KHZ_R1``   4k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ``V1_GWOSC_O2_16KHZ_R1``  16k Hz Strain *h(t)* and metadata as released by
                                |GWOSC|_ for the O2 data release
      ========================  =====================================================

.. admonition:: Not all datasets are available everywhere

   Not all datasets are available from all datafind servers.  Each LIGO Lab-operated
   computing centre has its own datafind server with a subset of the available
   datasets.

.. _pydischarge-timeseries-datafind-trends:

***************
LIGO trend data
***************

The LIGO observatories produce second- and minute- trends of all channels
automatically, and store them in the ``{H,L}1_T`` (second) and ``{H,L}1_M``
(minute) datasets.
However, **the channels in each trend type have the same names**, so
:meth:`TimeSeries.get` doesn't know how to distinguish between the two
different trends when given only the channel name.

To get around this you can directly specify (e.g.) ``frametype="H1_T"``
(for the LIGO-Hanford second trends) in your :meth:`TimeSeries.get`
method call, or you can use a suffix in the channel name:

.. table:: Channel name suffices for LIGO trends
   :name: pydischarge-timeseries-datafind-trend-types

   ==========  ============  ===========
   Trend type  Dataset       Suffix
   ==========  ============  ===========
   second      ``{H,L}1_T``  ``,s-trend``
   minute      ``{H,L}1_M``  ``,m-trend``
   ==========  ============  ===========

e.g.

.. code-block:: python

   >>> TimeSeries.get("L1:IMC-PWR_IN_OUT_DQ.mean,s-trend", 1186741850, 1186741870)

will specifically access the second trends of power incident on the
LIGO-Livingston IMC.

**************************
:meth:`TimeSeriesDict.get`
**************************

:meth:`TimeSeries.get` can only retrieve data for a single channel at a time.
Looping over a list of names to get data for many channels can be very slow,
as each individual call will have to discover and read/download the data for
each channel individually.
The :meth:`TimeSeriesDict.get` method enables retrieval of multiple channels
in a single call, for a single ``(start, stop)`` time interval, greatly
reducing the I/O overhead.

To access data for multiple channels in this way, just pass a `list` of names
rather than a single name.
In this example we download the second trend (average) of ground motion in
the 0.03Hz-0.1Hz range at two locations of the LIGO-Hanford observatory:

.. warning::

   This example uses proprietary data that are only available to members
   of the LIGO Scientific Collaboration and its partners.

.. plot::
   :context: reset

   >>> from pydischarge.timeseries import TimeSeriesDict
   >>> data = TimeSeriesDict.get(
   ...     ["H1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M.rms,s-trend",
   ...      "H1:ISI-GND_STS_ETMY_Z_BLRMS_30M_100M.rms,s-trend"],
   ...     "July 22 2021 12:00",
   ...     "July 22 2021 14:00",
   ... )
   >>> plot = data.plot(ylabel="Ground motion [nm/s]")
   >>> plot.show()
