################################################
Detector data channels (`gwpy.detector.Channel`)
################################################

Each of the laser interferometer gravitational-wave detectors record data in thousands of individual 'channels', each of which records the time-series for a single error or control signal.

Getting started
===============
A new channel can be defined as follows::

    >>> from gwpy.detector import Channel
    >>> hoft = Channel('G1:DER_DATA_H')
    >>> print(hoft.ifo, hoft.system)
    ('G1', 'DER')

LIGO Channel Information System
===============================
All of the LIGO interferometer data channels are recorded in the Channel Information System (https://cis.ligo.org), a queriable database containing the details of each channel recorded to disk from the observatories.

You can query for details on a given channel as follows::

    >>> from gwpy.detector import Channel
    >>> psl_odc = Channel.query('L1:PSL-ODC_CHANNEL_OUT_DQ')
    >>> print(psl_odc.sample_rate, psl_odc.model)
    (16384, 'l1psliss')

Here the `model` attribute records the specific instrumental code in which the `L1:PSL-ODC_CHANNEL_OUT_DQ` channel was defined and recorded.

Reference/API
=============

.. currentmodule:: gwpy.detector.channel

.. autoclass:: Channel
   :show-inheritance:
   :members: fetch_timeseries, query
