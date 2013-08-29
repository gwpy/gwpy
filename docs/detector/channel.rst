#######################################################
Detector data channels (:class:`gwpy.detector.Channel`)
#######################################################

.. currentmodule:: gwpy.detector.channel
.. |Quantity| replace:: :class:`~astropy.units.quantity.Quantity`
.. |Unit| replace:: :class:`~astropy.units.core.Unit`

Each of the laser interferometer gravitational-wave detectors record data in thousands of individual 'channels', each of which records the time-series for a single error or control signal.

Getting started
===============
A new `Channel` can be defined as follows::

    >>> from gwpy.detector import Channel
    >>> hoft = Channel('L1:PSL-ISS_PDB_OUT_DQ')

This new `Channel` has a number of attributes describing its source in the instrument, derived from the name::

    >>> print(hoft.ifo)
    'L1'
    >>> print(hoft.system)
    'PSL'
    >>> print(hoft.subsystem)
    'ISS'

and so on.

===============================
LIGO Channel Information System
===============================

All of the LIGO interferometer data channels are recorded in the Channel Information System (https://cis.ligo.org), a queriable database containing the details of each channel recorded to disk from the observatories.

You can query for details on a given channel as follows::

    >>> from gwpy.detector import Channel
    >>> psl_odc = Channel.query('L1:PSL-ODC_CHANNEL_OUT_DQ')
    >>> print(psl_odc.sample_rate)
    <Quantity 32768.0 Hz>
    >>> print(psl_odc.model)
    'l1psliss'

Here, the :class:`Channel.sample_rate` attribute is recorded as a |Quantity|, recording both the value and the unit of the the sample rate.

The :attr:`Channel.model` attribute records the specific instrumental code in which the :obj:`L1:PSL-ODC_CHANNEL_OUT_DQ` channel was defined and recorded.


=============
Reference/API
=============

.. autoclass:: Channel
   :show-inheritance:
   :members: fetch_timeseries, query
