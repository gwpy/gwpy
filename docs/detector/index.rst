.. currentmodule:: gwpy.detector

###########################################
Detector information (:mod:`gwpy.detector`)
###########################################

Trivially, the most critical component of the current efforts in gravitational-wave detection are the instruments themselves.
The :mod:`gwpy.detector` module provides methods and classes describing the current (second) generation of laser-interferometer gravitational-wave detectors.

The core of the :mod:`~gwpy.detector` module are the following two classes:

.. autosummary::
   :nosignatures:

   ~interferometers.LaserInterferometer
   ~channel.Channel

=============
Data channels
=============

.. currentmodule:: gwpy.detector.channel
.. |Quantity| replace:: :class:`~astropy.units.quantity.Quantity`
.. |Unit| replace:: :class:`~astropy.units.core.Unit`

Each of the laser interferometer gravitational-wave detectors record data in thousands of individual 'channels', each of which records the time-series for a single instrumental error or control signal, or an environmental sensor.
These channels are named according to a convention that describes its source in the instrument, and the signal it records, for example::

   L1:PSS-ISS_PDB_OUT_DQ

describes the output signal from photodiode-B inside the Intensity Stabilisation System of the Pre-Stabilised Laser powering the L1 instrument, hosted at the LIGO Livingston Observatory.
A simple representation of this physical signal is provided by the :class:`Channel` object::

    >>> from gwpy.detector import Channel
    >>> signal = Channel('L1:PSL-ISS_PDB_OUT_DQ')

This new :class:`Channel` has a number of attributes that describe its source, derived from its name::

    >>> print(hoft.ifo)
    'L1'
    >>> print(hoft.system)
    'PSL'
    >>> print(hoft.subsystem)
    'ISS'

and so on.

-------------------------------
LIGO Channel Information System
-------------------------------

All of the LIGO interferometer data channels are recorded in the Channel Information System (https://cis.ligo.org), a queryable database containing the details of each channel recorded to disk from the observatories.
This database holds information about the generation of each signal, including :attr:`~Channel.sample_rate` and real-time control :attr:`~Channel.model`.
You can query for details on a given :class:`Channel` as follows::

    >>> from gwpy.detector import Channel
    >>> psl_odc = Channel.query('L1:PSL-ODC_CHANNEL_OUT_DQ')
    >>> print(psl_odc.sample_rate)
    <Quantity 32768.0 Hz>
    >>> print(psl_odc.model)
    'l1psliss'

Here, the :class:`~Channel.sample_rate` attribute is recorded as a |Quantity|, recording both the value and the unit of the the sample rate.
The :attr:`~Channel.model` attribute records the specific instrumental code in which the `L1:PSL-ODC_CHANNEL_OUT_DQ` channel was defined and recorded.
