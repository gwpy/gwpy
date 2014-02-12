##############################################################################
Data channels (:class:`gwpy.detector.Channel <gwpy.detector.channel.Channel>`)
##############################################################################

.. currentmodule:: gwpy.detector.channel

.. code-block:: python

   >>> from gwpy.detector import Channel


====================
What is a 'channel'?
====================

Each of the laser-interferometer gravitational-wave detectors record thousands of different data streams capturing a single instrumental error or control signal, or the output of an environmental sensor.
These data streams are known as 'channels', and are named according to a convention that describes its source in the instrument, and the signal it records, for example::

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

Alongside nomnitive attributes, each :class:`Channel` has the following attributes:

.. autosummary::

   ~Channel.name
   ~Channel.sample_rate
   ~Channel.unit
   ~Channel.dtype
   ~Channel.url
   ~Channel.model

==========================================================
The LIGO Channel Information System (https://cis.ligo.org)
==========================================================

All of the LIGO interferometer data channels are recorded in the Channel Information System (https://cis.ligo.org), a queryable database containing the details of each channel recorded to disk from the observatories.
The :meth:`Channel.query` :class:`classmethod` allows you to query the database as follows::

    >>> from gwpy.detector import Channel
    >>> chan = Channel.query('L1:IMC-F_OUT_DQ')
    >>> print(chan.sample_rate)
    16384.0 Hz
    >>> print(chan.url)
    https://cis.ligo.org/channel/282666
    >>> print(chan.model)
    l1lsc

