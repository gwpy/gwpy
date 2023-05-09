.. currentmodule:: pydischarge.detector

.. _pydischarge-channel:

##########################
The :class:`Channel` class
##########################

.. code-block:: python

   >>> from pydischarge.detector import Channel


====================
What is a 'channel'?
====================

Each of the laser-interferometer gravitational-wave detectors record thousands of different data streams capturing a single instrumental error or control signal, or the output of an environmental sensor.
These data streams are known as 'channels', and are named according to a convention that describes its source in the instrument, and the signal it records, for example::

   L1:PSS-ISS_PDB_OUT_DQ

describes the output signal (``OUT``) from photodiode-B (``PDB``) inside the Intensity Stabilisation System (``ISS``) of the Pre-Stabilised Laser (``PSL``) powering the ``L1`` instrument, hosted at the LIGO Livingston Observatory. The ``DQ`` suffix indicates that this channel was recorded in data files for offline study (many more 'test point' channels are used only in real-time, and are never recorded).

A simple representation of this physical signal is provided by the :class:`Channel` object::

    >>> from pydischarge.detector import Channel
    >>> signal = Channel('L1:PSL-ISS_PDB_OUT_DQ')

This new :class:`Channel` has a number of attributes that describe its source, derived from its name::

    >>> print(hoft.ifo)
    'L1'
    >>> print(hoft.system)
    'PSL'
    >>> print(hoft.subsystem)
    'ISS'

and so on.

Alongside nominative attributes, each :class:`Channel` has the following attributes:

.. autosummary::

   ~Channel.name
   ~Channel.sample_rate
   ~Channel.unit
   ~Channel.dtype
   ~Channel.url
   ~Channel.model

Each of these can be manually passed to the `Channel` constructor, or downloaded directly from the LIGO Channel Information System.

==========================================================
The LIGO Channel Information System (https://cis.ligo.org)
==========================================================

All of the LIGO interferometer data channels are recorded in the Channel Information System (https://cis.ligo.org), a queryable database containing the details of each channel recorded to disk from the observatories.
The :meth:`Channel.query` `classmethod` allows you to query the database as follows::

    >>> from pydischarge.detector import Channel
    >>> chan = Channel.query('L1:IMC-F_OUT_DQ')
    >>> print(chan.sample_rate)
    16384.0 Hz
    >>> print(chan.url)
    https://cis.ligo.org/channel/282666
    >>> print(chan.model)
    l1lsc

In this example we have accessed the information for the frequency noise output signal (``F_OUT``) from the Input Mode Cleaner (``IMC``) of the ``L1`` instrument.

=================
The `ChannelList`
=================

Groups of channels may be collected together in a `ChannelList`, a simple extension of the built-in `list` with functionality for `finding <ChannelList.find>` and `sieveing <ChannelList.sieve>` for specific `Channel` names, sample-rates, or types.

===============
Class reference
===============

This reference contains the following `class` entries:

.. autosummary::
   :nosignatures:

   Channel
   ChannelList

.. autoclass:: Channel

.. autoclass:: ChannelList
