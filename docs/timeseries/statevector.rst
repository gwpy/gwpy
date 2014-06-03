.. currentmodule:: gwpy.timeseries

#####################################################
The :class:`StateTimeSeries` and :class:`StateVector`
#####################################################

.. code:: python

   >>> from gwpy.timeseries import (StateTimeSeries, StateVector)

A large quantity of important data from gravitational-wave detectors can be distilled into simple boolean (`True` or `False`) statements informing something about the state of the instrument at a given time.
These statements can be used to identify times during which a particular control system was active, or when the signal in a seismometer was above an alarming threshold, for example.
In GWpy, these data are represented by special cases (`sub-classes`) of the `TimeSeries` object:

.. autosummary::
   :nosignatures:

   StateTimeSeries
   StateVector

=================
State time-series
=================

The example of a threshold on signal time-series is the core of a large amount of low-level data quality information, used in searches for gravitational waves, and detector characterisation, and is described by the `StateTimeSeries` object, a specific type of :class:`~gwpy.timeseries.core.TimeSeries` containing only boolean values.

These arrays can be generated from simple arrays of booleans, as follows::

    >>> from gwpy.timeseries import StateTimeSeries
    >>> state = StateTimeSeries([True, True, False, False, False, True, False],
                                sample_rate=1, epoch=1064534416)
    >>> state
    <StateTimeSeries([ True,  True, False, False, False,  True, False], dtype=bool
                     name=None
                     unit=Unit(dimensionless)
                     epoch=<Time object: scale='tai' format='gps' value=1064534416.0>
                     channel=None
                     sample_rate=<Quantity 1 Hz>)>

Alternatively, applying a standard mathematical comparison to a regular :class:`~gwpy.timeseries.core.TimeSeries` will return a `StateTimeSeries`::

    >>> from gwpy.timeseries import TimeSeries
    >>> seisdata = TimeSeries.fetch('L1:HPI-BS_BLRMS_Z_3_10', 1064534416, 1064538016)
    >>> seisdata.unit = 'nm/s'
    >>> highseismic = seisdata > 400
    >>> highseismic
    <StateTimeSeries([False, False, False, ..., False, False, False], dtype=bool
                     name='L1:HPI-BS_BLRMS_Z_3_10 > 400 nm / s'
                     unit=Unit("nm / s")
                     epoch=<Time object: scale='tai' format='gps' value=1064534416.0>
                     channel=Channel("L1:HPI-BS_BLRMS_Z_3_10")
                     sample_rate=<Quantity 16.0 Hz>)>

The `StateTimeSeries` includes a handy :meth:`StateTimeSeries.to_dqflag` method to convert the boolean array into a :class:`~gwpy.segments.flag.DataQualityFlag`::

    >>> segments = highseismic.to_dqflag(round=True)
    >>> segments
    <DataQualityFlag(valid=[[1064534416 ... 1064538016)],
                     active=[[1064535295 ... 1064535296)
                             [1064535896 ... 1064535897)
                             [1064536969 ... 1064536970)
                             [1064537086 ... 1064537088)
                             [1064537528 ... 1064537529)],
                     ifo=None,
                     name=None,
                     version=None,
                     comment='L1:HPI-BS_BLRMS_Z_3_10 > 400 nm / s')>

=======================
Multi-bit state-vectors
=======================

Which the `StateTimeSeries` represents a single `True`/`False` statement about the state of a system, the `StateVector` gives a grouping of these, with a binary bitmask mapping bits in a binary word to descriptions of multiple states in a given compound system.

For example, the state of the full laser interferometer was described in Initial LIGO by a combination of separate states, including:

    - operator set to go to 'science mode'
    - EPICS control system record (conlog) OK
    - instrument locked in all resonant cavities
    - no signal injections
    - no unauthorised excitations

Additionall, the higher bits 5-15 were set 'ON' at all times, such that the word ``0xffff`` indicated 'science mode' operation of the instrument.

This `StateVector` can be read from the S6 frames as::

    >>> from gwpy.timeseries import StateVector
    >>> state = StateVector.fetch('H1:IFO-SV_STATE_VECTOR', 968631675, 968632275,
                                  ['science', 'conlog', 'up', 'no injection', 'no excitation'])
    >>> print(state)
    StateVector([65528 65528 65528 ..., 65535 65535 65535],
                name=H1:IFO-SV_STATE_VECTOR,
                unit=,
                epoch=968631675.0,
                channel=H1:IFO-SV_STATE_VECTOR,
                sample_rate=16.0 Hz,
                bitmask=BitMask(0: science
                                1: conlog
                                2: up
                                3: no injection
                                4: no excitation,
                                channel=H1:IFO-SV_STATE_VECTOR,
                                epoch=968631675.0))

As can be seen, the input bitmask (``['science', 'conlog', 'up', 'no injection', 'no excitation']``) is represented through the `BitMask` class, recording the bits as a list with some metdata about their purpose.

The `StateVector` fetched in the above example can then be parsed into a series of :class:`~gwpy.segments.flag.DataQualityFlag` objects, recording the active segments for that bit in the vector::

    >>> flags = state.to_dqflags(round=True)
    >>> print(flags[0])
    <DataQualityFlag(valid=[[968631675 ... 968632275)],
                     active=[[968632248 ... 968632275)],
                     ifo=None,
                     name='science',
                     version=None,
                     comment='H1:IFO-SV_STATE_VECTOR bit 0')>

===============
Class reference
===============

This reference contains the following `Class` entries:

.. autosummary::
   :nosignatures:

   StateVector
   StateTimeSeries
   StateVectorDict

.. autoclass:: StateVector

.. autoclass:: StateTimeSeries

.. autoclass:: StateVectorDict

