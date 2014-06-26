#############################################
Generating data-quality flags by thresholding
#############################################

.. currentmodule:: gwpy.segments

The first- and second-generation ground-based laser interferometer gravitational-wave detectors are subject to a large variety of linear noise sources, in which noise in control signals can couple directly into the gravitational-wave readout.
If the coupling between an auxiliary signal and the gravitational-wave signal can be detected, noise in the auxiliary signal can be flagged by recording times when the time-series signal exceeded a nominal range.

These times can then be recorded as GPS [start, stop) segments, and applied to any analysis of gravitational-wave data as a veto.

In GWpy, a `DataQualityFlag` can be generated from any :class:`~gwpy.timeseries.TimeSeries` by applying a simple mathematical operator::

    >>> from gwpy.timeseries import TimeSeries
    >>> seisdata = TimeSeries.fetch('L1:HPI-BS_BLRMS_Z_3_10', 1064534416, 1064538016)
    >>> seisdata.unit = 'nm/s'
    >>> highseismic = seisdata > 400
    >>> flag = highseismic.to_dqflag(name='L1:DCH-HIGH_SEISMIC_1_3', round=True)
    >>> print(flag)
    <DataQualityFlag(valid=[[1064534416 ... 1064538016)],
                     active=[[1064535295 ... 1064535296)
                             [1064535896 ... 1064535897)
                             [1064536969 ... 1064536970)
                             [1064537086 ... 1064537088)
                             [1064537528 ... 1064537529)],
                     ifo='L1',
                     name='DCH-HIGH_SEISMIC_1_3',
                     version=None,
                     comment='L1:HPI-BS_BLRMS_Z_3_10 > 400 nm / s')>

In this worked example, times of ground-motion above 400 nm/s in the 1-3 Hz band, as recorded by a seismometer, are recorded as a `DataQualityFlag`.
The keyword arguments given to the :meth:`~gwpy.timeseries.StateTimeSeries.to_dqflag` method give the flag a sensible name, using the standard naming convention, and make sure the segments are rounded outwards to integer start and stop times.
