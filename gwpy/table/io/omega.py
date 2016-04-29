# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Read events from an Omega-format ASCII file.
"""

import sys
from math import sqrt

from six import string_types

from numpy import loadtxt

from astropy.io import registry

from .ascii import table_from_ascii_factory
from ..lsctables import (SnglBurstTable, SnglBurst)
from ...io.cache import file_list
from ...time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OMEGA_DTYPE = [
    ('time', 'a20'),
    ('frequency', '<f8'),
    ('duration', '<f8'),
    ('bandwidth', '<f8'),
    ('energy', '<f8'),
    None,
    None,
    None,
]
OMEGA_LIGOLW_COLUMNS = ['search', 'event_id', 'peak_time', 'peak_time_ns',
                        'start_time', 'start_time_ns',
                        'stop_time', 'stop_time_ns', 'duration',
                        'central_freq', 'flow', 'fhigh', 'bandwidth',
                        'snr', 'amplitude', 'confidence']

OMEGADQ_DTYPE = [
    ('start', 'a20'),
    ('stop', 'a20'),
    ('peak', 'a20'),
    ('flow', '<f8'),
    ('fhigh', '<f8'),
    None,
    ('ms_start', 'a20'),
    ('ms_stop', 'a20'),
    ('ms_fmin', '<f8'),
    ('ms_fmax', '<f8'),
    None,
    ('energy', '<f8'),
    ('ms_snr', '<f8'),
]
OMEGADQ_LIGOLW_COLUMNS = ['search', 'event_id', 'peak_time', 'peak_time_ns',
                          'start_time', 'start_time_ns',
                          'stop_time', 'stop_time_ns', 'duration',
                          'ms_start_time', 'ms_start_time_ns',
                          'ms_stop_time', 'ms_stop_time_ns', 'ms_duration',
                          'central_freq', 'peak_frequency',
                          'flow', 'fhigh', 'bandwidth',
                          'ms_flow', 'ms_fhigh', 'ms_bandwidth',
                          'snr', 'ms_snr', 'amplitude', 'confidence']


def to_LIGOTimeGPS(x):
    if isinstance(x, string_types):
        x = str(x)
    return LIGOTimeGPS(x)


def sngl_burst_from_omega(line, columns=OMEGA_LIGOLW_COLUMNS):
    """Build a `SnglBurst` from an Omega ASCII line.

    Parameters
    ----------
    line : `str`
        the ASCII line to read
    columns : `list`
        a `list` of valid `LIGO_LW` column names to load.

    Returns
    -------
    burst : :class:`~gwpy.table.lsctables.SnglBurst`
        a `SnglBurst` with attributes populated from the ROOT data.
    """
    t = SnglBurst()
    t.search = u'omega'

    if isinstance(line, str):
        line = line.rstrip('\n').split()
    peak, freq, duration, band, nerg = line
    duration = float(duration)

    # parse time data
    peak = to_LIGOTimeGPS(peak)
    if 'time' in columns or 'peak_time' in columns:
        t.peak_time = peak.seconds
    if 'time' in columns or 'peak_time_ns' in columns:
        t.peak_time_ns = peak.nanoseconds
    start = peak - duration / 2.
    if 'start_time' in columns:
        t.start_time = start.seconds
    if 'start_time_ns' in columns:
        t.start_time_ns = start.nanoseconds
    stop = peak + duration / 2.
    if 'stop_time' in columns:
        t.stop_time = stop.seconds
    if 'stop_time_ns' in columns:
        t.stop_time_ns = stop.nanoseconds
    if 'duration' in columns:
        t.duration = duration

    # parse frequency data
    flow = freq - band / 2.
    fhigh = freq + band / 2.
    if 'central_freq' in columns:
        t.central_freq = freq
    if 'flow' in columns:
        t.flow = flow
    if 'fhigh' in columns:
        t.fhigh = fhigh
    if 'bandwidth' in columns:
        t.bandwidth = band

    # others
    snr = sqrt(2 * nerg)
    if 'snr' in columns:
        t.snr = snr
    if 'amplitude' in columns:
        t.amplitude = nerg
    if 'confidence' in columns:
        t.confidence = snr

    return t


def sngl_burst_from_omegadq(line, columns=OMEGADQ_LIGOLW_COLUMNS):
    """Build a `SnglBurst` from an Omega DQ (DetChar) ASCII line.

    Parameters
    ----------
    line : `str`
        the ASCII line to read
    columns : `list`
        a `list` of valid `LIGO_LW` column names to load.

    Returns
    -------
    burst : :class:`~gwpy.table.lsctables.SnglBurst`
        a `SnglBurst` with attributes populated from the ROOT data.
    """
    t = SnglBurst()
    t.search = u'omega'

    if isinstance(line, str):
        line = map(float, line.rstrip('\n').split())
    (start, stop, peak, flow, fhigh, ms_start, ms_stop, ms_flow,
     ms_fhigh, clst_energy, ms_snr) = line

    # parse time data
    peak = to_LIGOTimeGPS(peak)
    if 'time' in columns or 'peak_time' in columns:
        t.peak_time = peak.seconds
    if 'time' in columns or 'peak_time_ns' in columns:
        t.peak_time_ns = peak.nanoseconds
    start = to_LIGOTimeGPS(start)
    if 'start_time' in columns:
        t.start_time = start.seconds
    if 'start_time_ns' in columns:
        t.start_time_ns = start.nanoseconds
    stop = to_LIGOTimeGPS(stop)
    if 'stop_time' in columns:
        t.stop_time = stop.seconds
    if 'stop_time_ns' in columns:
        t.stop_time_ns = stop.nanoseconds
    if 'duration' in columns:
        t.duration = float(stop - start)

    # parse frequency data
    flow = flow
    fhigh = fhigh
    if 'flow' in columns:
        t.flow = flow
    if 'fhigh' in columns:
        t.fhigh = fhigh
    if 'bandwidth' in columns:
        t.bandwidth = fhigh-flow
    if 'central_freq' in columns:
        t.central_freq = flow + (fhigh - flow) * .5

    # most-significant tile information
    ms_start = to_LIGOTimeGPS(ms_start)
    if 'ms_start_time' in columns:
        t.ms_start_time = ms_start.seconds
    if 'ms_start_time_ns' in columns:
        t.ms_start_time_ns = ms_start.nanoseconds
    ms_stop = to_LIGOTimeGPS(ms_stop)
    if 'ms_stop_time' in columns:
        t.ms_stop_time = ms_stop.seconds
    if 'ms_stop_time_ns' in columns:
        t.ms_stop_time_ns = ms_stop.nanoseconds
    if 'ms_duration' in columns:
        t.ms_duration = float(ms_stop - ms_start)
    if 'ms_flow' in columns:
        t.ms_flow = ms_flow
    if 'ms_fhigh' in columns:
        t.ms_fhigh = ms_fhigh
    if 'ms_bandwidth' in columns:
        t.ms_bandwidth = ms_fhigh - ms_flow
    if 'peak_frequency' in columns:
        t.peak_frequency = ms_flow + (ms_fhigh - ms_flow) * .5

    # others
    snr = sqrt(2 * clst_energy)
    if 'snr' in columns:
        t.snr = snr
    if 'amplitude' in columns:
        t.amplitude = clst_energy
    if 'confidence' in columns:
        t.confidence = snr
    if 'ms_snr' in columns:
        t.ms_snr = sqrt(2 * ms_snr)

    return t


# register OmegaDQ
registry.register_reader(
    'omegadq', SnglBurstTable,
    table_from_ascii_factory(
        SnglBurstTable, 'omegadq', sngl_burst_from_omegadq,
        OMEGADQ_LIGOLW_COLUMNS,
        dtype=filter(lambda x: x is not None, OMEGADQ_DTYPE),
        usecols=[i for i, c in enumerate(OMEGADQ_DTYPE) if c is not None]))

# register Omega
registry.register_reader(
    'omega', SnglBurstTable,
    table_from_ascii_factory(
        SnglBurstTable, 'omega', sngl_burst_from_omega, OMEGA_LIGOLW_COLUMNS,
        dtype=filter(lambda x: x is not None, OMEGA_DTYPE), comments='%',
        usecols=[i for i, c in enumerate(OMEGA_DTYPE) if c is not None]))
