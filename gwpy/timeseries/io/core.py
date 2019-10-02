# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2019)
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

"""Basic I/O routines for :mod:`gwpy.timeseries`
"""

from ...io import cache as io_cache
from ...io.mp import read_multi as io_read_multi


def read(cls, source, *args, **kwargs):
    """Read data from a source into a `gwpy.timeseries` object.

    This method is just the internal worker for `TimeSeries.read`, and
    `TimeSeriesDict.read`, and isn't meant to be called directly.
    """
    # if reading a cache, read it now and sieve
    if io_cache.is_cache(source):
        from .cache import preformat_cache
        source = preformat_cache(source, *args[1:],
                                 start=kwargs.get('start'),
                                 end=kwargs.get('end'))

    # get join arguments
    pad = kwargs.pop('pad', None)
    gap = kwargs.pop('gap', 'raise' if pad is None else 'pad')
    joiner = _join_factory(
        cls,
        gap,
        pad,
        kwargs.get("start", None),
        kwargs.get("end", None),
    )
    # read
    return io_read_multi(joiner, cls, source, *args, **kwargs)


def _join_factory(cls, gap, pad, start, end):
    """Build a joiner for the given cls, and the given padding options
    """
    if issubclass(cls, dict):
        def _join(data):
            out = cls()
            data = list(data)
            while data:
                tsd = data.pop(0)
                out.append(tsd, gap=gap, pad=pad)
                del tsd
            for key in out:
                out[key] = _pad_series(
                    out[key],
                    pad,
                    start,
                    end,
                )
            return out
    else:
        from .. import TimeSeriesBaseList

        def _join(arrays):
            list_ = TimeSeriesBaseList(*arrays)
            return _pad_series(
                list_.join(pad=pad, gap=gap),
                pad,
                start,
                end,
            )
    return _join


def _pad_series(ts, pad, start=None, end=None):
    """Pad a timeseries to match the specified [start, end) limits

    To cover a gap in data returned from a data source
    """
    span = ts.span
    if start is None:
        start = span[0]
    if end is None:
        end = span[1]
    pada = max(int((span[0] - start) * ts.sample_rate.value), 0)
    padb = max(int((end - span[1]) * ts.sample_rate.value), 0)
    if pada or padb:
        return ts.pad((pada, padb), mode='constant', constant_values=(pad,))
    return ts
