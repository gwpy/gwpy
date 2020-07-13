# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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
            if gap in ("pad", "raise"):
                for key in out:
                    out[key] = _pad_series(
                        out[key],
                        pad,
                        start,
                        end,
                        error=(gap == "raise"),
                    )
            return out
    else:
        from .. import TimeSeriesBaseList

        def _join(arrays):
            list_ = TimeSeriesBaseList(*arrays)
            joined = list_.join(pad=pad, gap=gap)
            if gap in ("pad", "raise"):
                return _pad_series(
                    joined,
                    pad,
                    start,
                    end,
                    error=(gap == "raise"),
                )
            return joined
    return _join


def _pad_series(ts, pad, start=None, end=None, error=False):
    """Pad a timeseries to match the specified [start, end) limits

    To cover a gap in data returned from a data source.

    Parameters
    ----------
    ts : `gwpy.types.Series`
        the input series

    pad : `float`, `astropy.units.Quantity`
        the value with which to pad

    start : `float`, `astropy.units.Quantity`, optional
        the desired start point of the X-axis, defaults to
        the start point of the incoming series

    end : `float`, `astropy.units.Quantity`, optional
        the desired end point of the X-axis, defaults to
        the end point of the incoming series

    error : `bool`, optional
        raise `ValueError` when gaps are present, rather than padding
        anything

    Returns
    -------
    series : instance of incoming series type
        a padded version of the series. This may be the same
        object if not padding is needed.

    Raises
    ------
    ValueError
        if `error=True` is given and padding would have been required
        to match the request.
    """
    span = ts.span
    if start is None:
        start = span[0]
    if end is None:
        end = span[1]
    pada = max(int((span[0] - start) * ts.sample_rate.value), 0)
    padb = max(int((end - span[1]) * ts.sample_rate.value), 0)
    if not (pada or padb):  # if noop, just return the input
        return ts
    if error:  # if error, bail out now
        raise ValueError(
            "{} with span {} does not cover requested interval {}".format(
                type(ts).__name__,
                span,
                type(span)(start, end),
            )
        )
    # otherwise applying the padding
    return ts.pad((pada, padb), mode='constant', constant_values=(pad,))
