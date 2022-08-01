# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2022)
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

"""Read and write HDF5 files in the LIGO Open Science Center format

For more details, see :ref:`gwpy-table-io`.
"""

import os.path
import re
from math import ceil
from urllib.parse import urlparse

from astropy.io import registry
from astropy.units import Quantity
from astropy.utils.data import get_readable_fileobj

from gwosc.locate import get_urls

from .. import (StateVector, TimeSeries)
from ...io import (gwf as io_gwf, hdf5 as io_hdf5)
from ...io.cache import file_segment
from ...io.utils import file_path
from ...detector.units import parse_unit
from ...segments import Segment
from ...time import to_gps
from ...utils.env import bool_env

DQMASK_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*DQMASK\Z")
STRAIN_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*STRAIN\Z")

GWOSC_LOCATE_KWARGS = (
    'sample_rate',
    'tag',
    'version',
    'host',
    'format',
    'dataset',
)


# -- utilities ----------------------------------------------------------------

def _download_file(url, cache=None, verbose=False):
    if cache is None:
        cache = bool_env('GWPY_CACHE', False)
    return get_readable_fileobj(url, cache=cache, show_progress=verbose)


def _fetch_gwosc_data_file(url, *args, **kwargs):
    """Fetch a single GWOSC file and return a `Series`.
    """
    cls = kwargs.pop('cls', TimeSeries)
    cache = kwargs.pop('cache', None)
    verbose = kwargs.pop('verbose', False)

    # match file format
    if url.endswith('.gz'):
        ext = os.path.splitext(url[:-3])[-1]
    else:
        ext = os.path.splitext(url)[-1]
    if ext == '.hdf5':
        kwargs.setdefault('format', 'hdf5.gwosc')
    elif ext == '.txt':
        kwargs.setdefault('format', 'ascii.gwosc')
    elif ext == '.gwf':
        kwargs.setdefault('format', 'gwf')

    with _download_file(url, cache, verbose=verbose) as rem:
        # get channel for GWF if not given
        if ext == ".gwf" and (not args or args[0] is None):
            args = (_gwf_channel(rem, cls, kwargs.get("verbose")),)
        if verbose:
            print('Reading data...', end=' ')
        try:
            series = cls.read(rem, *args, **kwargs)
        except Exception as exc:
            if verbose:
                print('')
            exc.args = ("Failed to read GWOSC data from %r: %s"
                        % (url, str(exc)),)
            raise
        else:
            # parse bits from unit in GWF
            if ext == '.gwf' and isinstance(series, StateVector):
                try:
                    bits = {}
                    for bit in str(series.unit).split():
                        a, b = bit.split(':', 1)
                        bits[int(a)] = b
                    series.bits = bits
                    series.override_unit('')
                except (TypeError, ValueError):  # don't care, bad GWOSC
                    pass

            if verbose:
                print('[Done]')
            return series


def _overlapping(files):
    """Quick method to see if a file list contains overlapping files
    """
    segments = set()
    for path in files:
        seg = file_segment(path)
        for s in segments:
            if seg.intersects(s):
                return True
        segments.add(seg)
    return False


# -- remote data access (the main event) --------------------------------------

def fetch_gwosc_data(detector, start, end, cls=TimeSeries, **kwargs):
    """Fetch GWOSC data for a given detector

    This function is for internal purposes only, all users should instead
    use the interface provided by `TimeSeries.fetch_open_data` (and similar
    for `StateVector.fetch_open_data`).
    """
    # format arguments
    start = to_gps(start)
    end = to_gps(end)
    span = Segment(start, end)
    kwargs.update({
        'start': start,
        'end': end,
    })

    # find URLs (requires gwopensci)
    url_kw = {key: kwargs.pop(key) for key in GWOSC_LOCATE_KWARGS if
              key in kwargs}
    if 'sample_rate' in url_kw:  # format as Hertz
        url_kw['sample_rate'] = Quantity(url_kw['sample_rate'], 'Hz').value
    cache = get_urls(detector, int(start), int(ceil(end)), **url_kw)
    # if event dataset, pick shortest file that covers the request
    # -- this is a bit hacky, and presumes that only an event dataset
    # -- would be produced with overlapping files.
    # -- This should probably be improved to use dataset information
    if len(cache) and _overlapping(cache):
        cache.sort(key=lambda x: abs(file_segment(x)))
        for url in cache:
            a, b = file_segment(url)
            if a <= start and b >= end:
                cache = [url]
                break
    if kwargs.get('verbose', False):  # get_urls() guarantees len(cache) >= 1
        host = urlparse(cache[0]).netloc
        print("Fetched {0} URLs from {1} for [{2} .. {3}))".format(
            len(cache), host, int(start), int(ceil(end))))

    is_gwf = cache[0].endswith('.gwf')
    if is_gwf and len(cache):
        args = (kwargs.pop('channel', None),)
    else:
        args = ()

    # read data
    out = None
    kwargs['cls'] = cls
    for url in cache:
        keep = file_segment(url) & span
        kwargs["start"], kwargs["end"] = keep
        new = _fetch_gwosc_data_file(url, *args, **kwargs)
        if is_gwf and (not args or args[0] is None):
            args = (new.name,)
        if out is None:
            out = new.copy()
        else:
            out.append(new, resize=True)
    return out


# -- I/O ----------------------------------------------------------------------

@io_hdf5.with_read_hdf5
def read_gwosc_hdf5(
    h5f,
    path='strain/Strain',
    start=None,
    end=None,
    copy=False,
):
    """Read a `TimeSeries` from a GWOSC-format HDF file.

    Parameters
    ----------
    h5f : `str`, `h5py.HLObject`
        path of HDF5 file, or open `H5File`

    path : `str`
        name of HDF5 dataset to read.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    dataset = io_hdf5.find_dataset(h5f, path)
    # read data
    nddata = dataset[()]
    # read metadata
    xunit = parse_unit(dataset.attrs['Xunits'])
    epoch = dataset.attrs['Xstart']
    dt = Quantity(dataset.attrs['Xspacing'], xunit)
    unit = dataset.attrs['Yunits']
    # build and return
    return TimeSeries(nddata, epoch=epoch, sample_rate=(1/dt).to('Hertz'),
                      unit=unit, name=path.rsplit('/', 1)[1],
                      copy=copy).crop(start=start, end=end)


@io_hdf5.with_read_hdf5
def read_gwosc_hdf5_state(
    f,
    path='quality/simple',
    start=None,
    end=None,
    copy=False,
):
    """Read a `StateVector` from a GWOSC-format HDF file.

    Parameters
    ----------
    f : `str`, `h5py.HLObject`
        path of HDF5 file, or open `H5File`

    path : `str`
        path of HDF5 dataset to read.

    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data

    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data

    copy : `bool`, default: `False`
        create a fresh-memory copy of the underlying array

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    # find data
    dataset = io_hdf5.find_dataset(f, '%s/DQmask' % path)
    maskset = io_hdf5.find_dataset(f, '%s/DQDescriptions' % path)
    # read data
    nddata = dataset[()]
    bits = [bytes.decode(bytes(b), 'utf-8') for b in maskset[()]]
    # read metadata
    epoch = dataset.attrs['Xstart']
    try:
        dt = dataset.attrs['Xspacing']
    except KeyError:
        dt = Quantity(1, 's')
    else:
        xunit = parse_unit(dataset.attrs['Xunits'])
        dt = Quantity(dt, xunit)
    return StateVector(nddata, bits=bits, t0=epoch, name='Data quality',
                       dx=dt, copy=copy).crop(start=start, end=end)


def _gwf_channel(path, series_class=TimeSeries, verbose=False):
    """Find the right channel name for a GWOSC GWF file
    """
    channels = list(io_gwf.iter_channel_names(file_path(path)))
    if issubclass(series_class, StateVector):
        regex = DQMASK_CHANNEL_REGEX
    else:
        regex = STRAIN_CHANNEL_REGEX
    found, = list(filter(regex.match, channels))
    if verbose:
        print("Using channel {0!r}".format(found))
    return found


# register
registry.register_reader('hdf5.gwosc', TimeSeries, read_gwosc_hdf5)
registry.register_reader('hdf5.gwosc', StateVector, read_gwosc_hdf5_state)
