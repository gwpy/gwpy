# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

For more details, see https://losc.ligo.org
"""

from __future__ import print_function

import os.path
import re

from six import string_types

import numpy

from astropy.io import registry
from astropy.units import Quantity
from astropy.utils.data import get_readable_fileobj

from .gwf import get_default_gwf_api
from .. import (StateVector, TimeSeries)
from ...io import (hdf5 as io_hdf5, utils as io_utils, losc as io_losc)
from ...io.cache import (cache_segments, file_segment)
from ...detector.units import parse_unit
from ...segments import (Segment, SegmentList)
from ...time import to_gps

# ASCII parsing globals
LOSC_ASCII_HEADER_REGEX = re.compile(
    br'\A# starting GPS (?P<epoch>\d+) duration (?P<duration>\d+)\Z')
LOSC_ASCII_COMMENT = br'#'

# LOSC filename re
LOSC_URL_RE = re.compile(
    r"\A((.*/)*(?P<obs>[^/]+)-"
    r"(?P<ifo>[A-Z][0-9])_LOSC_"
    r"((?P<tag>[^/]+)_)?"
    r"(?P<samp>\d+)_"
    r"(?P<version>V\d+)-"
    r"(?P<strt>[^/]+)-"
    r"(?P<dur>[^/\.]+)\."
    r"(?P<ext>[^/]+))\Z"
)
LOSC_VERSION_RE = re.compile(r'V\d+')


# -- utilities ----------------------------------------------------------------

def _parse_formats(formats, cls=TimeSeries):
    """Parse ``formats`` into a `list`, handling `None`
    """
    if formats is None:  # build list of available formats, prizing efficiency
        # include txt.gz for TimeSeries only (no state info in ASCII)
        formats = [] if cls is StateVector else ['txt.gz']

        # prefer GWF if API available
        try:
            get_default_gwf_api()
        except ImportError:
            pass
        else:
            formats.insert(0, 'gwf')

        # prefer HDF5 if h5py available
        try:
            import h5py  # pylint: disable=unused-variable
        except ImportError:
            pass
        else:
            formats.insert(0, 'hdf5')
        return formats

    if isinstance(formats, (list, tuple)):
        return formats
    return [formats]


def _download_file(url, cache=None, verbose=False):
    if cache is None:
        cache = os.getenv('GWPY_CACHE', 'no').lower() in (
            '1', 'true', 'yes', 'y',
        )
    return get_readable_fileobj(url, cache=cache, show_progress=verbose)


# -- JSON handling ------------------------------------------------------------

def _match_urls(urls, start, end, tag=None, version=None):
    """Match LOSC URLs for a given [start, end) interval
    """
    matched = {}
    matched_tags = set()

    # sort URLs by duration, then start time, then ...
    urls.sort(key=lambda u:
              os.path.splitext(os.path.basename(u))[0].split('-')[::-1])

    # format version request
    if version and not LOSC_VERSION_RE.match(str(version)):
        version = 'V{}'.format(int(version))

    # loop URLS
    for url in urls:
        try:
            m = _match_url(url, start, end, tag=tag, version=version)
        except StopIteration:
            break
        if m is None:
            continue

        mtag, mvers = m
        matched_tags.add(mtag)
        matched.setdefault(mvers, [])
        matched[mvers].append(url)

    # if multiple file tags found, and user didn't specify, error
    if len(matched_tags) > 1:
        tags = ', '.join(map(repr, matched_tags))
        raise ValueError("multiple LOSC URL tags discovered in dataset, "
                         "please select one of: {}".format(tags))

    # extract highest version
    try:
        return matched[max(matched)]
    except ValueError:  # no matched files
        return []


def _match_url(url, start, end, tag=None, version=None):
    """Match a URL against requested parameters

    Returns
    -------
    None
        if the URL doesn't match the request

    tag, version : `str`, `int`
        if the URL matches the request

    Raises
    ------
    StopIteration
        if the start time of the URL is _after_ the end time of the
        request
    """
    reg = LOSC_URL_RE.match(os.path.basename(url)).groupdict()
    if (tag and reg['tag'] != tag) or (version and reg['version'] != version):
        return

    # match times
    gps = int(reg['strt'])
    if gps >= end:  # too late, stop
        raise StopIteration

    dur = int(reg['dur'])
    if gps + dur <= start:  # too early
        return

    return reg['tag'], int(reg['version'][1:])


# -- file discovery -----------------------------------------------------------

def find_losc_urls(detector, start, end, host=io_losc.LOSC_URL,
                   sample_rate=4096, tag=None, version=None, format=None):
    """Fetch the metadata from LOSC regarding a given GPS interval
    """
    start = int(start)
    end = int(end)
    span = SegmentList([Segment(start, end)])
    formats = _parse_formats(format)

    metadata = io_losc.fetch_dataset_json(start, end, host=host)

    # find dataset that provides required data
    for dstype in sorted(metadata, key=lambda x: 0 if x == 'events' else 1):

        # work out how to get the event URLS
        if dstype == 'events':
            def _get_urls(dataset):
                return io_losc.fetch_event_json(dataset, host=host)['strain']
        elif dstype == 'runs':
            def _get_urls(dataset):
                return io_losc.fetch_run_json(dataset, detector, start, end,
                                              host=host)['strain']
        else:
            raise ValueError(
                "Unrecognised LOSC dataset type {!r}".format(dstype))

        # search datasets
        for dataset in metadata[dstype]:
            dsmeta = metadata[dstype][dataset]

            # validate IFO is present
            if detector not in dsmeta['detectors']:
                continue

            # check GPS for run datasets
            try:
                seg = Segment(dsmeta['GPSstart'], dsmeta['GPSend'])
            except KeyError:  # probably not a 'run' dataset
                pass
            else:
                if not seg.intersects(span[0]):
                    continue

            # get URL list for this dataset
            urls = _get_urls(dataset)

            for form in formats:
                # sieve URLs based on basic parameters,
                # and match tag and version
                cache = _match_urls(
                    [u['url'] for u in io_losc.sieve_urls(
                        urls, detector=detector,
                        sampling_rate=sample_rate, format=form)],
                    start, end, tag=tag, version=version)

                # if event dataset, pick shortest file that covers request
                if dstype == 'events':
                    for url in cache:
                        a, b = file_segment(url)
                        if a <= start and b >= end:
                            return [url]

                # otherwise if url list covers the full requested interval
                elif not span - cache_segments(cache):
                    return cache

    raise ValueError("Cannot find a LOSC dataset for %s covering [%d, %d)"
                     % (detector, start, end))


# -- remote file reading ------------------------------------------------------

def _fetch_losc_data_file(url, *args, **kwargs):
    """Internal function for fetching a single LOSC file and returning a Series
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
        kwargs.setdefault('format', 'hdf5.losc')
    elif ext == '.txt':
        kwargs.setdefault('format', 'ascii.losc')
    elif ext == '.gwf':
        kwargs.setdefault('format', 'gwf')

    with _download_file(url, cache, verbose=verbose) as rem:
        if verbose:
            print('Reading data...', end=' ')
        try:
            series = cls.read(rem, *args, **kwargs)
        except Exception as exc:
            if verbose:
                print('')
            exc.args = ("Failed to read LOSC data from %r: %s"
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
                except (TypeError, ValueError):  # don't care, bad LOSC
                    pass

            if verbose:
                print('[Done]')
            return series


# -- remote data access (the main event) --------------------------------------

def fetch_losc_data(detector, start, end, host=io_losc.LOSC_URL,
                    sample_rate=4096, tag=None, version=None, format=None,
                    cls=TimeSeries, **kwargs):
    """Fetch LOSC data for a given detector

    This function is for internal purposes only, all users should instead
    use the interface provided by `TimeSeries.fetch_open_data` (and similar
    for `StateVector.fetch_open_data`).
    """
    # format arguments
    sample_rate = Quantity(sample_rate, 'Hz').value
    start = to_gps(start)
    end = to_gps(end)
    span = Segment(start, end)
    formats = _parse_formats(format, cls)

    # get cache of URLS
    cache = find_losc_urls(detector, start, end, host=host,
                           sample_rate=sample_rate, tag=tag, version=version,
                           format=formats)
    if kwargs.get('verbose', False):
        print("Fetched %d URLs from %s for [%d .. %d)"
              % (len(cache), host, int(start), int(end)))

    # handle GWF requirement on channel name
    if cache[0].endswith('.gwf'):
        try:
            args = (kwargs.pop('channel'),)
        except KeyError:  # no specified channel
            if cls is StateVector:
                args = ('{}:LOSC-DQMASK'.format(detector,),)
            else:
                args = ('{}:LOSC-STRAIN'.format(detector,),)
    else:
        args = ()

    # read data
    out = None
    kwargs['cls'] = cls
    for url in cache:
        keep = file_segment(url) & span
        new = _fetch_losc_data_file(
            url, *args, **kwargs).crop(*keep, copy=False)
        if out is None:
            out = new.copy()
        else:
            out.append(new, resize=True)
    return out


# -- I/O ----------------------------------------------------------------------

@io_hdf5.with_read_hdf5
def read_losc_hdf5(h5f, path='strain/Strain',
                   start=None, end=None, copy=False):
    """Read a `TimeSeries` from a LOSC-format HDF file.

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
    nddata = dataset.value
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
def read_losc_hdf5_state(f, path='quality/simple', start=None, end=None,
                         copy=False):
    """Read a `StateVector` from a LOSC-format HDF file.

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
    nddata = dataset.value
    bits = [bytes.decode(bytes(b), 'utf-8') for b in maskset.value]
    # read metadata
    epoch = dataset.attrs['Xstart']
    try:
        dt = dataset.attrs['Xspacing']
    except KeyError:
        dt = Quantity(1, 's')
    else:
        xunit = parse_unit(dataset.attrs['Xunits'])
        dt = Quantity(dt, xunit)
    return StateVector(nddata, bits=bits, epoch=epoch, name='Data quality',
                       dx=dt, copy=copy).crop(start=start, end=end)


# register
registry.register_reader('hdf5.losc', TimeSeries, read_losc_hdf5)
registry.register_reader('hdf5.losc', StateVector, read_losc_hdf5_state)
# DEPRECATED -- remove prior to 1.0 release
registry.register_reader('losc', TimeSeries, read_losc_hdf5)
registry.register_reader('losc', StateVector, read_losc_hdf5_state)


def read_losc_ascii(fobj):
    """Read a LOSC ASCII file into a `TimeSeries`
    """
    # read file path
    if isinstance(fobj, string_types):
        with io_utils.gopen(fobj, mode='rb') as fobj2:
            return read_losc_ascii(fobj2)

    # read header to get metadata
    metadata = {}
    for line in fobj:
        if not line.startswith(LOSC_ASCII_COMMENT):  # stop iterating
            break
        match = LOSC_ASCII_HEADER_REGEX.match(line.rstrip())
        if match:  # parse metadata
            metadata.update(
                (key, float(val)) for key, val in match.groupdict().items())

    # rewind to make sure we don't miss the first data point
    fobj.seek(0)

    # work out sample_rate from metadata
    try:
        dur = float(metadata.pop('duration'))
    except KeyError:
        raise ValueError("Failed to parse data duration from LOSC ASCII file")

    data = numpy.loadtxt(fobj, dtype=float, comments=b'#', usecols=(0,))

    metadata['sample_rate'] = data.size / dur
    return TimeSeries(data, **metadata)


# ASCII
registry.register_reader('ascii.losc', TimeSeries, read_losc_ascii)
