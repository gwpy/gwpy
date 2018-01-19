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
import sys
import json
import re

from six import string_types

import numpy

from astropy.io import registry
from astropy.units import Quantity
from astropy.utils.data import get_readable_fileobj

from .gwf import get_default_gwf_api
from .. import (StateVector, TimeSeries)
from ...io import (hdf5 as io_hdf5, utils as io_utils)
from ...io.losc import (LOSC_URL, fetch_json)
from ...io.cache import (cache_segments, file_segment)
from ...detector.units import parse_unit
from ...segments import (Segment, SegmentList)
from ...time import to_gps

# ASCII parsing globals
LOSC_ASCII_HEADER_REGEX = re.compile(
    br'\A# starting GPS (?P<epoch>\d+) duration (?P<duration>\d+)\Z')
LOSC_ASCII_COMMENT = br'#'


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
            import h5py
        except ImportError:
            pass
        else:
            formats.insert(0, 'hdf5')
        return formats

    if isinstance(formats, (list, tuple)):
        return formats
    return [formats]


# -- JSON handling ------------------------------------------------------------

def _parse_losc_json(metadata, detector, sample_rate=4096,
                     format='hdf5', duration=4096):
    """Parse a list of file URLs from a LOSC metadata packet
    """
    urls = []
    for fmd in metadata:  # loop over file metadata dicts
        # skip over files we don't want
        if (fmd['detector'] != detector or
                fmd['sampling_rate'] != sample_rate or
                fmd['format'] != format or
                fmd['duration'] != duration):
            continue
        urls.append(str(fmd['url']))
    return urls


# -- file discovery -----------------------------------------------------------

def find_losc_urls(detector, start, end, host=LOSC_URL,
                   sample_rate=4096, format=None):
    """Fetch the metadata from LOSC regarding a given GPS interval
    """
    start = int(start)
    end = int(end)
    span = SegmentList([Segment(start, end)])
    formats = _parse_formats(format)

    # -- step 1: query the interval
    url = '%s/archive/%d/%d/json/' % (host, start, end)
    metadata = fetch_json(url)

    # -- step 2: try and get data from an event (smaller files)
    for form in formats:
        for dstype in ['events', 'runs']:
            for dataset in metadata[dstype]:
                # validate IFO is present
                if detector not in metadata[dstype][dataset]['detectors']:
                    continue
                # get metadata for this dataset
                if dstype == 'events':
                    url = '%s/archive/%s/json/' % (host, dataset)
                else:
                    url = ('%s/archive/links/%s/%s/%d/%d/json/'
                           % (host, dataset, detector, start, end))
                emd = fetch_json(url)
                # get cache and sieve for our segment
                for duration in [32, 4096]:  # try short files for events first
                    cache = _parse_losc_json(
                        emd['strain'], detector, sample_rate=sample_rate,
                        format=form, duration=duration)
                    cache = [u for u in cache if
                             file_segment(u).intersects(span[0])]
                    # if full span covered, return now
                    if not span - cache_segments(cache):
                        return cache
    raise ValueError("Cannot find a LOSC dataset for %s covering [%d, %d)"
                     % (detector, start, end))


# -- remote file reading ------------------------------------------------------

def _fetch_losc_data_file(url, *args, **kwargs):
    """Internal function for fetching a single LOSC file and returning a Series
    """
    cls = kwargs.pop('cls', TimeSeries)
    cache = kwargs.pop('cache', False)
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

    with get_readable_fileobj(url, cache=cache, show_progress=verbose) as rem:
        if verbose:
            print('Reading data... ')
        try:
            series = cls.read(rem, *args, **kwargs)
        except Exception as exc:
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

            return series


# -- remote data access (the main event) --------------------------------------

def fetch_losc_data(detector, start, end, host=LOSC_URL, sample_rate=4096,
                    format=None, cls=TimeSeries, **kwargs):
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
                           sample_rate=sample_rate, format=formats)
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
