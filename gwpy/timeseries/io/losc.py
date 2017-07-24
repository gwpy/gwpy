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

from .. import (StateVector, TimeSeries)
from ...io import (hdf5 as io_hdf5, utils as io_utils)
from ...io.cache import (cache_segments, file_segment)
from ...detector.units import parse_unit
from ...segments import (Segment, SegmentList)
from ...time import to_gps

# default URL
LOSC_URL = 'https://losc.ligo.org'


# -- JSON handling ------------------------------------------------------------

def _fetch_losc_json(url):
    with get_readable_fileobj(url, show_progress=False,
                              encoding='utf-8') as response:
        data = response.read()
        try:
            return json.loads(data)
        except ValueError as e:
            e.args = ("Failed to parse LOSC JSON from %r: %s"
                      % (url, str(e)),)
            raise


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
    if format is None:
        formats = ['txt.gz', 'hdf5', 'gwf']
    elif isinstance(format, string_types):
        formats = [format]
    else:
        formats = format

    # -- step 1: query the interval
    url = '%s/archive/%d/%d/json/' % (host, start, end)
    md = _fetch_losc_json(url)

    # -- step 2: try and get data from an event (smaller files)
    for form in formats:
        for dstype in ['events', 'runs']:
            for dataset in md[dstype]:
                # validate IFO is present
                if detector not in md[dstype][dataset]['detectors']:
                    continue
                # get metadata for this dataset
                if dstype == 'events':
                    url = '%s/archive/%s/json/' % (host, dataset)
                else:
                    url = ('%s/archive/links/%s/%s/%d/%d/json/'
                           % (host, dataset, detector, start, end))
                emd = _fetch_losc_json(url)
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

def _fetch_losc_data_file(url, cls=TimeSeries, verbose=False, **kwargs):
    """Internal function for fetching a single LOSC file and returning a Series
    """
    if verbose:
        print("Reading %s..." % url, end=' ')
        sys.stdout.flush()

    # match file format
    if url.endswith('.gz'):
        ext = os.path.splitext(url[:-3])[-1]
    else:
        ext = os.path.splitext(url)[-1]
    if ext == '.hdf5':
        kwargs.setdefault('format', 'hdf5.losc')
    elif ext == '.txt':
        kwargs.setdefault('format', 'ascii.losc')

    with get_readable_fileobj(url, show_progress=False) as remote:
        try:
            return cls.read(remote, **kwargs)
        except Exception as e:
            if verbose:
                print("")
            e.args = ("Failed to read LOSC data from %r: %s" % (url, str(e)),)
            raise
        finally:
            if verbose:
                print(" Done")


# -- remote data access (the main event) --------------------------------------

def fetch_losc_data(detector, start, end, host=LOSC_URL, sample_rate=4096,
                    format='hdf5', cls=TimeSeries, verbose=False, **kwargs):
    """Fetch LOSC data for a given detector

    This function is for internal purposes only, all users should instead
    use the interface provided by `TimeSeries.fetch_open_data` (and similar
    for `StateVector.fetch_open_data`).
    """
    sample_rate = Quantity(sample_rate, 'Hz').value
    start = to_gps(start)
    end = to_gps(end)
    span = Segment(start, end)
    # get cache of URLS
    cache = find_losc_urls(detector, start, end, host=host,
                           sample_rate=sample_rate, format=format)
    if verbose:
        print("Fetched list of %d URLs to read from %s for [%d .. %d)"
              % (len(cache), host, int(start), int(end)))
    # read data
    out = None
    for url in cache:
        keep = file_segment(url) & span
        new = _fetch_losc_data_file(url, cls=cls, verbose=verbose,
                                    **kwargs).crop(*keep, copy=False)
        if out is None:
            out = new.copy()
        else:
            out.append(new, resize=True)
    return out

    # panic
    raise ValueError("%s data for %s not available in full from LOSC"
                     % (detector, span))


# -- I/O ----------------------------------------------------------------------

@io_hdf5.with_read_hdf5
def read_losc_hdf5(f, path='strain/Strain', start=None, end=None, copy=False):
    """Read a `TimeSeries` from a LOSC-format HDF file.

    Parameters
    ----------
    f : `str`, `h5py.HLObject`
        path of HDF5 file, or open `H5File`

    path : `str`
        name of HDF5 dataset to read.

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    dataset = io_hdf5.find_dataset(f, path)
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
    bits = list(map(lambda b: bytes.decode(bytes(b), 'utf-8'), maskset.value))
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

re_losc_ascii_header = re.compile('\A# starting GPS (?P<epoch>\d+) '
                                  'duration (?P<duration>\d+)\Z')


def read_losc_ascii(fobj):
    """Read a LOSC ASCII file into a `TimeSeries`
    """
    # read file path
    if isinstance(fobj, string_types):
        with io_utils.gopen(fobj) as f:
            return read_losc_ascii(f)

    # read header to get metadata
    metadata = {}
    for line in fobj:
        if not line.startswith('#'):  # stop iterating, and rewind one line
            break
        if line.startswith('# starting GPS'):  # parse metadata
            m = re_losc_ascii_header.match(line.rstrip('\n'))
            if m:
                metadata.update(m.groupdict())

    # rewind to make sure we don't miss the first data point
    fobj.seek(0)

    # work out sample_rate from metadata
    try:
        dur = float(metadata.pop('duration'))
    except KeyError:
        raise ValueError("Failed to parse data duration from LOSC ASCII file")

    data = numpy.loadtxt(fobj, dtype=float, comments='#', usecols=0)

    metadata['sample_rate'] = data.size / dur
    return TimeSeries(data, **metadata)


# ASCII
registry.register_reader('ascii.losc', TimeSeries, read_losc_ascii)
