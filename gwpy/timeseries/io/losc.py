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

import sys
import json
from tempfile import NamedTemporaryFile

from six.moves.urllib.request import urlopen

from glue.lal import (Cache, CacheEntry)

from astropy.io import registry
from astropy.units import Quantity

from .. import (StateVector, TimeSeries)
from ...utils.deps import with_import
from ...io.cache import (file_list, cache_segments)
from ...io.hdf5 import open_hdf5
from ...detector.units import parse_unit
from ...segments import (Segment, SegmentList)
from ...time import (to_gps, LIGOTimeGPS)

# default URL
LOSC_URL = 'https://losc.ligo.org'


# -- remote URL discovery -----------------------------------------------------

def _fetch_losc_json(url):
    # fetch the URL
    try:
        response = urlopen(url)
    except (IOError, Exception) as e:
        e.args = ("Failed to access LOSC metadata from %r: %s"
                  % (url, str(e)),)
        raise
    # parse the JSON
    data = response.read()
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    try:
        return json.loads(data)
    except ValueError as e:
        e.args = ("Failed to parse LOSC JSON from %r: %s"
                  % (url, str(e)),)
        raise


def _losc_json_cache(metadata, detector, sample_rate=4096,
                     format='hdf5', duration=4096):
    """Parse a :class:`~glue.lal.Cache` from a LOSC metadata packet
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
    return Cache.from_urls(urls, coltype=LIGOTimeGPS)


def fetch_losc_url_cache(detector, start, end, host=LOSC_URL,
                         sample_rate=4096, format='hdf5'):
    """Fetch the metadata from LOSC regarding a given GPS interval
    """
    start = int(start)
    end = int(end)
    span = SegmentList([Segment(start, end)])
    # -- step 1: query the interval
    url = '%s/archive/%d/%d/json/' % (host, start, end)
    md = _fetch_losc_json(url)
    # -- step 2: try and get data from an event (smaller files)
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
                cache = _losc_json_cache(
                    emd['strain'], detector, sample_rate=sample_rate,
                    format=format, duration=duration).sieve(segmentlist=span)
                # if full span covered, return now
                if not span - cache_segments(cache):
                    return cache
    raise ValueError("Cannot find a LOSC dataset for %s covering [%d, %d)"
                     % (detector, start, end))


def fetch_losc_data(detector, start, end, host=LOSC_URL,
                    channel='strain/Strain', sample_rate=4096, cls=TimeSeries,
                    verbose=False):
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
    cache = fetch_losc_url_cache(detector, start, end, host=host,
                                 sample_rate=sample_rate)
    if verbose:
        print("Fetched list of %d URLs to read from %s for [%d .. %d)"
              % (len(cache), host, int(start), int(end)))
    # read data
    out = None
    for e in cache:
        keep = e.segment & span
        new = _fetch_losc_data_file(e.url, host=host, channel=channel, cls=cls,
                                    verbose=verbose).crop(*keep, copy=False)
        if out is None:
            out = new.copy()
        else:
            out.append(new, resize=True)
    return out

    # panic
    raise ValueError("%s data for %s not available in full from LOSC"
                     % (detector, span))


def _fetch_losc_data_file(url, host=LOSC_URL, channel='strain/Strain',
                          cls=TimeSeries, verbose=False):
    """Internal function for fetching a single LOSC file and returning a Series
    """
    if verbose:
        print("Reading %s..." % url, end=' ')
        sys.stdout.flush()
    try:
        response = urlopen(url)
    except Exception as e:
        if verbose:
            print("")
        e.args = ("Failed to download LOSC data from %r: %s"
                  % (url, str(e)),)
        raise
    with NamedTemporaryFile() as f:
        f.write(response.read())
        f.seek(0)
        try:
            return cls.read(f.name, channel, format='losc')
        except Exception as e:
            if verbose:
                print("")
            e.args = ("Failed to read HDF-format LOSC data from %r: %s"
                      % (url, str(e)),)
            raise
        finally:
            if verbose:
                print(" Done")


# -- I/O ----------------------------------------------------------------------

def read_losc_data(filename, channel, group=None, copy=False):
    """Read a `TimeSeries` from a LOSC-format HDF file.

    Parameters
    ----------
    filename : `str`
        path to LOSC-format HDF5 file to read.
    channel : `str`
        name of HDF5 dataset to read.
    group : `str`, optional
        name of containing HDF5 group for ``channel``. If not given,
        the first dataset named ``channel`` will be assumed as the right
        one.
    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data

    Returns
    -------
    data : :class`~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    h5file = open_hdf5(filename)
    if group:
        channel = '%s/%s' % (group, channel)
    dataset = _find_dataset(h5file, channel)
    # read data
    nddata = dataset.value
    # read metadata
    xunit = parse_unit(dataset.attrs['Xunits'])
    epoch = dataset.attrs['Xstart']
    dt = Quantity(dataset.attrs['Xspacing'], xunit)
    unit = dataset.attrs['Yunits']
    # build and return
    return TimeSeries(nddata, epoch=epoch, sample_rate=(1/dt).to('Hertz'),
                      unit=unit, name=channel.rsplit('/', 1)[0], copy=copy)


def read_losc_data_cache(f, channel, start=None, end=None, resample=None,
                         group=None, target=TimeSeries):
    """Read a `TimeSeries` from a LOSC-format HDF file.

    Parameters
    ----------
    source : `str`, `list`, `~glue.lal.Cache`
        path to LOSC-format HDF5 file to read or cache of many files.
    channel : `str`
        name of HDF5 dataset to read.
    group : `str`, optional
        name of containing HDF5 group for ``channel``. If not given,
        the first dataset named ``channel`` will be assumed as the right
        one.
    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data

    Returns
    -------
    data : :class`~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    files = file_list(f)

    out = None
    for fp in files:
        if target is TimeSeries:
            new = read_losc_data(fp, channel, group=group, copy=False)
        elif target is StateVector:
            new = read_losc_state(fp, channel, group=group, copy=False)
        else:
            raise ValueError("Cannot read %s from LOSC data"
                             % (target.__name__))
        if out is None:
            out = new.copy()
        else:
            out.append(new)

    if resample:
        out = out.resample(resample)

    if start or end:
        out = out.crop(start=start, end=end)

    return out


def read_losc_state(filename, channel, group=None, start=None, end=None,
                    copy=False):
    """Read a `StateVector` from a LOSC-format HDF file.

    Parameters
    ----------
    filename : `str`
        path to LOSC-format HDF5 file to read.
    channel : `str`
        name of HDF5 dataset to read.
    group : `str`, optional
        name of containing HDF5 group for ``channel``. If not given,
        the first dataset named ``channel`` will be assumed as the right
        one.
    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data
    copy : `bool`, default: `False`
        create a fresh-memory copy of the underlying array

    Returns
    -------
    data : :class`~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    h5file = open_hdf5(filename)
    if group:
        channel = '%s/%s' % (group, channel)
    # find data
    dataset = _find_dataset(h5file, '%s/DQmask' % channel)
    maskset = _find_dataset(h5file, '%s/DQDescriptions' % channel)
    # read data
    nddata = dataset.value
    bits = list(map(lambda b: bytes.decode(bytes(b), 'utf-8'), maskset.value))
    # read metadata
    try:
        epoch = dataset.attrs['Xstart']
    except KeyError:
        try:
            ce = CacheEntry.from_T050017(h5file.filename, coltype=LIGOTimeGPS)
        except ValueError:
            epoch = None
        else:
            epoch = ce.segment[0]
    try:
        dt = dataset.attrs['Xspacing']
    except KeyError:
        dt = Quantity(1, 's')
    else:
        xunit = parse_unit(dataset.attrs['Xunits'])
        dt = Quantity(dt, xunit)
    return StateVector(nddata, bits=bits, epoch=epoch, name='Data quality',
                       dx=dt, copy=copy)


def read_losc_state_cache(*args, **kwargs):
    """Read a `StateVector` from a LOSC-format HDF file.

    Parameters
    ----------
    source : `str`, `list`, :class:`glue.lal.Cache`
        path to LOSC-format HDF5 file to read or cache of many files.
    channel : `str`
        name of HDF5 dataset to read.
    group : `str`, optional
        name of containing HDF5 group for ``channel``. If not given,
        the first dataset named ``channel`` will be assumed as the right
        one.
    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data


    Returns
    -------
    data : :class:`~gwpy.timeseries.statevector.StateVector`
        a new `TimeSeries` containing the data read from disk
    """
    kwargs.setdefault('target', StateVector)
    return read_losc_data_cache(*args, **kwargs)


@with_import('h5py')
def _find_dataset(h5group, name):
    """Find the named :class:`h5py.Dataset` in an HDF file.

    Parameters
    ----------
    h5group : :class:`h5py.File`, :class:`h5py.Group`
        open HDF file or group
    name : `str`
        name of :class:`h5py.Dataset` to find

    Returns
    -------
    data : :class:`h5ile.Dataset`
        HDF dataset
    """
    # find dataset directly
    if not isinstance(h5group, h5py.Group):
        raise ValueError("_find_dataset must be handed a h5py.Group object, "
                         "not %s" % h5group.__class__.__name__)
    if name in h5group and isinstance(h5group[name], h5py.Dataset):
        return h5group[name]
    # otherwise trawl through member groups
    for group in h5group.values():
        try:
            return _find_dataset(group, name)
        except ValueError:
            continue
    raise ValueError("Cannot find channel '%s' in file HDF object" % name)


registry.register_reader('losc', TimeSeries, read_losc_data_cache)
registry.register_reader('losc', StateVector, read_losc_state_cache)
