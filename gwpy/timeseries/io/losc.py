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

from tempfile import NamedTemporaryFile

from six.moves.urllib.request import urlopen

from glue.lal import CacheEntry

from astropy.io import registry
from astropy.units import Quantity

from gwpy.utils.compat import OrderedDict

from .. import (StateVector, TimeSeries, TimeSeriesList)
from ...utils.deps import with_import
from ...io.cache import file_list
from ...io.hdf5 import open_hdf5
from ...detector.units import parse_unit
from ...segments import Segment

# -- document LOSC data sets
# each set is keyed with (name, data-rate, file-duration)

# science runs
RUN_DATA = OrderedDict()
RUN_DATA[('S5', 4096, 4096)] = Segment(815155213, 875232014)
RUN_DATA[('S6', 4096, 4096)] = Segment(931035615, 971622015)

# event data
EVENT_DATA = OrderedDict()
EVENT_DATA[('GW150914', 4096, 32)] = Segment(1126259446, 1126259478)
EVENT_DATA[('GW150914', 4096, 4096)] = Segment(1126257414, 1126261510)
EVENT_DATA[('GW150914', 16384, 32)] = Segment(1126259446, 1126259478)
EVENT_DATA[('GW150914', 16384, 4096)] = Segment(1126257414, 1126261510)

# all data (for convenience)
AVAILABLE_DATA = OrderedDict()
for k, v in EVENT_DATA.items() + RUN_DATA.items():
    AVAILABLE_DATA[k] = v

# default URL
LOSC_URL = 'https://losc.ligo.org'


def fetch_losc_data(detector, start, end, host=LOSC_URL,
                    channel='strain/Strain', sample_rate=4096, cls=TimeSeries):
    """Fetch LOSC data for a given detector

    This function is for internal purposes only, all users should instead
    use the interface provided by `TimeSeries.fetch_open_data` (and similar
    for `StateVector.fetch_open_data`).
    """
    sample_rate = Quantity(sample_rate, 'Hz').value
    span = Segment(start, end)
    for dataset in AVAILABLE_DATA:
        epoch, rate, duration = dataset
        # first match rate
        if sample_rate != rate:
            continue
        # then match availability
        try:
            inthisset = span & AVAILABLE_DATA[dataset] == span
        except ValueError:  # overlap is zero or >1 segments
            inthisset = False
        # if not completely contained, try the next epoch
        if not inthisset:
            continue
        # from here we should be able to get the data
        if dataset in EVENT_DATA:
            s = EVENT_DATA[dataset][0]
        else:
            s = start & (0xFFFFFFFF - duration + 1)
        out = None
        # loop over all predicted file times, appending as we go
        while s < end:
            keep = Segment(s, s + duration) & span
            new = _fetch_losc_data_file(
                detector, s, dataset, host=host,
                channel=channel, cls=cls).crop(*keep)
            if out is None:
                out = new
            else:
                out.append(new, resize=True, copy=False)
            s += 4096
        return out

    # panic
    raise ValueError("%s data for %s not available in full from LOSC"
                     % (detector, span))


def _fetch_losc_data_file(detector, gps, dataset, host=LOSC_URL,
                          channel='strain/Strain', cls=TimeSeries):
    """Internal function for fetching a single LOSC file and returning a Series
    """
    epoch, rate, duration = dataset
    ratestr = int(rate / 1024.)
    filename = '%s-%s_LOSC_%s_V1-%s-%s.hdf5' % (
        detector[0], detector, ratestr, gps, duration)
    if dataset in RUN_DATA:
        dgps = gps & 0xFFF00000  # GPS of directory start
        url = '%s/archive/data/%s/%s/%s' % (host, epoch, dgps, filename)
    elif dataset in EVENT_DATA:
        url = '%s/s/events/%s/%s' % (host, epoch, filename)
    else:
        raise ValueError("Dataset %r not found" % dataset)
    try:
        response = urlopen(url)
    except Exception as e:
        e.args = ("Failed to download LOSC data from %r: %s"
                  % (url, str(e)),)
        raise
    with NamedTemporaryFile() as f:
        f.write(response.read())
        f.seek(0)
        try:
            return cls.read(f.name, channel, format='losc')
        except Exception as e:
            e.args = ("Failed to read HDF-format LOSC data from %r: %s"
                      % (url, str(e)),)
            raise


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
    bits = list(maskset.value)
    # read metadata
    try:
        epoch = dataset.attrs['Xstart']
    except KeyError:
        try:
            ce = CacheEntry.from_T050017(h5file.filename)
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
