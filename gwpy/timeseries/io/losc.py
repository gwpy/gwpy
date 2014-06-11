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

import h5py

from glue.lal import (Cache, CacheEntry)

from astropy.io import registry
from astropy.units import (Unit, Quantity)

from .. import (StateVector, TimeSeries, TimeSeriesList)
from ...io.cache import file_list


def open_hdf5(filename):
    """Wrapper to open a :class:`h5py.File` from disk, gracefully
    handling a few corner cases
    """
    if isinstance(filename, h5py.File):
        return filename
    elif isinstance(filename, file):
        return h5py.File(filename.name, 'r')
    else:
        return h5py.File(filename, 'r')


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
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data

    Returns
    -------
    data : :class`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    h5file = open_hdf5(filename)
    if group:
        channel = '%s/%s' % (group, channel)
    dataset = _find_dataset(h5file, channel)
    # read data
    nddata = dataset.value
    # read metadata
    xunit = Unit(dataset.attrs['Xunits'])
    epoch = dataset.attrs['Xstart']
    dt = Quantity(dataset.attrs['Xspacing'], xunit)
    unit = Unit(dataset.attrs['Yunits'])
    # build and return
    return TimeSeries(nddata, epoch=epoch, sample_rate=(1/dt).to('Hertz'),
                      unit=unit, name=channel.rsplit('/', 1)[0], copy=copy)


def read_losc_data_cache(f, channel, start=None, end=None, resample=None,
                         group=None, target=TimeSeries):
    """Read a `TimeSeries` from a LOSC-format HDF file.

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
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data

    Returns
    -------
    data : :class`~gwpy.timeseries.core.TimeSeries`
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


def read_losc_state(filename, channel, group=None, start=None, end=None):
    """Read a `StateVector` from a LOSC-format HDF file.
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
            from glue.lal import CacheEntry
        except ImportError:
            epoch = None
        else:
            ce = CacheEntry.from_T050017(h5file.filename)
            epoch = ce.segment[0]
    try:
        dt = dataset.attrs['Xspacing']
    except KeyError:
        dt = Quantity(1, 's')
    else:
        xunit = Unit(dataset.attrs['Xunit'])
        dt = Quantity(dt, xunit)
    return StateVector(nddata, bits=bits, epoch=epoch,
                       sample_rate=(1/dt).to('Hertz'), name='Data quality')


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
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data


    Returns
    -------
    data : :class:`~gwpy.timeseries.statevector.StateVector`
        a new `TimeSeries` containing the data read from disk
    """
    kwargs.setdefault('target', StateVector)
    return read_losc_data_cache(*args, **kwargs)


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


def identify_losc(*args, **kwargs):
    """Identify an input file as LOSC HDF based on its filename
    """
    filename = args[1]
    if (isinstance(filename, (unicode, str)) and
            (filename.endswith('hdf') or filename.endswith('hdf5'))):
        return True
    else:
        return False


registry.register_reader('losc', TimeSeries, read_losc_data_cache)
registry.register_reader('losc', StateVector, read_losc_state_cache)
