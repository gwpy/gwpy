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

"""Read events from the PyCBC live online GW search
"""

import re
from os.path import basename

from six import string_types

try:
    import h5py
except ImportError:
    HAS_H5PY = False
else:
    HAS_H5PY = True

from glue.lal import CacheEntry

from astropy.table import vstack as vstack_tables

from ...utils.deps import with_import
from ...io.cache import file_list
from ...io.registry import (register_reader, register_identifier)
from .. import EventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Nitz <alex.nitz@ligo.org>'

PYCBC_LIVE_FORMAT = 'hdf5.pycbc_live'

INVALID_COLUMNS = ['psd', 'loudest']
PYCBC_FILENAME = re.compile('([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.hdf')


@with_import('h5py')
def _table_from_file(source, ifo=None, columns=None, loudest=False):
    """Read a `Table` from a single PyCBC live HDF5 file

    This method is for internal use only.
    """
    close = False  # do we need to close the file when we're done

    # read HDF5 file
    if isinstance(source, CacheEntry):
        source = source.path
    if isinstance(source, str):
        h5file = source = h5py.File(source, 'r')
        close = True

    try:
        # find group
        if isinstance(source, h5py.File):
            if ifo is None:
                try:
                    ifo, = [key for key in list(source) if key != 'background']
                except ValueError as e:
                    e.args = ("PyCBC live HDF5 file contains dataset groups "
                              "for multiple interferometers, please specify "
                              "the prefix of the relevant interferometer via "
                              "the `ifo` keyword argument, e.g: `ifo=G1`",)
                    raise
            try:
                source = source[ifo]
            except KeyError as e:
                e.args = ("No group for ifo %r in PyCBC live HDF5 file" % ifo,)
                raise
        # at this stage, 'source' should be an HDF5 group in pycbc_live format
        if columns is None:
            columns = [c for c in source if c not in INVALID_COLUMNS]

        # set up meta dict
        meta = {'ifo': ifo}

        # record loudest in meta
        try:
            meta['loudest'] = source['loudest'][:]
        except KeyError:
            if loudest:
                raise

        # record PSD in meta
        try:
            psd = source['psd']
        except KeyError:
            pass
        else:
            from gwpy.frequencyseries import FrequencySeries
            df = psd.attrs['delta_f']
            meta['psd'] = FrequencySeries(
                psd[:], f0=0, df=df, name='pycbc_live')

        # map data to columns
        data = []
        get_ = []
        for c in columns:
            # convert hdf5 dataset into Column
            try:
                arr = source[c][:]
            except KeyError:
                if c in GET_COLUMN:
                    arr = GET_COLUMN[c](source)
                else:
                    raise
            if loudest:
                arr = arr[meta['loudest']]
            data.append(EventTable.Column(arr, name=c))
    finally:
        if close:
            h5file.close()

    return EventTable(data, meta=meta)


def table_from_pycbc_live(source, ifo=None, columns=None, **kwargs):
    """Read a `GWRecArray` from one or more PyCBC live files
    """
    source = file_list(source)
    source = filter_empty_files(source, ifo=ifo)
    return vstack_tables(
        [_table_from_file(x, ifo=ifo, columns=columns, **kwargs)
         for x in source])


def filter_empty_files(files, ifo=None):
    """Remove empty PyCBC-HDF5 files from a list

    Parameters
    ----------
    files : `list` of `str`, :class:`~glue.lal.Cache`
        a list of file paths to test

    ifo : `str`, optional
        prefix for the interferometer of interest (e.g. ``'L1'``),
        include this for a more robust test of 'emptiness'

    Returns
    -------
    nonempty : `list`
        the subset of the input ``files`` that are considered not empty

    See also
    --------
    empty_hdf5_file
        for details of the 'emptiness' test
    """
    return type(files)([f for f in files if not empty_hdf5_file(f, ifo=ifo)])


@with_import('h5py')
def empty_hdf5_file(fp, ifo=None):
    """Determine whether PyCBC-HDF5 file is empty

    A file is considered empty if it contains no groups at the base level,
    or if the ``ifo`` group contains only the ``psd`` dataset.

    Parameters
    ----------
    fp : `str`
        path of the pycbc_live file to test

    ifo : `str`, optional
        prefix for the interferometer of interest (e.g. ``'L1'``),
        include this for a more robust test of 'emptiness'

    Returns
    -------
    empty : `bool`
        `True` if the file looks to have no content, otherwise `False`
    """
    if isinstance(fp, CacheEntry):
        fp = fp.path
    with h5py.File(fp, 'r') as h5f:
        if list(h5f) == []:
            return True
        if ifo is not None and list(h5f[ifo]) == ['psd']:
            return True
        return False


def identify_pycbc_live(origin, filepath, fileobj, *args, **kwargs):
    """Identify a PyCBC Live file from its basename
    """
    if HAS_H5PY and isinstance(filepath, h5py.HLObject):
        filepath = filepath.file.name
    if (isinstance(filepath, string_types) and
            PYCBC_FILENAME.match(basename(filepath))):
        return True
    return False

# register for unified I/O
register_identifier(PYCBC_LIVE_FORMAT, EventTable, identify_pycbc_live)
register_reader(PYCBC_LIVE_FORMAT, EventTable, table_from_pycbc_live)

# -- processed columns --------------------------------------------------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in an `~h5py.Group` and return a `numpy.ndarray`

GET_COLUMN = {}


def get_new_snr(h5group, q=6., n=2.):
    newsnr = h5group['snr'][:].copy()
    rchisq = h5group['chisq'][:]
    idx = numpy.where(rchisq > 1.)[0]
    newsnr[idx] *= _new_snr_scale(newsnr[idx], rchisq[idx], q=q, n=n)
    return newsnr

GET_COLUMN['new_snr'] = get_new_snr


def get_mchirp(h5group):
    mass1 = h5group['mass1'][:]
    mass2 = h5group['mass2'][:]
    return (mass1 * mass2) ** (3/5.) / (mass1 + mass2) ** (1/5.)

GET_COLUMN['mchirp'] = get_mchirp


# -----------------------------------------------------------------------------
#
# -- DEPRECATED - remove before 1.0 release -----------------------------------
#
# -----------------------------------------------------------------------------

import re
from os.path import basename

import numpy
from numpy import rec
from numpy.lib import recfunctions

from glue.lal import CacheEntry

from ...utils.deps import with_import
from ...io.cache import file_list
from ...io.registry import (register_reader, register_identifier)
from ..rec import GWRecArray

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

INVALID_COLUMNS = ['psd', 'loudest']
PYCBC_FILENAME = re.compile('([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.hdf')


@with_import('h5py')
def recarray_from_file(source, ifo=None, columns=None, loudest=False):
    """Read a `GWRecArray` from a PyCBC live HDF5 file
    """
    # read HDF5 file
    if isinstance(source, CacheEntry):
        source = source.path
    if isinstance(source, str):
        h5f = source = h5py.File(source, 'r')
        opened = True
    else:
        opened = False
    # find group
    if isinstance(source, h5py.File):
        if ifo is None:
            try:
                ifo, = list(source)
            except ValueError as e:
                e.args = ("PyCBC live HDF5 file contains multiple IFO groups, "
                          "please select ifo manually",)
                raise
        try:
            source = source[ifo]
        except KeyError as e:
            e.args = ("No group for ifo %r in PyCBC live HDF5 file" % ifo,)
            raise
    # at this stage, 'source' should be an HDF5 group in the pycbc live format
    if columns is None:
        columns = [c for c in source if c not in INVALID_COLUMNS]
    names, data = zip(*[(k, source[k][:]) for k in source if k in columns])
    names = list(map(str, names))
    if loudest:  # recover only the 'loudest' events
        loudest = source['loudest'][:]
        data = [d[loudest] for d in data]
    else:
        data = list(data)
    # calculate new_snr on-the-fly
    if 'new_snr' in columns and 'new_snr' not in source:
        # get columns needed for newsnr
        snr = data[names.index('snr')]
        rchisq = data[names.index('chisq')]  # chisq is already reduced
        # calculate and append to column list
        data.append(get_new_snr(snr, rchisq))
        names.append('new_snr')
    # calculate mchirp
    if 'mchirp' in columns and 'mchirp' not in source:
        mass1 = data[names.index('mass1')]
        mass2 = data[names.index('mass2')]
        data.append(get_mchirp(mass1, mass2))
        names.append('mchirp')
    # read columns into numpy recarray
    out = rec.fromarrays(data, names=map(str, names)).view(GWRecArray)
    if 'end_time' in columns:
        out.sort(order='end_time')
    if opened:
        h5f.close()
    return out


def recarray_from_pycbc_live(source, ifo=None, columns=None, nproc=1, **kwargs):
    """Read a `GWRecArray` from one or more PyCBC live files
    """
    source = file_list(source)
    if nproc > 1:
        from ...io.cache import read_cache
        return read_cache(source, GWRecArray, nproc, None,
                          ifo=ifo, columns=columns, format='pycbc_live',
                          **kwargs)

    source = filter_empty_files(source, ifo=ifo)
    arrays = [recarray_from_file(x, ifo=ifo, columns=columns, **kwargs)
              for x in source]
    return recfunctions.stack_arrays(arrays, asrecarray=True, usemask=False,
                                     autoconvert=True).view(GWRecArray)


def filter_empty_files(files, ifo=None):
    return [f for f in files if not empty_hdf5_file(f, ifo=ifo)]


@with_import('h5py')
def empty_hdf5_file(fp, ifo=None):
    if isinstance(fp, CacheEntry):
        fp = fp.path
    if not isinstance(fp, str):
        return  # default to something that will evaluate as false
    with h5py.File(fp, 'r') as h5f:
        if list(h5f) == []:
            return True
        if ifo is not None and (ifo not in h5f or list(h5f[ifo]) == ['psd']):
            return True
        return False


def identify_pycbc_live(origin, path, fileobj, *args, **kwargs):
    """Identify a PyCBC Live file from its basename
    """
    if path is not None and PYCBC_FILENAME.match(basename(path)):
        return True
    return False


def get_new_snr(snr, reduced_x2, q=6., n=2.):
    newsnr = snr.copy()
    # map arrays
    if isinstance(newsnr, numpy.ndarray) and newsnr.shape:
        idx = numpy.where(reduced_x2 > 1.)[0]
        newsnr[idx] *= _new_snr_scale(newsnr[idx], reduced_x2[idx], q=q, n=n)
    # map single value if reduced_x2 > 1
    elif reduced_x2 > 1.:
        return newsnr * _new_snr_scale(newsnr, reduced_x2, q=q, n=n)
    return newsnr


def _new_snr_scale(snr, redx2, q=6., n=2.):
    return (.5 * (1. + redx2 ** (q/n))) ** (-1./q)


def get_mchirp(mass1, mass2):
    return (mass1 * mass2) ** (3/5.) / (mass1 + mass2) ** (1/5.)


# register for unified I/O
register_identifier('pycbc_live', GWRecArray, identify_pycbc_live)
register_reader('pycbc_live', GWRecArray, recarray_from_pycbc_live)
