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

import numpy
from numpy import rec
from numpy.lib import recfunctions

from glue.lal import CacheEntry

from ...utils.deps import with_import
from ...io.cache import file_list
from ...io.registry import (register_reader, register_identifier)
from ..rec import GWRecArray

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

INVALID_COLUMNS = ['psd']
PYCBC_FILENAME = re.compile('([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.hdf')


@with_import('h5py')
def recarray_from_file(source, ifo=None, columns=None):
    """Read a `GWRecArray` from a PyCBC live HDF5 file
    """
    # read HDF5 file
    if isinstance(source, CacheEntry):
        source = source.path
    if isinstance(source, str):
        source = h5py.File(source, 'r')
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
    return out


def recarray_from_pycbc_live(source, ifo=None, columns=None, nproc=1):
    """Read a `GWRecArray` from one or more PyCBC live files
    """
    source = file_list(source)
    if nproc > 1:
        from ...io.cache import read_cache
        return read_cache(source, GWRecArray, nproc, None,
                          ifo=ifo, columns=columns, format='pycbc_live')

    source = filter_empty_files(source, ifo=ifo)
    arrays = [recarray_from_file(x, ifo=ifo, columns=columns) for x in source]
    return recfunctions.stack_arrays(arrays, asrecarray=True, usemask=False,
                                     autoconvert=True).view(GWRecArray)


def filter_empty_files(files, ifo=None):
    return [f for f in files if not empty_hdf5_file(f, ifo=ifo)]


@with_import('h5py')
def empty_hdf5_file(fp, ifo=None):
    if isinstance(fp, CacheEntry):
        fp = fp.path
    if isinstance(fp, str):
        h5f = h5py.File(fp, 'r')
    elif isinstance(fp, h5py.File):
        h5f = fp
    else:
        return  # default to something that will evaluate as false
    if list(h5f) == []:
        return True
    if ifo is not None and list(h5f[ifo]) == ['psd']:
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
