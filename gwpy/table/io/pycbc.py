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

from ...io.hdf5 import (identify_hdf5, with_read_hdf5)
from ...io.registry import (register_reader, register_identifier)
from .. import (Table, EventTable)
from .utils import read_with_selection

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Nitz <alex.nitz@ligo.org>'

PYCBC_LIVE_FORMAT = 'hdf5.pycbc_live'

INVALID_COLUMNS = ['psd', 'loudest']
PYCBC_FILENAME = re.compile('([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.hdf')


@with_read_hdf5
@read_with_selection
def table_from_file(source, ifo=None, columns=None, loudest=False):
    """Read a `Table` from a PyCBC live HDF5 file
    """
    import h5py

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
        data.append(Table.Column(arr, name=c))

    return Table(data, meta=meta)


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


@with_read_hdf5
def empty_hdf5_file(h5f, ifo=None):
    """Determine whether PyCBC-HDF5 file is empty

    A file is considered empty if it contains no groups at the base level,
    or if the ``ifo`` group contains only the ``psd`` dataset.

    Parameters
    ----------
    h5f : `str`
        path of the pycbc_live file to test

    ifo : `str`, optional
        prefix for the interferometer of interest (e.g. ``'L1'``),
        include this for a more robust test of 'emptiness'

    Returns
    -------
    empty : `bool`
        `True` if the file looks to have no content, otherwise `False`
    """
    # the decorator opens the HDF5 file for us, so h5f is guaranteed to
    # be an h5py.Group object
    h5f = h5f.file
    if list(h5f) == []:
        return True
    if ifo is not None and (ifo not in h5f or list(h5f[ifo]) == ['psd']):
        return True
    return False


def identify_pycbc_live(origin, filepath, fileobj, *args, **kwargs):
    """Identify a PyCBC Live file as an HDF5 with the correct name
    """
    if identify_hdf5(origin, filepath, fileobj, *args, **kwargs) and (
            filepath is not None and PYCBC_FILENAME.match(basename(filepath))):
        return True
    return False


# register for unified I/O
register_identifier(PYCBC_LIVE_FORMAT, EventTable, identify_pycbc_live)
register_reader(PYCBC_LIVE_FORMAT, EventTable, table_from_file)

# -- processed columns --------------------------------------------------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in an `~h5py.Group` and return a `numpy.ndarray`

GET_COLUMN = {}


def _new_snr_scale(snr, redx2, q=6., n=2.):
    return (.5 * (1. + redx2 ** (q/n))) ** (-1./q)


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
