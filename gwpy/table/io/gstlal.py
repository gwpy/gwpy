# -*- coding: utf-8 -*-
# Copyright (C) California Institute of Technology (2019-2022)
#               Pensylvania State University (2019)
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

"""Read events from the GstLAL online GW search
"""

import re
from os.path import basename

from astropy.table import join

from ...io.registry import (register_reader, register_identifier)
from ...io.ligolw import is_ligolw
from .ligolw import read_table
from .. import EventTable
from .pycbc import get_mchirp

__author__ = 'Derk Davis <derek.davis@ligo.org>'
__credits__ = 'Patrick Godwin <patrick.godwin@ligo.org>'

GSTLAL_FORMAT = 'ligolw.gstlal'
GSTLAL_SNGL_FORMAT = 'ligolw.gstlal.sngl'
GSTLAL_COINC_FORMAT = 'ligolw.gstlal.coinc'

GSTLAL_FILENAME = re.compile('([A-Z][0-9])+-LLOID-[0-9.]+-[0-9.]+.xml.gz')


# singles format
def read_gstlal_sngl(source, **kwargs):
    """Read a `sngl_inspiral` table from one or more GstLAL LIGO_LW XML files

    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document` objects

    **kwargs
        keyword arguments for the read, or conversion functions

    See also
    --------
    gwpy.io.ligolw.read_table
        for details of keyword arguments for the read operation
    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation
    """
    from ligo.lw import lsctables
    extra_cols = []
    derived_cols = []
    val_col = lsctables.TableByName['sngl_inspiral'].validcolumns
    if 'columns' in kwargs:
        for name in kwargs['columns'].copy():
            if name in GET_COLUMN:
                derived_cols.append(name)
                kwargs['columns'].remove(name)
                required_cols = GET_COLUMN_EXTRA[name]
                missing_cols = [c for c in required_cols
                                if c not in kwargs['columns']]
                for r_col in missing_cols:
                    kwargs['columns'].append(r_col)
                    extra_cols.append(r_col)
            elif name not in val_col:
                name_list = list(val_col.keys())+list(GET_COLUMN.keys())
                raise ValueError(f"'{name}' is not a valid column name. "
                                 f"Valid column names: {name_list}")
    events = read_table(source, tablename='sngl_inspiral', **kwargs)
    for col_name in derived_cols:
        col_data = GET_COLUMN[col_name](events)
        events.add_column(col_data, name=col_name)
    for col_name in extra_cols:
        events.remove_column(col_name)
    return events


# coinc format
def read_gstlal_coinc(source, **kwargs):
    """Read a `Table` containing coincident event information
    from one or more GstLAL LIGO_LW XML files

    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document` objects

    **kwargs
        keyword arguments for the read, or conversion functions

    See also
    --------
    gwpy.io.ligolw.read_table
        for details of keyword arguments for the read operation
    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation
    """
    from ligo.lw import lsctables
    extra_cols = []
    if 'columns' in kwargs:
        columns = kwargs['columns']
        kwargs.pop('columns')
        val_col_inspiral = lsctables.TableByName['coinc_inspiral'].validcolumns
        val_col_event = lsctables.TableByName['coinc_event'].validcolumns
        for name in columns:
            if (name not in val_col_inspiral) and (name not in val_col_event):
                name_list = list(val_col_inspiral.keys()) + \
                    list(val_col_event.keys())
                raise ValueError(f"'{name}' is not a valid column name. "
                                 f"Valid column names: {name_list}")
        if 'coinc_event_id' not in columns:
            columns.append('coinc_event_id')
            extra_cols.append('coinc_event_id')
        inspiral_cols = [col for col in columns if col in val_col_inspiral]
        event_cols = [col for col in columns if col in val_col_event]
        inspiral_cols.append('coinc_event_id')
        coinc_inspiral = read_table(source, tablename='coinc_inspiral',
                                    columns=inspiral_cols, **kwargs)
        coinc_event = read_table(source, tablename='coinc_event',
                                 columns=event_cols, **kwargs)
    else:
        coinc_inspiral = read_table(source, tablename='coinc_inspiral',
                                    **kwargs)
        coinc_event = read_table(source, tablename='coinc_event', **kwargs)
    events = join(coinc_inspiral, coinc_event, keys="coinc_event_id",
                  metadata_conflicts='silent')
    events.meta['tablename'] = 'gstlal_coinc_inspiral'
    for col_name in extra_cols:
        events.remove_column(col_name)
    return events


# combined format
def read_gstlal(source, triggers='sngl', **kwargs):
    """Read a `Table` from one or more GstLAL LIGO_LW XML files

    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document` objects

    triggers : `str`, optional
        the `Name` of the relevant `Table` to read, if not given a table will
        be returned if only one exists in the document(s).
        'sngl' for single-detector trigger information,
        'coinc' for coincident trigger information

    **kwargs
        keyword arguments for the read, or conversion functions

    See also
    --------
    gwpy.io.ligolw.read_table
        for details of keyword arguments for the read operation
    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation
    """

    if triggers == 'sngl':
        return read_gstlal_sngl(source, **kwargs)
    if triggers == 'coinc':
        return read_gstlal_coinc(source, **kwargs)
    else:
        raise ValueError("The 'triggers' argument must be 'sngl' or 'coinc'")


def identify_gstlal(origin, filepath, fileobj, *args, **kwargs):
    """Identify a GstLAL file as a ligolw file with the correct name
    """
    if is_ligolw(origin, filepath, fileobj, *args, **kwargs) and (
            filepath is not None
            and GSTLAL_FILENAME.match(basename(filepath))):
        return True
    return False


# registers for unified I/O
register_identifier(GSTLAL_FORMAT, EventTable, identify_gstlal)
register_reader(GSTLAL_SNGL_FORMAT, EventTable, read_gstlal_sngl)
register_reader(GSTLAL_COINC_FORMAT, EventTable, read_gstlal_coinc)
register_reader(GSTLAL_FORMAT, EventTable, read_gstlal)

# -- processed columns --------------------------------------------------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in a `~gwpy.table.Table` and return a `numpy.ndarray`

GET_COLUMN = {}
GET_COLUMN_EXTRA = {}


def get_snr_chi(events, snr_pow=2., chi_pow=2.):
    """Calculate the 'SNR chi' column for this GstLAL ligolw table group
    """
    snr = events['snr'][:]
    chisq = events['chisq'][:]
    snr_chi = snr**snr_pow / chisq**(chi_pow/2.)
    return snr_chi


GET_COLUMN['snr_chi'] = get_snr_chi
GET_COLUMN_EXTRA['snr_chi'] = ['snr', 'chisq']


def get_chi_snr(events, snr_pow=2., chi_pow=2.):
    """Calculate the 'chi SNR' column for this GstLAL ligolw table group,
    reciprocal of the 'SNR chi' column
    """
    return 1./get_snr_chi(events, snr_pow, chi_pow)


GET_COLUMN['chi_snr'] = get_chi_snr
GET_COLUMN_EXTRA['chi_snr'] = ['snr', 'chisq']

# use same function as pycbc
GET_COLUMN['mchirp'] = get_mchirp
GET_COLUMN_EXTRA['mchirp'] = ['mass1', 'mass2']
