# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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
from astropy.table import join

from ...io.registry import (register_reader, register_identifier)
from .. import (Table, EventTable)
from ligo.lw import lsctables
from . import read_table

__author__ = 'Derk Davis <derek.davis@ligo.org>'
__credits__ = 'Patrick Godwin <patrick.godwin@ligo.org>'

GSTLAL_FORMAT = 'ligolw.gstlal'

GSTLAL_FILENAME = re.compile('([A-Z][0-9])+-[0-9.]+-LLOID-[0-9.]+-[0-9.]+.xml')

@read_with_selection
def read_gstlal(source, triggers=None, **kwargs):
    """Read a `Table` from one or more LIGO_LW XML documents

    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document` objects

    triggers : `str`, optional
        the `Name` of the relevant `Table` to read, if not given a table will
        be returned if only one exists in the document(s).
        'single' for sngl_inpsiral triggers, 
        'coinc' for coinc triggers

    **kwargs
        keyword arguments for the read, or conversion functions

    See also
    --------
    gwpy.io.ligolw.read_table
        for details of keyword arguments for the read operation
    gwpy.table.io.ligolw.to_astropy_table
        for details of keyword arguments for the conversion operation
    """

    extra_cols = []
    if triggers == 'single':
        derived_cols = []
        if kwargs['columns']:
            for name in kwargs['columns']:
                if name not in lsctables.TableByName['sngl_inspiral'].validcolum:
                    if name in GET_COLUMN:
                        derived_cols.append(name)
                        kwargs['columns'].pop(name)
                        required_cols = GET_COLUMN_EXTRA[name]
                        for r_col in required_col if r_col not in kwargs['columns']:
                            kwargs['columns'].append(r_col)
                            extra_cols.append(r_col)
                    else:
                        raise
        events = read_table(source, tablename='sngl_inspiral', **kwargs)
        for col_name in derived_cols:
            col_data = GET_COLUMN[col_name](events)
            events.add_column(col_data,name=col_name)
    if triggers == 'coinc':
        if kwargs['columns']:
            columns = kwargs['columns']
            kwargs.pop('columns')
            # Divvy up columnss
            if 'coinc_event_id' not in columns:
                columns.append('coinc_event_id')
                extra_cols.append('coinc_event_id')
            inspiral_cols = [col if col in lsctables.TableByName['coinc_inspiral'].validcolumn
                             for col in columns]]
            event_cols = [col if col in lsctables.TableByName['coinc_event'].validcolumn
                             for col in columns]]
            if 'end' in columns:
                inspiral_cols.append('end')
            coinc_inspiral = read_table(source, tablename='coinc_inspiral', columns=inspiral_cols, **kwargs)
            coinc_event = read_table(source, tablename='coinc_event', columns=event_cols, **kwargs)
        else:
            coinc_inspiral = read_table(source, tablename='coinc_inspiral', **kwargs)
            coinc_event = read_table(source, tablename='coinc_event', **kwargs)
        events = join(coinc_inspiral, coinc_event, keys="coinc_event_id",
                      metadata_conflicts='silent')
        events.meta['tablename'] = 'gstlal_coinc_inspiral'
    else:
        raise
    for col_name in extra_cols::
        events = events.remove_column(col_name)
    return events


### !!! unclear how to set up register
def identify_gstlal(origin, filepath, fileobj, *args, **kwargs):
    """Identify a PyCBC Live file as an HDF5 with the correct name
    """
    if identify_hdf5(origin, filepath, fileobj, *args, **kwargs) and (
            filepath is not None and PYCBC_FILENAME.match(basename(filepath))):
        return True
    return False


# register for unified I/O
register_identifier(GSTLAL_FORMAT, EventTable, identify_gstlal)
register_reader(GSTLAL_FORMAT, EventTable, table_from_file)

# -- processed columns --------------------------------------------------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in a `~gwpy.table.Table` and return a `numpy.ndarray`

GET_COLUMN = {}
GET_COLUMN_EXTRA = {}

def get_eta(events):  
    """Calculate the 'eta' column for this GstLAL ligolw table group
    """
    eta = events['chisq'][:]
    return eta

GET_COLUMN['eta'] = get_eta
GET_COLUMN_EXTRA['eta'] = ['chisq']

def get_eta_snr(events, snr_pow=2., eta_pow=2.): 
    """Calculate the 'new SNR' column for this GstLAL ligolw table group
    """
    snr = events['snr'][:]
    eta = events['chisq'][:]
    eta_snr = snr**snr_pow / eta**eta_pow 
    return eta_snr

GET_COLUMN['eta_snr'] = get_eta_snr
GET_COLUMN_EXTRA['eta_snr'] = ['snr','chisq']

def get_mchirp(events):
    """Calculate the chipr mass column for this GstLAL ligolw table group
    """
    mass1 = events['mass1'][:]
    mass2 = events['mass2'][:]
    return (mass1 * mass2) ** (3/5.) / (mass1 + mass2) ** (1/5.)

GET_COLUMN['mchirp'] = get_mchirp
GET_COLUMN_EXTRA['mchirp'] = ['mass1','mass2']
