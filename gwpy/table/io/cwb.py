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

"""Read events from an Omega-format ASCII file.
"""

import os.path
import re
import warnings

from glue.lal import CacheEntry

from .ascii import return_reassign_ids
from ..lsctables import SnglBurstTable
from ...io.registry import (register_reader, register_identifier)
from ...io.cache import (file_list, read_cache)
from ...io.utils import (gopen, GzipFile)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SNGL_BURST_COLUMNS = SnglBurstTable.validcolumns.keys()
CWB_ASCII_SNGL_BURST_COLUMN_MAP = {
    'central frequency': 'central_freq',
    'low frequency': 'flow',
    'high frequency': 'fhigh',
    'effective correlated amplitude rho': 'amplitude',
    'likelhood': 'confidence',
    'sSNR': 'snr',
    'time': 'time',
    'time shift': 'time_lag',
}
RE_CWB_ASCII_COLNAME = re.compile('\A#\s+(?P<index>\d+) - (?P<colname>(.*))\Z')
RE_SNGL_COLNAME = re.compile('(?P<name>\w+) for (?P<ifo>[A-Z][0-9]) detector')


def get_column_list(filename, ifo, comments='#'):
    """Read the column list from a cWB ASCII file

    Parameters
    ----------
    filename : `str`
        path of file to read
    ifo : `str`
        prefix of IFO for which to find columns
    comments : `str`, optional, default: '#'
        comment character used for this file

    Returns
    -------
    usecols : `list` of `int`
        the list of column positions to pass to :meth:`numpy.loadtxt`
    columns : `list` of `str`
        the list of column names corresponding to the `usecols` list

    Notes
    -----
    The cWB 'EVENTS.txt' files give a header of (number, name) column
    definitions, however the numbers don't actually match up to the
    actual position of the relevant column when counting from zero, so the
    numbers given in the file are ignored.
    """
    columns = []
    plus2 = 0
    foundifo = False
    with gopen(filename, 'rb') as f:
        while True:
            line = f.readline()
            # stop when the header finishes
            if not line.startswith(comments):
                break
            # work out whether there are + - columns (that aren't counted)
            elif line.startswith('%s -/+' % comments):
                plus2 = 2
                continue
            # parse line, or move on
            try:
                match = RE_CWB_ASCII_COLNAME.match(line.rstrip()).groupdict()
            except AttributeError:
                continue
            colname = match['colname']
            # try and match as a ifo-specific parameter
            try:
                sngl = RE_SNGL_COLNAME.match(colname).groupdict()
            except AttributeError:
                pass
            else:
                colname = sngl['name']
                if sngl['ifo'].lower() != ifo.lower():
                    columns.append(None)
                    continue
                else:
                    foundifo = True
            # otherwise try the custom mapping, then just the sngl_burst name
            try:
                columns.append(CWB_ASCII_SNGL_BURST_COLUMN_MAP[colname])
            except KeyError:
                if colname in SNGL_BURST_COLUMNS:
                    columns.append(colname)
                else:
                    columns.append(None)
    if not foundifo:
        warnings.warn("No %s-specific columns were found in %s"
                      % (ifo, filename), stacklevel=2)
    usecols, names = zip(*[(i+plus2, c) for (i, c) in enumerate(columns)
                           if c is not None])
    return usecols, names


def sngl_burst_table_from_cwb_ascii(f, columns=None, ifo=None, filt=None,
                                    usecols=None, nproc=1, **loadtxtkwargs):
    """Read a `SnglBurstTable` from a cWB-format ASCII file

    Parameters
    ----------
    f : `file`, `str`, `~glue.lal.CacheEntry`, `list`, `~glue.lal.Cache`
        object representing one or more files. One of

        - an open `file`
        - a `str` pointing to a file path on disk
        - a formatted `~glue.lal.CacheEntry` representing one file
        - a `list` of `str` file paths
        - a formatted `~glue.lal.Cache` representing many files

    columns : `list`, optional
        list of column name strings to read, default all.

    ifo : `str`, required
        prefix of IFO to read, required for 'cwb-ascii' format
        (but not for others)

    usecols : `list` of `int`
        the list of column indices to read (absolute, zero-indexed)

    **loadtxtkwargs
        other keyword arguments are passed to `numpy.loadtxt`

    Returns
    -------
    table : `SnglBurstTable`
        a new `SnglBurstTable` filled with data read from the file(s)

    Raises
    ------
    ValueError
        if `ifo` is not given, OR

        if `columns` is given (and not `usecols`) and that column is not
        detected in the file, OR

        if `usecols` is given (and not `columns`) and that column isn't
        parseable from the given file (no `sngl_burst` equivalent)
    """
    # allow multiprocessing
    if nproc != 1:
        return read_cache(f, SnglBurstTable, nproc, return_reassign_ids,
                          columns=columns, usecols=usecols, ifo=ifo,
                          filt=filt, format='cwb-ascii', **loadtxtkwargs)

    comments = loadtxtkwargs.pop('comments', '#')
    files = file_list(f)
    if ifo is None:
        raise ValueError("ifo keyword argument must be given to read cWB "
                         "events from ASCII")
    # parse columns from file header
    allusecols, allcolumns = get_column_list(files[0], ifo, comments=comments)
    # work out which columns the user asked for
    if columns and not usecols:
        given = list(columns)
        columns = []
        usecols = []
        for c in given:
            if c.lower().startswith('peak_time') and 'time' in columns:
                continue
            elif c.lower().startswith('peak_time'):
                c = 'time'
            try:
                index = allcolumns.index(c)
            except ValueError as e:
                e.args = ('Column %r not found in %s for %s'
                          % (c, files[0], ifo),)
                raise
            else:
                columns.append(c)
                usecols.append(allusecols[index])
    elif usecols and not columns:
        columns = []
        for coln in usecols:
            try:
                index = allusecols.index(coln)
            except ValueError as e:
                e.args = ('Column %d not parseable in %s for %s\n'
                          'Note: when reading cWB ASCII files, usecols is the '
                          'list of absolute zero-indexed column positions, not'
                          ' necessarily what is given in the cWB file header'
                          % (coln, files[0], ifo),)
                raise
            else:
                columns.append(allcolumns[index])
    # or use all columns
    elif not columns and not usecols:
        columns = allcolumns
        usecols = allusecols
    # read data
    out = SnglBurstTable.read(files, columns, usecols=usecols, format='ascii',
                              comments=comments, filt=filt, **loadtxtkwargs)
    # append search information and fix SNR
    out.appendColumn('ifo')
    out.appendColumn('search')
    for t in out:
        t.ifo = ifo
        t.search = 'cwb'
        if 'snr' in columns:
            t.snr **= 1/2.
    return out


def sngl_burst_from_cwb(f, *args, **kwargs):
    """Read a `SnglBurstTable` from a cWB file either ROOT of EVENTS.TXT ASCII
    """
    files = file_list(f)
    extensions = list(set(os.path.splitext(fp)[1] for fp in files))
    if len(extensions) == 1 and extensions[0].lower() == '.root':
        raise NotImplementedError("Reading cWB from ROOT files has not been "
                                  "implemented yet")
    elif len(extensions) <= 1:
        return sngl_burst_table_from_cwb_ascii(f, *args, **kwargs)
    raise ValueError("Cannot determine correct cWB reader for multiple file "
                     "extensions: %s" % ", ".join(extensions))


def identify_cwb_ascii(origin, path, fileobj, *args, **kwargs):
    """Automatically identify a fileobj as 'cwb-ascii' format

    Notes
    -----
    This method won't actually let you automatically identify and
    read cwb-ascii files, because the basic 'ascii' format will also
    identify as valid for 'EVENTS.txt' files.

    This function means that when someone runs
    SnglBurstTable.read('EVENTS.txt') an exception will be raised with a
    a sensible error message prompting them to manually specify
    `format='cwb-ascii'`.
    """
    # identify by name
    if path is None or (
            os.path.basename(path) in ['EVENTS.txt', 'EVENTS.txt.gz']):
        return False
    # verify contents
    if fileobj is None:
        try:
            fileobj = open(path, 'rb')
        except IOError:
            return False
        close = True
    else:
        close = False
    pos = fileobj.tell()
    try:
        fileobj.seek(0)
        if fileobj.readline().startswith('# correlation threshold'):
            return True
        else:
            return False
    finally:
        fileobj.seek(pos)
        if close:
            fileobj.close()


register_identifier('cwb-ascii', SnglBurstTable, identify_cwb_ascii)
register_reader('cwb-ascii', SnglBurstTable, sngl_burst_table_from_cwb_ascii)
register_reader('cwb', SnglBurstTable, sngl_burst_from_cwb)
