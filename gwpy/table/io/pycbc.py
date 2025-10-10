# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Read events from the PyCBC live online GW search."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy
from astropy.table import Table

from ...io.hdf5 import (
    identify_hdf5,
    with_read_hdf5,
)
from .. import EventTable
from ..filter import (
    filter_table,
    parse_column_filters,
)
from .utils import (
    DYNAMIC_COLUMN_FUNC,
    DYNAMIC_COLUMN_INPUT,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        Any,
        Literal,
    )

    from ...frequencyseries import FrequencySeries
    from ...io.utils import (
        FileLike,
        FileSystemPath,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Alex Nitz <alex.nitz@ligo.org>"

PYCBC_LIVE_FORMAT = "hdf5.pycbc_live"

META_COLUMNS: set[str] = {
    "loudest",
    "psd",
}
PYCBC_FILENAME = re.compile(
    "([A-Z][0-9])+-Live-[0-9.]+-[0-9.]+.(h5|hdf|hdf5)",
)


def _get_column(
    source: h5py.Group,
    name: str,
    loudest: numpy.ndarray | None = None,
) -> Table.Column:
    """Read a `~astropy.table.Column` from a PyCBC `h5py.Group`."""
    try:  # normal column
        arr = source[name][:]
    except KeyError:
        # try and generate the column on-the-fly
        if name in GET_COLUMN:
            arr = GET_COLUMN[name](source)
        else:
            raise
    if loudest is not None:  # return only the loudest events
        arr = arr[loudest]
    return Table.Column(arr, name=name)


@with_read_hdf5
def table_from_file(
    source: h5py.Group,
    ifo: str | None = None,
    columns: list[str] | None = None,
    *,
    where: str | list[str] | None = None,
    loudest: bool = False,
    extended_metadata: bool = True,
) -> Table:
    """Read a `Table` from a PyCBC live HDF5 file.

    The PyCBC project stores tabular data in HDF5 files using an `h5py.Group`
    to describe the table, and `h5py.Dataset` structures for each column.

    Parameters
    ----------
    source : `str`, `pathlib.Path`, `file`, `h5py.Group`
        The file path or open `h5py` object from which to read the data.

    ifo : `str`, optional
        The interferometer prefix (e.g. ``'G1'``) to read; this is required
        if reading from a file path or a `h5py.File` and the containing file
        stores data for multiple interferometers.

    columns : `list` of `str`, optional
        The list of column names to include in returned table.
        Default is all columns.

    where : `str`, `list` of `str`, optional
        One or more filter condition strings to apply, e.g. ``'snr>6'``.

    loudest : `bool`, optional
        If `True` read only those events marked as 'loudest'.
        Default is `False` (read all rows in the table).

    extended_metadata : `bool`, optional
        If `True` record non-column datasets found in the H5 group (e.g. ``'psd'``)
        in the ``meta`` dict. Default is `True`.

    Returns
    -------
    table : `~astropy.table.Table`
    """
    # find group
    if isinstance(source, h5py.File):
        source, ifo = _find_table_group(source, ifo=ifo)

    # -- by this point 'source' is guaranteed to be an h5py.Group

    # list of columns to read (default: everything that's there)
    read_cols = set(columns or _get_columns(source))

    # parse where conditions
    filters = parse_column_filters(where or [])
    for fs in filters:
        cols = fs.column
        if isinstance(cols, str):
            read_cols.add(cols)
        else:
            read_cols.update(cols)

    # set up meta dict
    meta = {"ifo": ifo}
    meta.update(source.attrs)
    if extended_metadata:
        meta.update(_get_extended_metadata(source))

    # extract loudest array index
    loudidx: numpy.ndarray | None = None
    if loudest:
        loudidx = source["loudest"][:]

    # map data to columns
    data = [
        _get_column(source, name, loudest=loudidx)
        for name in read_cols
    ]

    # read, applying filters, and column filters
    tab = Table(data, meta=meta)
    if filters:
        tab = filter_table(tab, filters)
    if columns:  # return only the columns the user asked for
        return tab[columns]
    return tab


def _find_table_group(
    h5file: h5py.Group,
    ifo: str | None = None,
) -> tuple[h5py.Group, str]:
    """Find the right `h5py.Group` within the given `h5py.File`."""
    exclude = ("background",)
    if ifo is None:
        try:
            ifo, = (key for key in h5file if key not in exclude)
        except ValueError as exc:
            exc.args = (
                "PyCBC live HDF5 file contains dataset groups "
                "for multiple interferometers, please specify "
                "the prefix of the relevant interferometer via "
                "the `ifo` keyword argument, e.g: `ifo=G1`",
            )
            raise
    try:
        return h5file[ifo], ifo
    except KeyError as exc:
        exc.args = (f"No group for ifo '{ifo}' in PyCBC live HDF5 file",)
        raise


def _get_columns(h5group: h5py.Group) -> set[str]:
    """Find valid column names from a PyCBC HDF5 Group.

    Returns a `set` of names.
    """
    columns = set()
    for name in sorted(h5group):
        # not a column, or one to ignore
        if (
            not isinstance(h5group[name], h5py.Dataset)
            or name == "template_boundaries"
        ):
            continue

        # template parameters, ignore those
        if (
            name.endswith("_template")
            and name[:-9] in columns
        ):
            continue

        columns.add(name)
    return columns - META_COLUMNS


def _parse_psd(dataset: h5py.Dataset) -> FrequencySeries:
    """Parse a PyCBC PSD from an `h5py.Dataset`.

    Returns
    -------
    gwpy.frequencyseries.FrequencySeries
    """
    from gwpy.frequencyseries import FrequencySeries
    return FrequencySeries(
        dataset[:],
        f0=0,
        df=dataset.attrs["delta_f"],
        name="pycbc_live",
    )


def _get_extended_metadata(h5group: h5py.Group) -> dict[str, Any]:
    """Extract the extended metadata for a PyCBC table in HDF5.

    This method packs non-table-column datasets in the given h5group into
    a metadata `dict`.

    Returns
    -------
    meta : `dict`
       The metadata dict.
    """
    meta = {}

    # get PSD
    try:
        psdds = h5group["psd"]
    except KeyError:
        pass
    else:
        meta["psd"] = _parse_psd(psdds)

    # get everything else
    for key in META_COLUMNS - {"psd"}:
        try:
            value = h5group[key][:]
        except KeyError:
            pass
        else:
            meta[key] = value

    return meta


def filter_empty_files(
    files: list[str],
    ifo: str | None = None,
) -> list[str]:
    """Remove empty PyCBC-HDF5 files from a list.

    Parameters
    ----------
    files : `list`
        A list of file paths to test.

    ifo : `str`, optional
        Prefix for the interferometer of interest (e.g. ``'L1'``),
        include this for a more robust test of 'emptiness'.

    Returns
    -------
    nonempty : `list`
        The subset of the input ``files`` that are considered not empty.

    See Also
    --------
    empty_hdf5_file
        For details of the 'emptiness' test.
    """
    return type(files)([f for f in files if not empty_hdf5_file(f, ifo=ifo)])


@with_read_hdf5
def empty_hdf5_file(
    h5f: h5py.File,
    ifo: str | None = None,
) -> bool:
    """Determine whether a PyCBC-HDF5 file is empty.

    A file is considered empty if it contains no groups at the base level,
    or if the ``ifo`` group contains only the ``psd`` dataset.

    Parameters
    ----------
    h5f : `str`, `pathlib.Path`, `h5py.File`
        The path of the pycbc_live file to test.

    ifo : `str`, optional
        Prefix for the interferometer of interest (e.g. ``'L1'``),
        include this for a more robust test of 'emptiness'.

    Returns
    -------
    empty : `bool`
        `True` if the file looks to have no content, otherwise `False`.
    """
    # the decorator opens the HDF5 file for us, so h5f is guaranteed to
    # be an h5py.Group object
    h5f = h5f.file
    # if the root group is empty, the file is empty
    if list(h5f) == []:
        return True

    # for each group (or the IFO group given by the user),
    # check whether there is any useful content
    groups = h5f.keys() if ifo is None else [ifo]
    return all(set(h5f.get(group, [])) <= {"gates", "psd"} for group in groups)


def identify_pycbc_live(
    origin: Literal["read", "write"],
    filepath: FileSystemPath | None,
    fileobj: FileLike | None,
    *args,  # noqa: ANN002
    **kwargs,
) -> bool:
    """Identify a PyCBC Live file as an HDF5 with the correct name."""
    # first, check that this is a valid HDF5 file
    if not identify_hdf5(origin, filepath, fileobj, *args, **kwargs):
        return False

    # next, check that the filename matches what we expect from PyCBC live
    # -- this is terrible, but it's the best I can think of
    if filepath is None and fileobj is not None:  # need file name
        filepath = getattr(fileobj, "name", None)
    return bool(
        filepath is not None
        and PYCBC_FILENAME.match(Path(filepath).name),
    )


# register for unified I/O (with higher priority than HDF5 reader)
EventTable.read.registry.register_identifier(
    PYCBC_LIVE_FORMAT,
    EventTable,
    identify_pycbc_live,
)
EventTable.read.registry.register_reader(
    PYCBC_LIVE_FORMAT,
    EventTable,
    table_from_file,
    priority=1,
)

# -- processed columns ---------------
#
# Here we define methods required to build commonly desired columns that
# are just a combination of the basic columns.
#
# Each method should take in an `~h5py.Group` and return a `numpy.ndarray`

GET_COLUMN: dict[str, Callable] = {}
GET_COLUMN_EXTRA: dict[str, set[str]] = {}


def _new_snr_scale(
    redx2: float | numpy.ndarray,
    q: float = 6.,
    n: float = 2.,
) -> float | numpy.ndarray:
    return (.5 * (1. + redx2 ** (q/n))) ** (-1./q)


def get_new_snr(
    h5group: h5py.Group,
    q: float = 6.,
    n: float = 2.,
) -> numpy.ndarray:
    """Calculate the 'new SNR' column for this PyCBC HDF5 table group."""
    newsnr = h5group["snr"][:].copy()
    rchisq = h5group["chisq"][:]
    idx = numpy.where(rchisq > 1.)[0]
    newsnr[idx] *= _new_snr_scale(rchisq[idx], q=q, n=n)
    return newsnr


GET_COLUMN["new_snr"] = get_new_snr
GET_COLUMN_EXTRA["new_snr"] = {"snr", "chisq"}

# use the generic mass functions
for _key in (
    "mchirp",
    "mtotal",
):
    GET_COLUMN[_key] = DYNAMIC_COLUMN_FUNC[_key]
    GET_COLUMN_EXTRA[_key] = DYNAMIC_COLUMN_INPUT[_key]

# update custom columns using pycbc's ranking function dict
try:
    from pycbc.events.ranking import (
        required_datasets,
        sngls_ranking_function_dict,
    )
except ImportError:
    pass
else:
    GET_COLUMN.update(sngls_ranking_function_dict)
    GET_COLUMN_EXTRA.update(required_datasets)
