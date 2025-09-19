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

"""Basic HDF5 I/O methods for Array and sub-classes."""

from __future__ import annotations

import contextlib
import pickle
from decimal import Decimal
from functools import (
    partial,
    wraps,
)
from operator import attrgetter
from typing import TYPE_CHECKING

from astropy.units import (
    Quantity,
    UnitBase,
)

from ...detector import Channel
from ...io import hdf5 as io_hdf5
from ...time import LIGOTimeGPS
from .. import (
    Array,
    Index,
    Series,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import (
        IO,
        Any,
    )

    import h5py

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

ATTR_TYPE_MAP: dict[type, Callable] = {
    Quantity: attrgetter("value"),
    Channel: str,
    UnitBase: str,
    Decimal: float,
    LIGOTimeGPS: float,
}


# -- read ----------------------------


@io_hdf5.with_read_hdf5
def read_hdf5_array(
    source: h5py.Group,
    path: str | None = None,
    array_type: type[Array] = Array,
) -> Array:
    """Read an `Array` from the given HDF5 object.

    Parameters
    ----------
    source : `str`, `Path`, `file`, `h5py.File`, `h5py.Group`
        Path to HDF5 file on disk, or open `h5py.Group`.

    path : `str`, optional
        Path of dataset in HDF5 file.
        This option is required if the ``source`` contains more than a single
        dataset.

    array_type : `type`
        Desired return type.
    """
    dataset = io_hdf5.find_dataset(source, path=path)
    attrs = dict(dataset.attrs)

    # unpickle channel object
    with contextlib.suppress(KeyError):
        attrs["channel"] = _unpickle_channel(attrs["channel"])

    # unpack byte strings for python3
    for key, attr in attrs.items():
        if isinstance(attr, bytes):
            attrs[key] = attr.decode("utf-8")
    return array_type(dataset[()], **attrs)


def _unpickle_channel(raw: str | bytes, **kwargs) -> str:
    """Try and unpickle a channel with sensible error handling."""
    if isinstance(raw, str):
        return raw

    try:
        return pickle.loads(raw, **kwargs)
    except (
        ValueError,
        pickle.UnpicklingError,
        EOFError,
        IndexError,
    ):
        raw = raw.decode("utf-8")
        return _unpickle_channel(raw)


# -- write ---------------------------

class IgnoredAttributeError(ValueError):
    """Internal exception to indicate an attribute to be ignored."""


def _format_metadata_attribute(value: Any) -> Any:
    """Format a value for writing to HDF5 as a `h5py.Dataset` attribute."""
    if value is None or (isinstance(value, Index) and value.regular):
        raise IgnoredAttributeError

    # map type to something HDF5 can handle
    for typekey, func in ATTR_TYPE_MAP.items():
        if issubclass(type(value), typekey):
            return func(value)
    return value


def write_array_metadata(
    dataset: h5py.Dataset,
    array: Array,
) -> None:
    """Write metadata for ``array`` into the `h5py.Dataset`."""
    for attr in ("unit", *array._metadata_slots):
        # format attribute
        try:
            value = _format_metadata_attribute(
                getattr(array, f"_{attr}", None),
            )
        except IgnoredAttributeError:
            continue

        # store attribute
        try:
            dataset.attrs[attr] = value
        except (
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            exc.args = (
                f"Failed to store {attr} ({type(value).__name__}) "
                f"for {type(array).__name__}: '{exc}'",
            )
            raise


@io_hdf5.with_write_hdf5
def write_hdf5_array(
    array: Array,
    h5g: h5py.Group,
    path: str | None = None,
    *,
    attrs: dict[str, Any] | None = None,
    append: bool = False,  # used by `with_write_hdf5`  # noqa: ARG001
    overwrite: bool = False,
    compression: str = "gzip",
    **kwargs,
) -> h5py.Dataset:
    """Write ``array`` to a `h5py.Dataset`.

    Parameters
    ----------
    array : `gwpy.types.Array`
        The data object to write.

    h5g : `str`, `Path`, `file`, `h5py.Group`
        A file path to write to, or an `h5py.Group` in which to create
        a new dataset.

    path : `str`, optional
        The path inside the group at which to create the new dataset,
        defaults to ``array.name``.

    attrs : `dict`, optional
        Extra metadata to write into `h5py.Dataset.attrs`, on top of
        the default metadata.

    append : `bool`, default: `False`
        Tf `True`, write new dataset to existing file, otherwise an
        exception will be raised if the output file exists.
        Default is `False`.

    overwrite : `bool`, default: `False`
        If `True`, overwrite an existing dataset in an existing file,
        otherwise an exception will be raised if a dataset exists with
        the given name.
        Default is `False`.

    compression : `str`, `int`, optional
        Compression option to pass to :meth:`h5py.Group.create_dataset`.

    kwargs
        Other keyword arguments for :meth:`h5py.Group.create_dataset`.

    Returns
    -------
    datasets : `h5py.Dataset`
        The newly created dataset.

    See Also
    --------
    h5py.Group.create_dataset
        For documentation of other valid keyword arguments.
    """
    if path is None:
        path = array.name
    if path is None:
        msg = (
            f"Cannot determine HDF5 path for {type(array).__name__}, "
            "please set `name` attribute, or pass `path` keyword when writing"
        )
        raise ValueError(
            msg,
        )

    # create dataset
    dset = io_hdf5.create_dataset(
        h5g,
        path,
        overwrite=overwrite,
        data=array.value,
        compression=compression,
        **kwargs,
    )

    # write default metadata
    write_array_metadata(dset, array)

    # allow caller to specify their own metadata dict
    if attrs:
        for key in attrs:
            dset.attrs[key] = attrs[key]

    return dset


def format_index_array_attrs(series: Series) -> dict[str, Any]:
    """Format metadata attributes for and indexed array.

    This function is used to provide the necessary metadata to meet
    the (proposed) LIGO Common Data Format specification for series data
    in HDF5.
    """
    attrs = {}
    # loop through named axes
    for _, axis in zip(range(series.ndim), ("x", "y", "z"), strict=False):
        # find property names
        unit = f"{axis}unit"
        origin = f"{axis}0"
        delta = f"d{axis}"

        # store attributes
        aunit = getattr(series, unit)
        attrs.update(
            {
                unit: str(aunit),
                origin: getattr(series, origin).to(aunit).value,
                delta: getattr(series, delta).to(aunit).value,
            },
        )
    return attrs


@wraps(write_hdf5_array)
def write_hdf5_series(
    series: Series,
    output: str | Path | IO | h5py.Group,
    path: str | None = None,
    attrs: dict[str, Any] | None = None,
    **kwargs,
) -> h5py.Dataset:
    """Write a Series to HDF5.

    See `write_hdf5_array` for details of arguments and keywords.
    """
    if attrs is None:
        attrs = format_index_array_attrs(series)
    return write_hdf5_array(
        series,
        output,
        path=path,
        attrs=attrs,
        **kwargs,
    )


# -- register ------------------------

def register_hdf5_array_io(
    array_type: type[Array],
    format: str = "hdf5",  # noqa: A002
    *,
    read: bool = True,
    write: bool = True,
    identify: bool = True,
) -> None:
    """Registry read() and write() methods for the HDF5 format."""
    if read:
        array_type.read.registry.register_reader(
            format,
            array_type,
            partial(read_hdf5_array, array_type=array_type),
        )

    if write:
        if issubclass(array_type, Series):
            writer = write_hdf5_series
        else:
            writer = write_hdf5_array
        array_type.write.registry.register_writer(
            format,
            array_type,
            writer,
        )

    if identify:
        array_type.read.registry.register_identifier(
            format,
            array_type,
            io_hdf5.identify_hdf5,
        )
