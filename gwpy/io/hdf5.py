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

"""Write GWpy objects to HDF5 files."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
from astropy.io.misc.hdf5 import is_hdf5

from .utils import FileSystemPath

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        Concatenate,
        Literal,
        ParamSpec,
        TypeVar,
    )

    from .utils import (
        FileLike,
        Readable,
        Writable,
    )

    P = ParamSpec("P")
    R = TypeVar("R")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "create_dataset",
    "find_dataset",
    "identify_hdf5",
    "open_hdf5",
    "with_read_hdf5",
    "with_write_hdf5",
]


def identify_hdf5(
    origin: Literal["read", "write"],
    filepath: FileSystemPath | None = None,
    fileobj: FileLike | None = None,
    *args: object,
    **kwargs: object,
) -> bool:
    """Identify an HDF5 file based on its signature.

    Parameters
    ----------
    origin : `str`
        Either ``'read'`` or ``'write'``, indicating whether the
        identification is being done for reading or writing.

    filepath : `str`, `pathlib.Path`, optional
        The file path to check.

    fileobj : file-like, optional
        A file-like object to check.

    args, kwargs
        Any additional arguments are passed directly to
        `astropy.io.misc.hdf5.is_hdf5`.

    Returns
    -------
    is_hdf5 : `bool`
        `True` if the file is an HDF5 file, `False` otherwise.

    See Also
    --------
    astropy.io.misc.hdf5.is_hdf5
    """
    if filepath is not None:
        # is_hdf5 requires a string filepath
        filepath = str(filepath)
    return is_hdf5(origin, filepath, fileobj, *args, **kwargs)


def open_hdf5(
    source: FileLike | h5py.HLObject,
    mode: str = "r",
    **kwargs,
) -> h5py.HLObject:
    """Open a :class:`h5py.File` from disk, gracefully handling corner cases.

    If ``source`` is already an HDF5 object, it is simply returned.
    Otherwise, it is opened as an HDF5 file in the given mode.

    Parameters
    ----------
    source : file-like, `h5py.Group`, `h5py.Dataset`
        The file path or file-like object to open, or an existing
        HDF5 object to return as-is.

    mode : `str`
        The mode in which to open the file, default: ``'r'``.

    kwargs
        Any additional keyword arguments are passed directly to
        :class:`h5py.File`.

    Returns
    -------
    h5obj : `h5py.File`, `h5py.Group`, `h5py.Dataset`
        The opened HDF5 file object, or the original object if it was
        already an HDF5 object.
    """
    if isinstance(source, h5py.HLObject):
        return source
    return h5py.File(source, mode=mode, **kwargs)


def with_read_hdf5(
    func: Callable[Concatenate[Readable | h5py.HLObject, P], R],
) -> Callable[Concatenate[h5py.HLObject, P], R]:
    """Decorate an HDF5-reading function to open a filepath if needed.

    The decorated function will accept file paths or readable objects,
    but will always pass an h5py.Group as the first argument to the
    original function.

    Parameters
    ----------
    func : callable
        Function that expects an h5py.Group as its first argument

    Returns
    -------
    callable
        Decorated function that accepts file paths/readable objects
    """
    @wraps(func)
    def decorated_func(
        fobj: Readable,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        if isinstance(fobj, h5py.HLObject):
            return func(fobj, *args, **kwargs)
        with h5py.File(fobj, "r") as h5f:
            return func(h5f, *args, **kwargs)

    return decorated_func


def find_dataset(
    h5o: h5py.File | h5py.Group | h5py.Dataset,
    path: str | None = None,
) -> h5py.Dataset:
    """Find and return the relevant dataset inside the given H5 object.

    If ``path=None`` is given, and ``h5o`` contains a single dataset, that
    will be returned

    Parameters
    ----------
    h5o : `h5py.File`, `h5py.Group`
        the HDF5 object in which to search

    path : `str`, optional
        the path (relative to ``h5o``) of the desired data set

    Returns
    -------
    dset : `h5py.Dataset`
        the recovered dataset object

    Raises
    ------
    ValueError
        if ``path=None`` and the HDF5 object contains multiple datasets
    KeyError
        if ``path`` is given but is not found within the HDF5 object
    """
    # find dataset
    if isinstance(h5o, h5py.Dataset):
        return h5o
    if path is None and len(h5o) == 1:
        path = next(iter(h5o.keys()))
    elif path is None:
        msg = "Please specify the HDF5 path via the ``path=`` keyword argument"
        raise ValueError(msg)
    return h5o[path]


# -- writing utilities ---------------

def with_write_hdf5(func: Callable[..., object]) -> Callable[..., object]:
    """Decorate an HDF5-writing function to open a filepath if needed.

    ``func`` should be written to take the object to be written as the
    first argument, and then presume an `h5py.Group` as the second.

    This method uses keywords ``append`` and ``overwrite`` as follows if
    the output file already exists:

    - ``append=False, overwrite=False``: raise `~exceptions.IOError`
    - ``append=True``: open in mode ``a``
    - ``append=False, overwrite=True``: open in mode ``w``
    """
    @wraps(func)
    def decorated_func(
        obj: object,
        fobj: Writable | h5py.HLObject,
        *args: object,
        **kwargs: object,
    ) -> object:
        if not isinstance(fobj, h5py.HLObject):
            append = kwargs.get("append", False)
            overwrite = kwargs.get("overwrite", False)
            # Check if file exists (only for path-like objects)
            if (
                isinstance(fobj, FileSystemPath)
                and Path(fobj).exists()
                and not (overwrite or append)
            ):
                msg = f"File exists: {fobj}"
                raise OSError(msg)
            with h5py.File(fobj, "a" if append else "w") as h5f:
                return func(obj, h5f, *args, **kwargs)
        return func(obj, fobj, *args, **kwargs)

    return decorated_func


def create_dataset(
    parent: h5py.Group | h5py.File,
    path: str,
    *,
    overwrite: bool = False,
    **kwargs: object,
) -> h5py.Dataset:
    """Create a new dataset inside the parent HDF5 object.

    Parameters
    ----------
    parent : `h5py.Group`, `h5py.File`
        the object in which to create a new dataset

    path : `str`
        the path at which to create the new dataset

    overwrite : `bool`
        if `True`, delete any existing dataset at the desired path,
        default: `False`

    **kwargs
        other arguments are passed directly to
        :meth:`h5py.Group.create_dataset`

    Returns
    -------
    dataset : `h5py.Dataset`
        the newly created dataset
    """
    # force deletion of existing dataset
    if path in parent and overwrite:
        del parent[path]

    # create new dataset with improved error handling
    try:
        return parent.create_dataset(path, **kwargs)
    except RuntimeError as exc:
        if str(exc) == "Unable to create link (Name already exists)":
            msg = (
                f"{exc}: '{path}', pass overwrite=True to ignore "
                f"existing datasets"
            )
            exc.args = (msg,)
        raise
