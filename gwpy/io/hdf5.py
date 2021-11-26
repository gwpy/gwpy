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

"""Write GWpy objects to HDF5 files
"""

import os.path
from functools import wraps

# pylint: disable=unused-import
from astropy.io.misc.hdf5 import is_hdf5 as identify_hdf5  # noqa: F401

import h5py

from .cache import FILE_LIKE

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def open_hdf5(filename, mode='r'):
    """Wrapper to open a :class:`h5py.File` from disk, gracefully
    handling a few corner cases
    """
    if isinstance(filename, (h5py.Group, h5py.Dataset)):
        return filename
    if isinstance(filename, FILE_LIKE):
        return h5py.File(filename.name, mode)
    return h5py.File(filename, mode)


def with_read_hdf5(func):
    """Decorate an HDF5-reading function to open a filepath if needed

    ``func`` should be written to presume an `h5py.Group` as the first
    positional argument.
    """
    @wraps(func)
    def decorated_func(fobj, *args, **kwargs):
        # pylint: disable=missing-docstring
        if not isinstance(fobj, h5py.HLObject):
            if isinstance(fobj, FILE_LIKE):
                fobj = fobj.name
            with h5py.File(fobj, 'r') as h5f:
                return func(h5f, *args, **kwargs)
        return func(fobj, *args, **kwargs)

    return decorated_func


def find_dataset(h5o, path=None):
    """Find and return the relevant dataset inside the given H5 object

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
    elif path is None and len(h5o) == 1:
        path = list(h5o.keys())[0]
    elif path is None:
        raise ValueError("Please specify the HDF5 path via the "
                         "``path=`` keyword argument")
    return h5o[path]


# -- writing utilities --------------------------------------------------------

def with_write_hdf5(func):
    """Decorate an HDF5-writing function to open a filepath if needed

    ``func`` should be written to take the object to be written as the
    first argument, and then presume an `h5py.Group` as the second.

    This method uses keywords ``append`` and ``overwrite`` as follows if
    the output file already exists:

    - ``append=False, overwrite=False``: raise `~exceptions.IOError`
    - ``append=True``: open in mode ``a``
    - ``append=False, overwrite=True``: open in mode ``w``
    """
    @wraps(func)
    def decorated_func(obj, fobj, *args, **kwargs):
        # pylint: disable=missing-docstring
        if not isinstance(fobj, h5py.HLObject):
            append = kwargs.get('append', False)
            overwrite = kwargs.get('overwrite', False)
            if os.path.exists(fobj) and not (overwrite or append):
                raise IOError(f"File exists: {fobj}")
            with h5py.File(fobj, 'a' if append else 'w') as h5f:
                return func(obj, h5f, *args, **kwargs)
        return func(obj, fobj, *args, **kwargs)

    return decorated_func


def create_dataset(parent, path, overwrite=False, **kwargs):
    """Create a new dataset inside the parent HDF5 object

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
        if str(exc) == 'Unable to create link (Name already exists)':
            exc.args = (
                f"{exc}: '{path}', pass overwrite=True to ignore "
                f"existing datasets",
            )
        raise
