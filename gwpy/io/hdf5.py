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

"""Write GWpy objects to HDF5 files
"""

import os.path
from functools import wraps

from astropy.io.misc.hdf5 import is_hdf5 as identify_hdf5

from .cache import FILE_LIKE

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def open_hdf5(filename, mode='r'):
    """Wrapper to open a :class:`h5py.File` from disk, gracefully
    handling a few corner cases
    """
    import h5py
    if isinstance(filename, (h5py.Group, h5py.Dataset)):
        return filename
    elif isinstance(filename, FILE_LIKE):
        return h5py.File(filename.name, mode)
    else:
        return h5py.File(filename, mode)


def with_read_hdf5(func):
    """Decorate an HDF5-reading function to open a filepath if needed

    ``func`` should be written to presume an `h5py.Group` as the first
    positional argument.
    """
    @wraps(func)
    def decorated_func(f, *args, **kwargs):
        import h5py
        if not isinstance(f, h5py.HLObject):
            if isinstance(f, FILE_LIKE):
                f = f.name
            with h5py.File(f, 'r') as h5f:
                return func(h5f, *args, **kwargs)
        return func(f, *args, **kwargs)

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
    import h5py
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
    def decorated_func(obj, f, *args, **kwargs):
        import h5py
        if not isinstance(f, h5py.HLObject):
            append = kwargs.get('append', False)
            overwrite = kwargs.get('overwrite', False)
            if os.path.exists(f) and not (overwrite or append):
                raise IOError("File exists: %s" % f)
            with h5py.File(f, 'a' if append else 'w') as h5f:
                return func(obj, h5f, *args, **kwargs)
        return func(obj, f, *args, **kwargs)

    return decorated_func


@with_write_hdf5
def write_object_dataset(obj, f, create_func, append=False, overwrite=False,
                         **kwargs):
    """Write the given dataset to the file

    Parameters
    ----------
    f : `str`, `h5py.File`, `h5py.Group`
        the output filepath, or the HDF5 object in which to write

    obj : `object`
        the object to write into the dataset

    create_func : `function`
        a callable that can write the ``obj`` into an `h5py.Dataset`,
        must take an ``h5py.Group`` as the first argument, and ``obj``
        as the second, other keyword arguments may follow

    append : `bool`, default: `False`
        if `True`, write new dataset to existing file, otherwise an
        exception will be raised if the output file exists (only used if
        ``f`` is `str`)

    overwrite : `bool`, default: `False`
        if `True`, overwrite an existing dataset in an existing file,
        otherwise an exception will be raised if a dataset exists with
        the given name (only used if ``f`` is `str`)

    **kwargs
        other keyword arguments to pass to ``create_func``

    Returns
    -------
    dset : `h5py.Dataset`
        the dataset as created in the file

    Raises
    ------
    ValueError
        if the output file exists and ``append=False``
    """
    return create_func(f, obj, **kwargs)
