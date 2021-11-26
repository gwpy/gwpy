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

"""Utilities for unified input/output
"""

import gzip
import os
import tempfile
from functools import wraps
from urllib.parse import urlparse

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# build list of file-like types
from io import IOBase
FILE_LIKE = (
    IOBase, gzip.GzipFile,
    tempfile._TemporaryFileWrapper,  # pylint: disable=protected-access
)

GZIP_SIGNATURE = b'\x1f\x8b\x08'


def identify_factory(*extensions):
    """Factory function to create I/O identifiers for a set of extensions

    The returned function is designed for use in the unified I/O registry
    via the `astropy.io.registry.register_identifier` hool.

    Parameters
    ----------
    extensions : `str`
        one or more file extension strings

    Returns
    -------
    identifier : `callable`
        an identifier function that tests whether an incoming file path
        carries any of the given file extensions (using `str.endswith`)
    """
    def identify(origin, filepath, fileobj, *args, **kwargs):
        """Identify the given extensions in a file object/path
        """
        # pylint: disable=unused-argument
        return (
            isinstance(filepath, str)
            and filepath.endswith(extensions)
        )
    return identify


def gopen(name, *args, **kwargs):
    """Open a file handling optional gzipping

    If ``name`` ends with ``'.gz'``, or if the GZIP file signature is
    found at the beginning of the file, the file will be opened with
    `gzip.open`, otherwise a regular file will be returned from `open`.

    Parameters
    ----------
    name : `str`, `pathlib.Path`
        path or name of file to open.

    *args, **kwargs
        other arguments to pass to either `open` for regular files, or
        `gzip.open` for gzipped files.

    Returns
    -------
    file : `io.TextIoBase`, `file`, `gzip.GzipFile`
        the open file object
    """
    # filename declares gzip
    if str(name).endswith('.gz'):
        return gzip.open(name, *args, **kwargs)

    # open regular file
    fobj = open(name, *args, **kwargs)
    sig = fobj.read(3)
    fobj.seek(0)
    if sig == GZIP_SIGNATURE:  # file signature declares gzip
        fobj.close()  # GzipFile won't close orig file when it closes
        return gzip.open(name, *args, **kwargs)
    return fobj


def with_open(func=None, mode="r", pos=0):
    """Decorate a function to ensure the chosen argument is an open file

    Parameters
    ----------
    func : `callable`
        the function to decorate

    mode : `str`, optional
        the mode with which to open files

    pos : `int`, optional
        which argument to look at

    Examples
    --------
    To ensure that the first argument is an open read-only file, just use
    the decorator without functional parentheses or arguments:

    >>> @with_open
    >>> def my_func(pathorfile, *args, **kwargs)
    >>>     ...

    To ensure that the second argument (position 1) is a file open for writing:

    >>> @with_open(mode="w", pos=1)
    >>> def my_func(stuff, pathorfile, *args, **kwargs)
    >>>     stuff.write_to(pathorfile, *args, **kwargs)
    """
    def _decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # if the relevant positional argument isn't an open
            # file, or something that looks like one, ...
            if not isinstance(args[pos], FILE_LIKE):
                # open the file, ...
                with open(args[pos], mode=mode) as fobj:
                    # replace the argument with the open file, ...
                    args = list(args)
                    args[pos] = fobj
                    # and re-execute the function call
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapped_func
    if func:
        return _decorator(func)
    return _decorator


# -- file list utilities ------------------------------------------------------

def file_list(flist):
    """Parse a number of possible input types into a list of filepaths.

    Parameters
    ----------
    flist : `file-like` or `list-like` iterable
        the input data container, normally just a single file path, or a list
        of paths, but can generally be any of the following

        - `str` representing a single file path (or comma-separated collection)
        - open `file` or `~gzip.GzipFile` object
        - :class:`~lal.utils.CacheEntry`
        - `str` with ``.cache`` or ``.lcf`` extension
        - simple `list` or `tuple` of `str` paths

    Returns
    -------
    files : `list`
        `list` of `str` file paths

    Raises
    ------
    ValueError
        if the input `flist` cannot be interpreted as any of the above inputs
    """
    # open a cache file and return list of paths
    if (
        isinstance(flist, str)
        and flist.endswith(('.cache', '.lcf', '.ffl'))
    ):
        from .cache import read_cache
        return read_cache(flist)

    # separate comma-separate list of names
    if isinstance(flist, str):
        return flist.split(',')

    # parse list of entries (of some format)
    if isinstance(flist, (list, tuple)):
        return list(map(file_path, flist))

    # otherwise parse a single entry
    try:
        return [file_path(flist)]
    except ValueError as exc:
        exc.args = (
            f"Could not parse input {flist!r} as "
            "one or more file-like objects",
        )
        raise


def file_path(fobj):
    """Determine the path of a file.

    This doesn't do any sanity checking to check that the file
    actually exists, or is readable.

    Parameters
    ----------
    fobj : `file`, `str`, `CacheEntry`, ...
        the file object or path to parse

    Returns
    -------
    path : `str`
        the path of the underlying file

    Raises
    ------
    ValueError
        if a file path cannnot be determined

    Examples
    --------
    >>> from gwpy.io.utils import file_path
    >>> import pathlib
    >>> file_path("test.txt")
    'test.txt'
    >>> file_path(pathlib.Path('dir') / 'test.txt')
    'dir/test.txt'
    >>> file_path(open("test.txt", "r"))
    'test.txt'
    >>> file_path("file:///home/user/test.txt")
    '/home/user/test.txt'
    """
    if isinstance(fobj, str) and fobj.startswith("file:"):
        return urlparse(fobj).path
    if isinstance(fobj, (str, os.PathLike)):
        return str(fobj)
    if (isinstance(fobj, FILE_LIKE) and hasattr(fobj, "name")):
        return fobj.name
    try:
        return fobj.path
    except AttributeError:
        raise ValueError(f"Cannot parse file name for {fobj!r}")
