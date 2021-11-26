# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Multi-processing utilities for input/output

This module provides the `read_multi` method, which enables spreading
reading multiple files across multiple cores, returning a flattened result.
"""

from xml.sax import SAXException

from astropy.io.registry import (read as io_read)

from .registry import get_read_format
from .utils import file_list
from ..utils import mp as mp_utils


def read_multi(flatten, cls, source, *args, **kwargs):
    """Read sources into a `cls` with multiprocessing

    This method should be called by `cls.read` and uses the `nproc`
    keyword to enable and handle pool-based multiprocessing of
    multiple source files, using `flatten` to combine the
    chunked data into a single object of the correct type.

    Parameters
    ----------
    flatten : `callable`
        a method to take a list of ``cls`` instances, and combine them
        into a single ``cls`` instance

    cls : `type`
        the object type to read

    source : `str`, `list` of `str`, ...
        the input data source, can be of in many different forms

    *args
        positional arguments to pass to the reader

    **kwargs
        keyword arguments to pass to the reader
    """
    verbose = kwargs.pop('verbose', False)

    # parse input as a list of files
    try:  # try and map to a list of file-like objects
        files = file_list(source)
    except ValueError:  # otherwise treat as single file
        files = [source]
        path = None  # to pass to get_read_format()
    else:
        path = files[0] if files else None

    # raise IndexError early when reading from empty cache
    if not files:
        raise IndexError(f"cannot read {cls.__name__} from empty source list")

    # determine input format (so we don't have to do it multiple times)
    if kwargs.get('format', None) is None:
        kwargs['format'] = get_read_format(cls, path, (source,) + args, kwargs)

    # calculate maximum number of processes
    nproc = min(kwargs.pop('nproc', 1), len(files))

    # format verbosity
    if verbose is True:
        verbose = f"Reading ({format})"

    # bundle inputs
    inputs = [(f, cls, nproc, args, kwargs) for f in files]

    # read files
    output = mp_utils.multiprocess_with_queues(
        nproc, _read_single_file, inputs, verbose=verbose, unit='files')

    # raise exceptions (from multiprocessing, single process raises inline)
    for fobj, exc in output:
        if isinstance(exc, Exception):
            exc.args = (f"Failed to read {fobj}: {exc}",)
            raise exc

    # return combined object
    _, out = zip(*output)
    return flatten(out)


def _read_single_file(bundle):
    """Reads a single file and returns the output

    This is designed only to be passed to
    :func:`gwpy.utils.mp.multiprocess_with_queues`

    Returns
    -------
    f, output :
        the input file (object) that was (attempted to be) read, and
        the result of the read operation, which may be an exception
        object.
    """
    fobj, cls, nproc, args, kwargs = bundle
    try:
        return fobj, io_read(cls, fobj, *args, **kwargs)
    # pylint: disable=broad-except,redefine-in-handler
    except Exception as exc:
        if nproc == 1:
            raise
        if isinstance(exc, SAXException):  # SAXExceptions don't pickle
            return fobj, exc.getException()  # pylint: disable=no-member
        return fobj, exc
