# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

import sys
import multiprocessing
from xml.sax import SAXException

from six import string_types

from astropy.io.registry import (_get_valid_format as get_format,
                                 read as io_read)
from astropy.utils.data import get_readable_fileobj

from .cache import (FILE_LIKE, file_list)


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
    # parse input as a list of files
    try:  # try and map to a list of file-like objects
        files = file_list(source)
    except ValueError:  # otherwise treat as single file
        files = [source]

    # determine input format (so we don't have to do it multiple times)
    # -- this is basically harvested from astropy.io.registry.read()
    if kwargs.get('format', None) is None:
        ctx = None
        if isinstance(source, FILE_LIKE):
            fileobj = source
        elif isinstance(source, string_types):
            try:
                ctx = get_readable_fileobj(files[0], encoding='binary')
                fileobj = ctx.__enter__()
            except IOError:
                raise
            except Exception:
                fileobj = None
        kwargs['format'] = get_format(
            'read', cls, files[0], fileobj, args, kwargs)
        if ctx is not None:
            ctx.__exit__(*sys.exc_info())

    # calculate maximum number of processes
    nproc = kwargs.pop('nproc', 1)
    num = len(files)
    nproc = min(nproc, num)

    # define multiprocessing method
    def _read_single_file(f):
        try:
            return f, io_read(cls, f, *args, **kwargs)
        except Exception as e:
            if nproc == 1:
                raise
            elif isinstance(e, SAXException):  # SAXExceptions don't pickle
                return f, e.getException()
            else:
                return f, e

    # read single file
    if nproc == 1:
        output = map(_read_single_file, files)

    # read multiple files
    else:
        # create input and output queues
        q_in = multiprocessing.Queue(1)
        q_out = multiprocessing.Queue()

        # create child processes and start
        proc = [multiprocessing.Process(target=_read_from_queue,
                                        args=(_read_single_file, q_in, q_out))
                for _ in range(nproc)]
        for p in proc:
            p.daemon = True
            p.start()

        # populate queue
        sent = [q_in.put((i, f)) for i, f in enumerate(files)]
        [q_in.put((None, None)) for _ in range(nproc)]  # queue is full

        # get results
        res = [q_out.get() for _ in range(len(sent))]

        # close processes and sort output
        [p.join() for p in proc]
        output = [table for i, table in sorted(res)]

        # raise exceptions
        for f, x in output:
            if isinstance(x, Exception):
                x.args = ('Failed to read %s: %s' % (f, str(x)),)
                raise x

    # return combined object
    _, out = zip(*output)
    return flatten(out)


def _read_from_queue(f, q_in, q_out):
    while True:
        # pick item out of input wqueue
        i, x = q_in.get()
        if i is None:
            break
        # execute method and put the result in the output queue
        q_out.put((i, f(x)))
