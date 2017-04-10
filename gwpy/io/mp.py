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

from __future__ import (division, absolute_import)

from functools import wraps
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)
from xml.sax import SAXException

from .cache import (FILE_LIKE, file_list)

from astropy.table import vstack
from astropy.io.registry import (_get_valid_format as get_format,
                                 get_reader, read as io_read)


def read_multi(flatten, cls, source, *args, **kwargs):
    """Decorate a Class.read `classmethod` with multiprocessing

    This method adds the `nproc` keyword argument to the `reader` method
    which enables splitting an input list of files into chunks and
    reading each chunk in parallel, using `flatten` to combine the
    chunked data into a single object of the correct type
    """
    # parse input as a list of files
    if isinstance(source, list):
        files = source
    else:
        try:  # try and map to a list of file-like objects
            files = file_list(source)
        except ValueError:  # otherwise treat as single
            files = [source]

    # determine input format
    if kwargs.get('format', None) is None:
        if isinstance(source, FILE_LIKE):
            fileobj = source
        else:
            fileobj = None
        kwargs['format'] = get_format(
            'read', cls, files[0], fileobj, args, kwargs)

    # calculate maximum number of processes
    nproc = kwargs.pop('nproc', 1)
    num = len(files)
    nproc = min(nproc, num)

    # read single file or single process
    if num == 1:
        return io_read(cls, files[0], *args, **kwargs)
    if nproc == 1:
        return io_read(cls, source, *args, **kwargs)

    # define multiprocessing method
    def _read_chunk(q, chunk, index):
        if len(chunk) == 1:
            chunk = chunk[0]
        try:
            q.put((index, io_read(cls, chunk, *args, **kwargs)))
        except Exception as e:
            if isinstance(e, SAXException):
                q.put(e.getException())
            else:
                q.put(e)

    # split source into parts
    numperproc = int(ceil(num / nproc))
    chunks = [type(files)(files[i:i+numperproc]) for i in
              range(0, num, numperproc)]

    # process
    queue = ProcessQueue(nproc)
    processes = []
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        process = Process(target=_read_chunk, args=(queue, chunk, i))
        process.daemon = True
        process.start()
        processes.append(process)

    # get data and block
    output = []
    for i in range(len(processes)):
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        output.append(result)
    for process in processes:
        process.join()

    # return chunks sorted into input order
    return flatten(zip(*sorted(output, key=lambda out: out[0]))[1])
