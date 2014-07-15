# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Input/Output utilities for LAL Cache files.
"""

from __future__ import division
from math import ceil
from multiprocessing import (cpu_count, Process, Queue as ProcessQueue)
from six import string_types
import warnings

from glue.lal import (Cache, CacheEntry)

from astropy.io.registry import _get_valid_format

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def open_cache(lcf):
    """Read a LAL-format cache file into memory as a
    :class:`glue.lal.Cache`.
    """
    if isinstance(lcf, file):
        return Cache.fromfile(lcf)
    else:
        with open(lcf, 'r') as f:
            return Cache.fromfile(f)


def identify_cache_file(*args, **kwargs):
    """Determine an input object as either a LAL-format cache file.
    """
    cachefile = args[3]
    if isinstance(cachefile, file):
        cachefile = cachefile.name
    # identify string
    if (isinstance(cachefile, string_types) and
            cachefile.endswith(('.lcf', '.cache'))):
        return True
    # identify cache object
    else:
        return False


def identify_cache(*args, **kwargs):
    """Determine an input object as a :class:`glue.lal.Cache` or a
    :lalsuite:`LALCache`.
    """
    cacheobj = args[3]
    if isinstance(cacheobj, Cache):
        return True
    else:
        return False


def file_list(flist):
    """Parse a number of possible input types into a list of filepaths.
    """
    # format list of files
    if isinstance(flist, CacheEntry):
        return [flist.path]
    elif (isinstance(flist, string_types) and
              flist.endswith(('.cache', '.lcf'))):
        return open_cache(flist).pfnlist()
    elif isinstance(flist, string_types):
        return flist.split(',')
    elif isinstance(flist, Cache):
        return flist.pfnlist()
    elif isinstance(flist, (list, tuple)):
        return flist
    else:
        return list([flist])


# ----------------------------------------------------------------------------
# generic multiprocessing from caches

def read_cache(cache, target, nproc, post, *args, **kwargs):
    """Read arbitrary data from a cache file

    Parameters
    ----------
    cache : :class:`glue.lal.Cache`, `str`
        cache of files files, or path to a LAL-format cache file
        on disk.
    target : `type`
        target class to read into.
    nproc : `int`
        number of individual processes to use.
    post : `function`
        function to post-process output object before returning.
        The output of this method will be returns, so in-place operations
        must return the object.
    *args
        other positional arguments to pass to the target.read()
        classmethod.
    **kwargs
        keyword arguments to pass to the target.read() classmethod.

    Returns
    -------
    data : target
        an instance of the target class, seeded with data read from
        the cache.

    Notes
    -----
    The returned object is constructed from the output of each
    sub-process via the '+=' in-place addition operator.

    If the input cache is indeed a :class:`~glue.lal.Cache` object,
    the sub-processes will be combined in time order, otherwise the ordering
    is given by the order of entries in the input cache (for example,
    if it is a simple `list` of files).

    .. warning::

       no protection is given against overloading the host, for example,
       no checks are done to ensure that ``nproc`` is less than the number
       of available cores.

       High values of ``nproc`` should be used at the users discretion,
       the GWpy team accepts to liability for loss as a result of abuse
       of this feature.
    """
    # read the cache
    if isinstance(cache, (file, unicode, str)):
        cache = open_cache(cache)
    if isinstance(cache, Cache):
        cache.sort(key=lambda ce: ce.segment[0])

    # force one file per process minimum
    nproc = min(nproc, len(cache))
    if nproc > cpu_count():
        warnings.warn("Using %d processes on a %d-core machine is "
                      "unrecommended...but not forbidden."
                      % (nproc, cpu_count()))

    # work out underlying data type
    try:
        kwargs.setdefault(
            'format', _get_valid_format('read', target, None,
                                        None, (cache[0],), {}))
    # if empty, put anything, since it doesn't matter
    except IndexError:
        kwargs.setdefault('format', 'ligolw')
    except Exception:
        if 'format' not in kwargs:
            raise

    if nproc <= 1:
        return target.read(cache, *args, **kwargs)

    # define how to read each sub-cache
    def _read(q, sc, i):
        try:
            q.put((i, target.read(sc, *args, **kwargs)))
        except Exception as e:
            q.put(e)

    # separate cache into parts
    fperproc = int(ceil(len(cache) / nproc))
    subcaches = [cache.__class__(cache[i:i+fperproc]) for
                 i in range(0, len(cache), fperproc)]

    # start all processes
    queue = ProcessQueue(nproc)
    proclist = []
    for i, subcache in enumerate(subcaches):
        process = Process(target=_read, args=(queue, subcache, i))
        process.daemon = True
        proclist.append(process)
        process.start()

    # get data and block
    pout = []
    for i in range(len(proclist)):
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        pout.append(result)
    for process in proclist:
        process.join()

    # combine and return
    data = zip(*sorted(pout, key=lambda out: out[0]))[1]
    try:
        out = data[0].copy()
    except AttributeError:
        out = data[0]
    for datum in data[1:]:
        out += datum

    if post:
        return post(out)
    else:
        return out


def read_cache_factory(target):
    """Generate a read_cache method specific to a given target class.

    Parameters
    ----------
    target : `type`
        `class` object to bind to returned method

    Returns
    -------
    read_target_cache : `function`
        newly-minted method taking arbitrary arguments and keyword
        arguments, and handling type associations and multiprocessing.
    """
    def _read(f, *args, **kwargs):
        nproc = kwargs.pop('nproc', 1)
        post = kwargs.pop('post', None)
        return read_cache(f, target, nproc, post, *args, **kwargs)
    return _read
