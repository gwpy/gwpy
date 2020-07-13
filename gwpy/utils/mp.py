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

"""Utilities for multi-processing
"""

import warnings
from multiprocessing import (Queue, Process)
from operator import itemgetter

from .progress import progress_bar


def _process_in_out_queues(func, q_in, q_out):
    """Iterate through a Queue, call, ``func`, and Queue the result

    Parameters
    ----------
    func : `callable`
        any function that can take an element of the input `Queue` as
        the only argument

    q_in : `multiprocessing.queue.Queue`
        the input `Queue`

    q_out : `multiprocessing.queue.Queue`
        the output `Queue`

    Notes
    -----
    To close the input `Queue`, add ``(None, None)`` as the last item
    """
    while True:
        # pick item out of input queue
        idx, arg = q_in.get()

        # sentinel
        if idx is None:
            break

        # execute method and put the result in the output queue,
        # exceptions are returned and handled upstream
        try:
            q_out.put((idx, func(arg)))
        except Exception as exc:  # pylint: disable=broad-except
            q_out.put((idx, exc))


def multiprocess_with_queues(nproc, func, inputs, verbose=False,
                             **progress_kw):
    """Map a function over a list of inputs using multiprocess

    This essentially duplicates `multiprocess.map` but allows for
    arbitrary functions (that aren't necessarily importable)

    Parameters
    ----------
    nproc : `int`
        number of processes to use, if ``1`` is given, the current process
        is used, and no child processes are forked

    func : `callable`
        the function to call in each iteration, should take a single
        argument that is the next element from ``inputs``

    inputs : `iterable`
        iterable (e.g. `list`) of inputs, each element of which is
        passed to ``func`` in one of the child processes

    verbose : `bool`, `str`, optional
        if `True`, print progress to the console as a bar, pass a
        `str` to customise the heading for the progress bar, default: `False`,
        (default heading ``'Processing:'`` if ``verbose=True`)

    Returns
    -------
    outputs : `list`
        the `list` of results from calling ``func(x)`` for each element
        of ``inputs``
    """
    if progress_kw.pop('raise_exceptions', None) is not None:
        warnings.warn("the `raise_exceptions` keyword to "
                      "multiprocess_with_queues is deprecated, and will be "
                      "removed in a future release, all exceptions will be "
                      "raised if they occur", DeprecationWarning)

    # create progress bar for verbose output
    if bool(verbose):
        if not isinstance(verbose, bool):
            progress_kw['desc'] = str(verbose)
        if isinstance(inputs, (list, tuple)):
            progress_kw.setdefault('total', len(inputs))
        pbar = progress_bar(**progress_kw)
    else:
        pbar = None

    # -------------------------------------------

    # shortcut single process
    if nproc == 1:
        def _inner(x):
            try:
                return func(x)
            finally:
                if pbar:
                    pbar.update(1)

        return list(map(_inner, inputs))

    # -------------------------------------------

    # create input and output queues
    q_in = Queue()
    q_out = Queue()

    # create child processes and start
    proclist = [
        Process(
            target=_process_in_out_queues,
            args=(func, q_in, q_out),
        ) for _ in range(nproc)
    ]

    for proc in proclist:
        proc.daemon = True
        proc.start()

    # populate queue (no need to block in serial put())
    sent = [q_in.put(x, block=False) for x in enumerate(inputs)]
    for _ in range(nproc):  # add sentinel for each process
        q_in.put((None, None))

    # get results
    res = []
    for _ in range(len(sent)):
        x = q_out.get()
        if pbar:
            pbar.update()
        res.append(x)

    # close processes and unwrap results
    for proc in proclist:
        proc.join()

    if pbar:
        pbar.close()

    # unwrap results in order
    results = [out for _, out in sorted(res, key=itemgetter(0))]

    # raise exceptions here
    for res in results:
        if isinstance(res, Exception):
            raise res

    return results
