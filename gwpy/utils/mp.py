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

"""Utilities for multi-processing
"""

from multiprocessing import (Queue, Process)
from operator import itemgetter


def process_in_out_queues(func, q_in, q_out):
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
    To close the input `Queue`, add ``(None, None)` as the last item
    """
    while True:
        # pick item out of input wqueue
        i, x = q_in.get()
        if i is None:
            break
        # execute method and put the result in the output queue
        q_out.put((i, func(x)))


def multiprocess_with_queues(nproc, func, inputs, raise_exceptions=False):
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

    raise_exceptions : `bool`, optional, default: `False`
        if `True`, if the output of any ``func(input)`` calls is
        an `Exception`, it will be raised directly, otherwise the outputs
        are simply returned. This is useful if ``func`` can be built to
        detect exceptions and return then, rather than raising, so that
        child processes don't hang in multiprocessing when errors occur

    Returns
    -------
    outputs : `list`
        the `list` of results from calling ``func(x)`` for each element
        of ``inputs``
    """
    # shortcut single process
    if nproc == 1:
        return list(map(func, inputs))

    # create input and output queues
    q_in = Queue(1)
    q_out = Queue()

    # create child processes and start
    proc = [Process(target=process_in_out_queues, args=(func, q_in, q_out))
            for _ in range(nproc)]
    for p in proc:
        p.daemon = True
        p.start()

    # populate queue
    sent = list(map(q_in.put, enumerate(inputs)))
    [q_in.put((None, None)) for _ in range(nproc)]  # queue is full

    # get results
    res = [q_out.get() for _ in range(len(sent))]

    # close processes and unwrap results
    [p.join() for p in proc]
    results = [out for i, out in sorted(res, key=itemgetter(0))]

    # raise exceptions
    if raise_exceptions:
        for e in results:
            if isinstance(e, Exception):
                raise e

    return results
