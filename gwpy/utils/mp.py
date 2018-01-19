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

import sys
from multiprocessing import (Queue, Process, Value)
from operator import itemgetter
from functools import partial

from six.moves import StringIO

from astropy.utils.console import (ProgressBar, ProgressBarOrSpinner,
                                   color_print)

from .misc import null_context


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
        idx, arg = q_in.get()
        if idx is None:  # sentinel
            break
        # execute method and put the result in the output queue
        q_out.put((idx, func(arg)))


def multiprocess_with_queues(nproc, func, inputs, raise_exceptions=False,
                             verbose=False, **progress_kw):
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
    # handle verbose printing with a progress bar
    if verbose:
        total = len(inputs) if isinstance(inputs, (list, tuple)) else None
        if verbose is True:
            verbose = 'Processing:'
        bar = _MultiProgressBarOrSpinner(total, verbose, file=sys.stdout)

        def _inner_func(in_):  # update progress bar on each iteration
            res = func(in_)
            bar.update(None)
            return res

    else:
        # or don't
        bar = null_context()
        _inner_func = func

    # -------------------------------------------

    # shortcut single process
    if nproc == 1:
        with bar:
            return list(map(_inner_func, inputs))

    # create input and output queues
    q_in = Queue()
    q_out = Queue()

    # create child processes and start
    proclist = [Process(target=process_in_out_queues,
                        args=(_inner_func, q_in, q_out)) for _ in range(nproc)]

    with bar:

        for proc in proclist:
            proc.daemon = True
            proc.start()

        # populate queue (no need to block in serial put())
        sent = list(map(lambda x: q_in.put(x, block=False), enumerate(inputs)))
        for _ in range(nproc):  # add sentinel for each process
            q_in.put((None, None))

        # get results
        res = [q_out.get() for _ in range(len(sent))]

        # close processes and unwrap results
        for proc in proclist:
            proc.join()

    results = [out for i, out in sorted(res, key=itemgetter(0))]

    # raise exceptions
    if raise_exceptions:
        for exc in results:
            if isinstance(exc, Exception):
                raise exc

    return results


# -- progress bars with counter locking ---------------------------------------
# this is very much a hack, and should be pushed upstream to astropy proper

class _MultiProgressBar(ProgressBar):
    def update(self, value=None):
        # Update self.value
        if value is None:
            self._current_value.value += 1
        else:
            self._current_value = Value('i', value)
        value = self._current_value.value

        # Choose the appropriate environment
        if self._ipython_widget:
            self._update_ipython_widget(value)
        else:
            self._update_console(value)


class _MultiProgressBarOrSpinner(ProgressBarOrSpinner):
    def __init__(self, total, msg, color='default', file=None):
        if total is None:
            super(_MultiProgressBarOrSpinner, self).__init__(
                total, msg, color=color, file=file)
        else:
            self._is_spinner = False
            color_print(msg, color, file=file)
            self._obj = _MultiProgressBar(total, file=file)
