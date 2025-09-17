# Copyright (c) 2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Utilities for multi-processing."""

from __future__ import annotations

from multiprocessing import (
    Process,
    Queue,
)
from operator import itemgetter
from typing import TYPE_CHECKING

from .progress import progress_bar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    T = TypeVar("T")
    R = TypeVar("R")


def _process_in_out_queues(
    func: Callable,
    q_in: Queue,
    q_out: Queue,
) -> None:
    """Iterate through a Queue, call, ``func`, and Queue the result.

    Parameters
    ----------
    func : `callable`
        Any function that can take an element of the input `Queue` as
        the only argument.

    q_in : `multiprocessing.queue.Queue`
        The input `Queue`.

    q_out : `multiprocessing.queue.Queue`
        The output `Queue`.

    Notes
    -----
    To close the input `Queue`, add ``(None, None)`` as the last item.
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
        except Exception as exc:
            q_out.put((idx, exc))


def multiprocess_with_queues(
    nproc: int,
    func: Callable[[T], R],
    inputs: list[T],
    *,
    verbose: bool = False,
    **progress_kw,
) -> list[R]:
    """Map a function over a list of inputs using multiprocess.

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

    progress_kw : `dict`, optional
        Keyword arguments to pass to `gwpy.utils.progress.progress_bar`
        for customising the progress bar, e.g. ``unit='events'``.

    Returns
    -------
    outputs : `list`
        The `list` of results from calling ``func(x)`` for each element
        of ``inputs``.
    """
    # create progress bar for verbose output
    if bool(verbose):
        if not isinstance(verbose, bool):
            progress_kw["desc"] = str(verbose)
        if isinstance(inputs, list | tuple):
            progress_kw.setdefault("total", len(inputs))
        pbar = progress_bar(**progress_kw)
    else:
        pbar = None

    # -------------------------------------------

    # shortcut single process
    if nproc == 1:

        def _inner(x: T) -> R:
            try:
                return func(x)
            finally:
                if pbar:
                    pbar.update(1)

        return list(map(_inner, inputs))

    # -------------------------------------------

    # create input and output queues
    q_in: Queue = Queue()
    q_out: Queue = Queue()

    # create child processes and start
    proclist = [
        Process(
            target=_process_in_out_queues,
            args=(func, q_in, q_out),
        )
        for _ in range(nproc)
    ]

    for proc in proclist:
        proc.daemon = True
        proc.start()

    # populate queue (no need to block in serial put())
    for item in enumerate(inputs):
        q_in.put(item, block=False)
    for _ in range(nproc):  # add sentinel for each process
        q_in.put((None, None))

    # get results
    out = []
    for _ in range(len(inputs)):
        x: tuple[int, R] = q_out.get()
        if pbar:
            pbar.update()
        out.append(x)

    # close processes and unwrap results
    for proc in proclist:
        proc.join()

    if pbar:
        pbar.close()

    # unwrap results in order
    results = [res for _, res in sorted(out, key=itemgetter(0))]

    # raise exceptions here
    for res in results:
        if isinstance(res, Exception):
            raise res

    return results
