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

"""Unit test for utils module
"""

import os
from math import sqrt

import pytest

from .. import mp as utils_mp
from ..misc import null_context

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def _ctx(nproc):
    """Returns the correct context in which to run `multiprocess_with_queues`

    On Windows, with ``nproc>1``, a `UserWarning` will be raised by
    :func:`gwpy.utils.mp.multiprocess_with_queues` because multiprocessing
    doesn't work.
    """
    if nproc > 1 and os.name == 'nt':
        return pytest.warns(UserWarning)
    return null_context()


@pytest.mark.filterwarnings(
    'always:multiprocessing is currently not supported on Windows')
@pytest.mark.parametrize('verbose', [False, True, 'Test'])
@pytest.mark.parametrize('nproc', [1, 2])
def test_multiprocess_with_queues(capsys, nproc, verbose):
    inputs = [1, 4, 9, 16, 25]

    with _ctx(nproc):  # on windows, mp warns and falls back to nproc=1
        out = utils_mp.multiprocess_with_queues(
            nproc, sqrt, inputs, verbose=verbose,
        )
    assert out == [1, 2, 3, 4, 5]

    # assert progress bar prints correctly
    cap = capsys.readouterr()
    if verbose is True:
        assert cap.out.startswith('\rProcessing: ')
    elif verbose is False:
        assert cap.out == ''
    else:
        assert cap.out.startswith('\r{}: '.format(verbose))

    with _ctx(nproc), pytest.raises(ValueError):
        utils_mp.multiprocess_with_queues(nproc, sqrt, [-1],
                                          verbose=verbose)


def test_multiprocess_with_queues_raise():
    with pytest.warns(DeprecationWarning):
        utils_mp.multiprocess_with_queues(1, sqrt, [1],
                                          raise_exceptions=True)
