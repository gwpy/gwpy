# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

pytestmark = pytest.mark.filterwarnings(
    'always:multiprocessing is currently not supported on Windws')


@pytest.mark.parametrize('verbose', [False, True, 'Test'])
@pytest.mark.parametrize('nproc', [1, 2])
def test_multiprocess_with_queues(capsys, nproc, verbose):
    inputs = [1, 4, 9, 16, 25]
    if os.name == 'nt' and nproc > 1:
        ctx = pytest.warns(UserWarning)
    else:
        ctx = null_context()

    with ctx:  # on windows, mp prints warning and falls back to nproc=1
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

    with pytest.raises(ValueError):
        utils_mp.multiprocess_with_queues(nproc, sqrt, [-1],
                                          verbose=verbose)


def test_multiprocess_with_queues_raise():
    with pytest.deprecated_call():
        utils_mp.multiprocess_with_queues(1, sqrt, [1],
                                          raise_exceptions=True)
