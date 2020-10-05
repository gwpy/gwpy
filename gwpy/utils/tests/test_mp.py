# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

from math import sqrt

import pytest

from .. import mp as utils_mp

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.mark.parametrize('verbose', [False, True, 'Test'])
@pytest.mark.parametrize('nproc', [1, 2])
def test_multiprocess_with_queues(capsys, nproc, verbose):
    inputs = [1, 4, 9, 16, 25]
    out = utils_mp.multiprocess_with_queues(
        nproc,
        sqrt,
        inputs,
        verbose=verbose,
    )
    assert out == [1, 2, 3, 4, 5]

    # assert progress bar prints correctly
    stdout = capsys.readouterr().out.strip()
    if verbose is True:
        assert stdout.startswith('Processing: ')
    elif verbose is False:
        assert stdout == ''
    else:
        assert stdout.startswith(verbose)


@pytest.mark.parametrize('nproc', [1, 2])
def test_multiprocess_with_queues_errors(nproc):
    """Check that errors from child processes propagate to the parent
    """
    with pytest.raises(ValueError) as exc:
        utils_mp.multiprocess_with_queues(
            nproc,
            sqrt,
            [-1, -1, -1, -1],
        )
    assert str(exc.value) == "math domain error"


def test_multiprocess_with_queues_raise():
    with pytest.deprecated_call():
        utils_mp.multiprocess_with_queues(
            1,
            sqrt,
            [1],
            raise_exceptions=True,
        )
