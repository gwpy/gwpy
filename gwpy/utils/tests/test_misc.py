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

"""Tests for :mod:`gwpy.utils.misc`
"""

import sys

import pytest

from .. import misc as utils_misc


def test_gprint(capsys):
    """Test for :func:`gwpy.utils.misc.gprint`
    """
    utils_misc.gprint('test')
    assert capsys.readouterr().out == 'test\n'
    utils_misc.gprint('test', end=' ')
    assert capsys.readouterr().out == 'test '
    utils_misc.gprint('test', end='x', file=sys.stderr)
    cap = capsys.readouterr()
    assert not cap.out
    assert cap.err == 'testx'


def test_null_context():
    """Test for :func:`gwpy.utils.misc.null_context`
    """
    with pytest.warns(DeprecationWarning):
        ctx = utils_misc.null_context()
    with ctx:
        print('this should work')


def test_round_to_power():
    """Test for :func:`gwpy.utils.misc.round_to_power`
    """
    # test basic features
    assert utils_misc.round_to_power(2) == 2
    assert utils_misc.round_to_power(9, base=10) == 10
    assert utils_misc.round_to_power(5, which='lower') == 4
    assert utils_misc.round_to_power(5, which='upper') == 8
    # test output
    base = 10.
    rounded = utils_misc.round_to_power(9, base=base)
    assert base == rounded
    assert type(base) == type(rounded)


def test_round_to_power_error():
    """Test for an errored use case of :func:`gwpy.utils.misc.round_to_power`
    """
    with pytest.raises(ValueError) as exc:
        utils_misc.round_to_power(7, which='')
    assert str(exc.value) == (
        "'which' argument must be one of 'lower', 'upper', or None")


def test_unique():
    """Test for :func:`gwpy.utils.misc.unique`
    """
    a = [1, 2, 4, 3, 5, 4, 5, 3]
    assert utils_misc.unique(a) == [1, 2, 4, 3, 5]


@pytest.mark.parametrize('func, value, out', [
    (str, None, None),
    (str, 1, '1'),
])
def test_if_not_none(func, value, out):
    """Test for :func:`gwpy.utils.misc.if_not_none`
    """
    assert utils_misc.if_not_none(func, value) == out
