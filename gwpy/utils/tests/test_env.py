# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Tests for :mod:`gwpy.utils.env`
"""

from unittest import mock

import pytest

from .. import env as utils_env

BOOL_TRUE = {
    'TEST_y': 'y',
    'TEST_Y': 'Y',
    'TEST_yes': 'yes',
    'TEST_Yes': 'Yes',
    'TEST_YES': 'YES',
    'TEST_ONE': '1',
    'TEST_true': 'true',
    'TEST_True': 'True',
    'TEST_TRUE': 'TRUE',
}
BOOL_FALSE = {
    'TEST_no': 'no',
    'TEST_No': 'No',
    'TEST_ZERO': '0',
    'TEST_false': 'false',
    'TEST_False': 'False',
    'TEST_FALSE': 'FALSE',
    'TEST_OTHER': 'blah',
}
BOOL_ENV = BOOL_TRUE.copy()
BOOL_ENV.update(BOOL_FALSE)


@mock.patch.dict('os.environ', values=BOOL_ENV)
@pytest.mark.parametrize(
    'env, result',
    [(k, True) for k in sorted(BOOL_TRUE)]
    + [(k, False) for k in sorted(BOOL_FALSE)],
)
def test_bool_env(env, result):
    """Test :meth:`gwpy.utils.env.bool_env` _without_ the `default` keyword
    """
    assert utils_env.bool_env(env) is result


@mock.patch.dict('os.environ', values=BOOL_TRUE)
@pytest.mark.parametrize('env, default, result', [
    ('TEST_YES', False, True),
    ('TEST_MISSING', True, True),
])
def test_bool_env_default(env, default, result):
    """Test :meth:`gwpy.utils.env.bool_env` _with_ the `default` keyword
    """
    assert utils_env.bool_env(env, default=default) is result
