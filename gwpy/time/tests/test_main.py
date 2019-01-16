# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019)
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

"""Tests for the `gwpy.time` command-line interface
"""

import pytest

from .. import __main__ as gwpy_time_cli


@pytest.mark.parametrize('args, result', [
    (['Jan 1 2010'], 946339215),
    (['Jan', '1', '2010'], 946339215),
    (['Oct 30 2016 12:34 CST'], 1161887657),
    (['946339215'], '2010-01-01 00:00:00.000000 UTC'),
    (['1161887657'], '2016-10-30 18:34:00.000000 UTC'),
])
def test_main(args, result, capsys):
    gwpy_time_cli.main(args)
    out, err = capsys.readouterr()
    assert not err
    assert out.rstrip() == str(result)
