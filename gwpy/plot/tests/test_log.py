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

"""Tests for `gwpy.plot.log`
"""

from unittest import mock

import pytest

from matplotlib import (
    rc_context,
)

from .. import log as plot_log


class TestLogFormatter(object):
    TEST_CLASS = plot_log.LogFormatter

    @classmethod
    @pytest.fixture
    def formatter(cls):
        with mock.patch(
            "gwpy.plot.log.LogFormatter._num_ticks",
            return_value=2,
        ):
            yield cls.TEST_CLASS()

    @pytest.mark.parametrize('x, fmt, result, texresult', [
        pytest.param(
            0.,
            None,
            r'$\mathdefault{0}$',
            '$0$',
            id="0",
        ),
        pytest.param(
            1,
            None,
            r'$\mathdefault{10^{0}}$',
            r'$\mathdefault{10^{0}}$',
            id="fmt=None",
        ),
        pytest.param(
            1,
            "%s",
            r'$\mathdefault{1}$',
            r'$1$',
            id="fmt=%s",
        ),
    ])
    def test_call(self, formatter, x, fmt, result, texresult):
        with rc_context(rc={'text.usetex': False}):
            assert formatter(x, fmt=fmt) == result
        with rc_context(rc={'text.usetex': True}):
            assert formatter(x, fmt=fmt) == texresult

    @mock.patch(  # we don't need this function for this test
        "gwpy.plot.log.LogFormatter.set_locs",
        mock.MagicMock(),
    )
    @pytest.mark.parametrize("values, result", [
        # normal output
        pytest.param(
            [1e-1, 1e2, 1e5, 1e8],
            [plot_log._math(x) for x in
             ("10^{-1}", "10^{2}", "10^{5}", "10^{8}")],
            id="mpl",
        ),
        # custom output
        pytest.param(
            [1e-1, 1e0, 1e1, 1e2],
            [plot_log._math(x) for x in ("0.1", "1", "10", "100")],
            id="gwpy",
        ),
    ])
    def test_format_ticks(self, formatter, values, result):
        assert formatter.format_ticks(values) == result
