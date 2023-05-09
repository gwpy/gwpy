# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
#
# This file is part of pyDischarge.
#
# pyDischarge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDischarge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyDischarge.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for `pydischarge-plot` command line module `pydischarge.cli.pydischarge_plot`
"""

from unittest import mock

import pytest

from .. import (
    PRODUCTS,
    pydischarge_plot,
)
from .base import mock_nds2_connection


@pytest.mark.parametrize("mode", [None] + list(PRODUCTS.keys()))
def test_pydischarge_plot_help(mode):
    args = [mode, "--help"] if mode else ["--help"]
    with pytest.raises(SystemExit) as exc:
        pydischarge_plot.main(args)
    assert exc.value.code == 0


@pytest.mark.requires("nds2")
def test_pydischarge_plot_timeseries(tmp_path):
    tmp = tmp_path / "plot.png"
    with mock.patch(
        'nds2.connection',
        return_value=mock_nds2_connection()[0],
    ):
        args = [
            "timeseries",
            "--chan", "X1:TEST-CHANNEL",
            "--start", 0,
            "--nds2-server", "nds.test.pydischarge",  # don't use datafind
            "--out", str(tmp),
        ]
        exitcode = pydischarge_plot.main(args)
        assert not exitcode  # passed
        assert tmp.is_file()  # plot was created
