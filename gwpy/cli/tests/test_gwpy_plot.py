# Copyright (c) 2014-2017 Louisiana State University
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

"""Tests for `gwpy-plot` command line module `gwpy.cli.gwpy_plot`."""

import pytest

from .. import (
    PRODUCTS,
    gwpy_plot,
)
from .base import NDS2_CONNECTION_FIXTURE_DATA  # noqa: F401


@pytest.mark.parametrize("mode", [None, *list(PRODUCTS.keys())])
def test_gwpy_plot_help(mode):
    args = [mode, "--help"] if mode else ["--help"]
    with pytest.raises(SystemExit) as exc:
        gwpy_plot.main(args)
    assert exc.value.code == 0


@pytest.mark.requires("nds2")
def test_gwpy_plot_timeseries(tmp_path, nds2_connection):
    tmp = tmp_path / "plot.png"
    args = [
        "timeseries",
        "--chan", "X1:TEST-CHANNEL",
        "--start", 0,
        "--nds2-server", "nds.test.gwpy",  # don't use datafind
        "--out", str(tmp),
    ]
    exitcode = gwpy_plot.main(args)
    assert not exitcode  # passed
    assert tmp.is_file()  # plot was created
