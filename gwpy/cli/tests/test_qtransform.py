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

"""Unit tests for :mod:`gwpy.cli.qtransform`."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import pytest

from .. import QtransformProduct
from .base import update_namespace
from .test_spectrogram import TestSpectrogramProduct

if TYPE_CHECKING:
    from pathlib import Path
    from typing import ClassVar


class TestQtransformProduct(TestSpectrogramProduct):
    """Tests for `gwpy.cli.QtransformProduct`."""

    TEST_CLASS = QtransformProduct
    ACTION = "qtransform"
    TEST_ARGS: ClassVar[list[str]] = [
        "--chan", "X1:TEST-CHANNEL",
        "--gps", "5",
        "--search", "10",
        "--nds2-server", "nds.test.gwpy",
        "--outdir", os.path.curdir,
        "--average-method", "median",
    ]

    def test_finalize_arguments(self, prod):
        """Test finalising arguments for `Qtransform`."""
        assert prod.start_list == [prod.args.gps - prod.args.search/2.]
        assert prod.duration == prod.args.search
        assert prod.args.color_scale == "linear"
        assert prod.args.xmin is None
        assert prod.args.xmax is None

    def test_init(self, args):
        """Test initialising a `Qtransform`."""
        update_namespace(args, qrange=(100., 110.))
        prod = self.TEST_CLASS(args)
        assert prod.qxfrm_args["gps"] == prod.args.gps
        assert prod.qxfrm_args["qrange"] == (100., 110.)

    def test_get_title(self, dataprod: QtransformProduct):
        """Test `QtransformProduct.get_title()`."""
        _float_reg = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
        title_reg = re.compile(
            r"\A"
            fr"q: {_float_reg}, "
            fr"tres: {_float_reg}, "
            r"whitened, "
            fr"f-range: \[{_float_reg}, {_float_reg}\], "
            fr"e-range: \[{_float_reg}, {_float_reg}\]"
            r"\Z",
            re.IGNORECASE,
        )
        assert title_reg.match(dataprod.get_title())

    def test_get_suptitle(self, prod: QtransformProduct):
        """Test `QtransformProduct.get_suptitle()`."""
        assert prod.get_suptitle() == f"Q-transform: {prod.chan_list[0]}"

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_run(self, prod: QtransformProduct, tmp_path: Path):
        """Test `QtransformProduct.run()`."""
        prod.args.out = tmp_path
        prod.run()
        assert (tmp_path / "X1-TEST_CHANNEL-5.0-0.5.png").is_file()
        assert prod.plot_num == 1
        assert not prod.has_more_plots()
