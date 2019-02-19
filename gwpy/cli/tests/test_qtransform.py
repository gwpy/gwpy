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

"""Unit tests for :mod:`gwpy.cli.qtransform`
"""

import os
import re

from ... import cli
from ...testing.compat import mock
from ...testing.utils import skip_missing_dependency
from .base import (update_namespace, mock_nds2_connection)
from .test_spectrogram import TestCliSpectrogram as _TestCliSpectrogram


class TestCliQtransform(_TestCliSpectrogram):
    TEST_CLASS = cli.Qtransform
    ACTION = 'qtransform'
    TEST_ARGS = [
        '--chan', 'X1:TEST-CHANNEL', '--gps', '5', '--search', '10',
        '--nds2-server', 'nds.test.gwpy', '--outdir', os.path.curdir,
    ]

    def test_finalize_arguments(self, prod):
        assert prod.start_list == [prod.args.gps - prod.args.search/2.]
        assert prod.duration == prod.args.search
        assert prod.args.color_scale == 'linear'
        assert prod.args.xmin is None
        assert prod.args.xmax is None

    def test_init(self, args):
        update_namespace(args, qrange=(100., 110.))
        prod = self.TEST_CLASS(args)
        assert prod.qxfrm_args['gps'] == prod.args.gps
        assert prod.qxfrm_args['qrange'] == (100., 110.)

    def test_get_title(self, dataprod):
        _float_reg = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
        title_reg = re.compile(
            r"\A"
            r"q: {float}, "
            r"tres: {float}, "
            r"whitened, "
            r"f-range: \[{float}, {float}\], "
            r"e-range: \[{float}, {float}\]"
            r"\Z".format(float=_float_reg),
            re.I,
        )
        assert title_reg.match(dataprod.get_title())

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Q-transform: {0}'.format(
            prod.chan_list[0])

    @skip_missing_dependency('nds2')
    def test_run(self, prod):
        conn, _ = mock_nds2_connection()
        outf = 'X1-TEST_CHANNEL-5.0-0.5.png'
        with mock.patch('nds2.connection') as mocker:
            mocker.return_value = conn
            try:
                prod.run()
                assert os.path.isfile(outf)
            finally:
                if os.path.isfile(outf):
                    os.remove(outf)
            assert prod.plot_num == 1
            assert not prod.has_more_plots()
