# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Utilities for multi-processing
"""

import sys

from tqdm import tqdm

TQDM_BAR_FORMAT = ("{desc}: |{bar}| "
                   "{n_fmt}/{total_fmt} ({percentage:3.0f}%) "
                   "ETA {remaining:6s}")


def progress_bar(**kwargs):
    """Create a `tqdm.tqdm` progress bar

    This is just a thin wrapper around `tqdm.tqdm` to set some updated defaults
    """
    tqdm_kw = {
        'desc': 'Processing',
        'file': sys.stdout,
        'bar_format': TQDM_BAR_FORMAT,
    }
    tqdm_kw.update(kwargs)
    pbar = tqdm(**tqdm_kw)
    if not pbar.disable:
        pbar.desc = pbar.desc.rstrip(': ')
        pbar.refresh()
    return pbar
