# Copyright (c) 2017 Louisiana State University
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

"""Utilities for progress bars."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from typing import TextIO

TQDM_BAR_FORMAT: str = (
    "{desc}: |{bar}| "
    "{n_fmt}/{total_fmt} ({percentage:3.0f}%) "
    "ETA {remaining:6s}"
)


def progress_bar(
    desc: str = "Processing",
    file: TextIO = sys.stdout,
    bar_format: str = TQDM_BAR_FORMAT,
    **kwargs,
) -> tqdm:
    """Create a `tqdm.tqdm` progress bar.

    This is just a thin wrapper around `tqdm.tqdm` with some updated defaults.
    """
    pbar = tqdm(
        desc=desc,
        file=file,
        bar_format=bar_format,
        **kwargs,
    )
    if not pbar.disable:
        pbar.desc = pbar.desc.rstrip(": ")
        pbar.refresh()
    return pbar
