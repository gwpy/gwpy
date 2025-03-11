# Copyright (c) 2021-2025 Cardiff University
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

"""Custom pytest marks for GWpy.

This is mainly to allow downstream users/builders/packagers to skip tests
that require annoying/flaky configuration, or just take too long.

This module is imported in :mod:`gwpy.conftest` such that all marks are
registered up front and visible via ``python -m pytest gwpy --markers``.
"""

MARKS: dict[str, str] = {
    # nothing here, used to support 'cvmfs' but that was removed
}


def _register_marks(config):
    """Register all marks for GWpy.

    This function is designed to be called from :mod:`gwpy.conftest`.
    """
    for name, doc in MARKS.items():
        config.addinivalue_line(
            "markers",
            f"{name}: {doc}",
        )
