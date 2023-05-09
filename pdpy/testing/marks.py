# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2021)
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

"""Custom pytest marks for pyDischarge

This is mainly to allow downstream users/builders/packagers to skip tests
that require annoying/flaky configuration, or just take too long, e.g.
tests that use CVMFS could be skipped via

    $ python -m pytest --pyargs pydischarge -m 'not cvmfs'

This module is imported in :mod:`pydischarge.conftest` such that all marks are
registered up front and visible via ``python -m pytest pydischarge --markers``.
"""

MARKS = {
    "cvmfs": "mark a test as requiring CVMFS",
}


def _register_marks(config):
    """Register all marks for pyDischarge

    This function is designed to be called from :mod:`pydischarge.conftest`.
    """
    for name, doc in MARKS.items():
        config.addinivalue_line(
            "markers",
            "{}: {}".format(name, doc),
        )
