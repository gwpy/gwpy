#!/bin/bash
# Copyright (C) Cardiff University (2019-2020)
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

set -ex

# get local functions
. ci/lib.sh

#
# Install this package
#

find_python_and_pip

# update pip to support --progress-bar
${PIP} install --quiet "pip>=10.0.0" "wheel"

# install from tarball if exists, or cwd
if ls gwpy-*.tar.* &> /dev/null; then
    ${PIP} install --progress-bar=off gwpy-*.tar.*
else
    ${PIP} install --progress-bar=off .
fi
