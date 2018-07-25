#!/bin/bash
# Copyright (C) Duncan Macleod (2017)
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

#
# Run the test suite for GWpy on the current system
#

. ci/lib.sh

# move to new empty directory (so that tests run on installed code)
mkdir -p test
pushd test

get_environment  # sets PIP etc
get_python_version  # sets PYTHON_VERSION

# macports PATH doesn't persist from install stage, which is annoying
if [ $(get_package_manager) == port ]; then
    . terryfy/travis_tools.sh
    export PATH=$MACPORTS_PREFIX/bin:$PATH
fi

set -ex

# upgrade pip to some minimal level
# NOTE: need >=7.0.0 to understand '.[tests]'
#       need >=9.0.0 to understand list --format
${PIP} install "pip>=9.0.0"

# print installed packages
echo "------------------------------------------------------------------------"
echo
echo "GWpy installed to $(${PYTHON} -c 'import gwpy; print(gwpy.__file__)')"
echo
echo "------------------------------------------------------------------------"

echo "Dependencies:"
echo "-------------"
${PYTHON} -m pip list installed --format=columns

# run tests
${PYTHON} -m pytest --pyargs gwpy --cov=gwpy

popd > /dev/null  # return to starting directory

cp -v test/.coverage .
