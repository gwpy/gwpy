#!/usr/bin/env python

# Copyright (C) Duncan Macleod (2013)
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

import sys
import imp
try:
    # This incantation forces distribute to be used (over setuptools) if it is
    # available on the path; otherwise distribute will be downloaded.
    import pkg_resources
    distribute = pkg_resources.get_distribution('distribute')
    if pkg_resources.get_distribution('setuptools') != distribute:
        sys.path.insert(1, distribute.location)
        distribute.activate()
        imp.reload(pkg_resources)
except:  # There are several types of exceptions that can occur here
    from distribute_setup import use_setuptools
    use_setuptools()

import glob
import os
from setuptools import setup, find_packages

#A dirty hack to get around some early import/configurations ambiguities
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

from astropy.version_helpers import get_git_devstr, generate_version_py

PACKAGENAME = 'gwpy'
DESCRIPTION = 'Community package for gravitational wave astronomy in Python'
LONG_DESCRIPTION = ''
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.0.dev'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(False)

# Freeze build information in version.py
generate_version_py(PACKAGENAME, VERSION, RELEASE)

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=packagenames,
      ext_modules=[],
      requires=['astropy', 'glue', 'numpy', 'lal', 'lalframe'],
      install_requires=['astropy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      use_2to3=True
      )
