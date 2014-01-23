#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import glob
import os.path
from setuptools import setup, find_packages

utils = __import__('utils', fromlist=['version'], level=1)

PACKAGENAME = 'gwpy'
DESCRIPTION = 'Community package for gravitational wave astronomy in Python'
LONG_DESCRIPTION = ''
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

# set version information
VERSION_PY = '%s/version.py' % PACKAGENAME
vcinfo = utils.version.GitStatus()
vcinfo(VERSION_PY, PACKAGENAME, AUTHOR, AUTHOR_EMAIL)

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = vcinfo.version

# Indicates if this version is a release version
RELEASE = vcinfo.version != vcinfo.id and 'dev' not in VERSION

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

# glob for all scripts
if os.path.isdir('bin'):
    scripts = glob.glob(os.path.join('bin', '*'))
else:
    scripts = []

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=packagenames,
      ext_modules=[],
      scripts=scripts,
      install_requires=['astropy >= 0.3', 'glue'],
      requires=['astropy', 'glue', 'numpy', 'lal', 'lalframe', 'matplotlib',
                'scipy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,
      use_2to3=True
      )
