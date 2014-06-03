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

"""Setup the GWpy package
"""

from __future__ import print_function

import sys
if sys.version < '2.6':
    raise ImportError("Python versions older than 2.6 are not supported.")

import glob
import os.path
import ez_setup
ez_setup.use_setuptools()

from distutils.command.clean import (clean, log, remove_tree)
from setuptools import (setup, find_packages)

# test for OrderedDict
extra_install_requires = []
try:
    from collections import OrderedDict
except ImportError:
    extra_install_requires.append('ordereddict>=1.1')

# set basic metadata
PACKAGENAME = 'gwpy'
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

# -----------------------------------------------------------------------------
# Process complicated dependencies

try:
    from glue import git_version
except ImportError:
    print("GWpy requires the GLUE package, which isn\'t available from PyPI.\n"
          "Please visit\n" 
          "https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html\n"
          "to download and install it manually.", file=sys.stderr)
    sys.exit(1)

NUMPY_REQUIRED = '1.5'

if 'pip-' in __file__:
    no_numpy = False
    numpy_too_old = False
    try:
        import numpy
    except ImportError:
        no_numpy = True
    else:
        if numpy.__version__ < NUMPY_REQUIRED:
            numpy_too_old = True

    if no_numpy or numpy_too_old:
        print("BUILD FAILURE ANTICIPATED", file=sys.stderr)
        print("Pip does not install dependencies in logical order, and so "
              "will not completely build numpy before moving onto matplotlib, "
              "meaning the matplotlib build will fail. Please install numpy "
              "first by running:", file=sys.stderr)
    if no_numpy:
        print("pip install numpy", file=sys.stderr)
    elif numpy_too_old:
        print("pip install --upgrade numpy", file=sys.stderr)
        sys.exit(1)

# -----------------------------------------------------------------------------
# Clean up after sphinx

class GWpyClean(clean):
    def run(self):
        if self.all:
            sphinx_dir = os.path.join(self.build_base, 'sphinx')
            if os.path.exists(sphinx_dir):
                remove_tree(sphinx_dir, dry_run=self.dry_run)
            else:
                log.warn("'%s' does not exist -- can't clean it", sphinx_dir)
        clean.run(self)


# -----------------------------------------------------------------------------
# Set version information

_version = __import__('utils.version', fromlist=[''])
vcinfo = _version.GitStatus()

# write version information to gwpy.version
vcinfo('%s/version.py' % PACKAGENAME, PACKAGENAME, AUTHOR, AUTHOR_EMAIL)

# get version metadata
VERSION = vcinfo.version
RELEASE = vcinfo.version != vcinfo.id and 'dev' not in VERSION

# -----------------------------------------------------------------------------
# Get more metadata from __init__.py

import gwpy
DESCRIPTION, LONG_DESCRIPTION = gwpy.__doc__.split('\n', 1)

# -----------------------------------------------------------------------------
# Find files

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages(exclude=['utils', 'utils.*'])

# glob for all scripts
if os.path.isdir('bin'):
    scripts = glob.glob(os.path.join('bin', '*'))
else:
    scripts = []

# -----------------------------------------------------------------------------
# Define dependencies

requires = dict()

# define core dependencies
requires['install'] = ['astropy >= 0.3', 'numpy >= 1.5', 'python-dateutil',
                       'matplotlib >= 1.3.0', 'glue'] + extra_install_requires

extras = dict()
extras['nds'] = ['nds2']

# -----------------------------------------------------------------------------
# run setup

setup(name=PACKAGENAME,
      provides=[PACKAGENAME],
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION.strip('\n'),
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url='https://gwpy.github.io/',
      packages=packagenames,
      #package_data={
      #    PACKAGENAME: ['gwpy/tests/data/*'],
      #    },
      include_package_data=True,
      cmdclass=dict(clean=GWpyClean),
      scripts=scripts,
      requires=[
          'glue',
          'dateutil',
          'numpy',
          'matplotlib',
          'astropy'],
      install_requires=[
          'python-dateutil',
          'numpy >= %s' % NUMPY_REQUIRED,
          'matplotlib >= 1.3.0',
          'astropy >= 0.3',
          ] + extra_install_requires,
      extras_require={
          'nds': ['nds2-client'],
          'gwf': ['frameCPP'],
          'doc': ['sphinx'],
          'dsp': ['scipy'],
          },
      test_suite='gwpy.tests',
      use_2to3=False,
      classifiers=[
          'Programming Language :: Python',
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          ],
      )
