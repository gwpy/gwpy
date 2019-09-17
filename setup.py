# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016-2019)
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

# ignore all invalid names (pylint isn't good at looking at executables)
# pylint: disable=invalid-name

from __future__ import print_function

import os
import sys
from distutils.version import LooseVersion

from setuptools import (setup, find_packages,
                        __version__ as setuptools_version)

import versioneer
from setup_utils import (CMDCLASS, get_setup_requires, get_scripts)

__version__ = versioneer.get_version()

PEP_508 = LooseVersion(setuptools_version) >= '20.2.2'

# read description
with open('README.md', 'rb') as f:
    longdesc = f.read().decode().strip()

# -- dependencies -------------------------------------------------------------

# build dependencies
setup_requires = get_setup_requires()

# runtime dependencies
install_requires = [
    'astropy >= 1.1.1, < 3.0.0 ; python_version < \'3.5\'',
    'astropy >= 1.1.1 ; python_version >= \'3.5\'',
    'dqsegdb2',
    'enum34 ; python_version < \'3.4\'',
    'gwdatafind',
    'gwosc >= 0.4.0',
    'h5py >= 1.3',
    'ligo-segments >= 1.0.0',
    'ligotimegps >= 1.2.1',
    'matplotlib >= 1.2.0, != 2.1.0, != 2.1.1',
    'numpy >= 1.7.1',
    'pathlib ; python_version < \'3.4\'',
    'python-dateutil',
    'scipy >= 0.12.1',
    'six >= 1.5',
    'tqdm >= 4.10.0',
]

# if setuptools is too old and we are building an EL7 or Debian 8
# distribution, empty the install_requires, the distribution files
# will handle dependencies anyway
# NOTE: this probably isn't very robust
if not PEP_508 and (
        os.getenv('RPM_BUILD_ROOT') or os.getenv('PYBUILD_NAME')):
    install_requires = []

# test dependencies
tests_require = [
    'pytest>=3.3.0,<5.0.0',
    'freezegun>=0.2.3',
    'sqlparse>=0.2.0',
    'beautifulsoup4',
]
if sys.version < '3':
    tests_require.append('mock')

# -- run setup ----------------------------------------------------------------

setup(
    # metadata
    name='gwpy',
    provides=['gwpy'],
    version=__version__,
    description="A python package for gravitational-wave astrophysics",
    long_description=longdesc,
    long_description_content_type='text/markdown',
    author='Duncan Macleod',
    author_email='duncan.macleod@ligo.org',
    license='GPLv3+',
    url='https://github.com/gwpy/gwpy',

    # package content
    packages=find_packages(),
    scripts=get_scripts(),
    include_package_data=True,

    # dependencies
    cmdclass=CMDCLASS,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,

    # classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
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
        'License :: OSI Approved :: '
        'GNU General Public License v3 or later (GPLv3+)',
    ],
)
