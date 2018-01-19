#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2016)
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

import sys
from distutils.version import LooseVersion

from setuptools import (setup, find_packages,
                        __version__ as setuptools_version)

import versioneer
from setup_utils import (CMDCLASS, get_setup_requires, get_scripts)

__version__ = versioneer.get_version()

# -- dependencies -------------------------------------------------------------

# build dependencies
setup_requires = get_setup_requires()

# package dependencies
install_requires = [
    'numpy>=1.7.1',
    'scipy>=0.12.1',
    'matplotlib>=1.2.0',
    'astropy>=1.1.1',
    'six>=1.5',
    'lscsoft-glue>=1.55.2',
    'python-dateutil',
]

# exclude matplotlib 2.1.x (see matplotlib/matplotlib#10003) if possible
if LooseVersion(setuptools_version) >= '25':  # exclude matplotlib 2.1.x
    install_requires[2] += ',!=2.1.*'

# test for LAL
try:
    import lal  # pylint: disable=unused-import
except ImportError as e:
    install_requires.append('ligotimegps>=1.2.1')

# enum34 required for python < 3.4
try:
    import enum  # pylint: disable=unused-import
except ImportError:
    install_requires.append('enum34')

# define extras
extras_require = {
    'hdf5': ['h5py>=1.3'],
    'root': ['root_numpy'],
    'segments': ['dqsegdb'],
    'hacr': ['pymysql'],
    'docs': ['sphinx>=1.6.1', 'numpydoc', 'sphinx-bootstrap-theme>=0.6',
             'sphinxcontrib-programoutput', 'sphinx-automodapi'],
}

# define 'all' as the intersection of all extras
extras_require['all'] = set(p for extra in extras_require.values()
                            for p in extra)

# test dependencies
tests_require = [
    'pytest>=3.1',
    'freezegun',
    'sqlparse',
    'bs4',
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
    long_description=("GWpy is a collaboration-driven Python package "
                      "providing tools for studying data from "
                      "ground-based gravitational-wave detectors"),
    author='Duncan Macleod',
    author_email='duncan.macleod@ligo.org',
    license='GPLv3',
    url='https://gwpy.github.io/',

    # package content
    packages=find_packages(),
    scripts=get_scripts(),
    include_package_data=True,

    # dependencies
    cmdclass=CMDCLASS,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,

    # classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
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
