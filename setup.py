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
import hashlib
import os.path
import subprocess

try:
    import setuptools
except ImportError:
    import ez_setup
    ez_setup.use_setuptools()
finally:
    from setuptools import (setup, find_packages)
    from setuptools.command import (build_py, egg_info)

from distutils.dist import Distribution
from distutils.cmd import Command
from distutils.command.clean import (clean, log, remove_tree)

# test for OrderedDict
extra_install_requires = []
try:
    from collections import OrderedDict
except ImportError:
    extra_install_requires.append('ordereddict>=1.1')

# importlib required for cli programs
try:
    from importlib import import_module
except ImportError:
    extra_install_requires.append('importlib>=1.0.3')

# test for unittest2
extra_tests_require = []
if sys.version < '2.7':
    extra_tests_require.append('unittest2')

# import sphinx commands
try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    cmdclass = {}
else:
    cmdclass = {'build_sphinx': BuildDoc}

# set basic metadata
PACKAGENAME = 'gwpy'
AUTHOR = 'Duncan Macleod'
AUTHOR_EMAIL = 'duncan.macleod@ligo.org'
LICENSE = 'GPLv3'

# set versioning information
import versioneer
__version__ = versioneer.get_version()
cmdclass.update(versioneer.get_cmdclass())


# -----------------------------------------------------------------------------
# Clean up, including Sphinx, and setup_requires eggs

class GWpyClean(clean):
    def run(self):
        if self.all:
            # remove dist
            if os.path.exists('dist'):
                remove_tree('dist')
            else:
                log.warn("'dist' does not exist -- can't clean it")
            # remove docs
            sphinx_dir = os.path.join(self.build_base, 'sphinx')
            if os.path.exists(sphinx_dir):
                remove_tree(sphinx_dir, dry_run=self.dry_run)
            else:
                log.warn("%r does not exist -- can't clean it", sphinx_dir)
            # remove setup eggs
            for egg in glob.glob('*.egg'):
                if os.path.isdir(egg):
                    remove_tree(egg, dry_run=self.dry_run)
                else:
                    log.info('removing %r' % egg)
                    os.unlink(egg)
            # remove Portfile
            portfile = 'Portfile'
            if os.path.exists(portfile) and not self.dry_run:
                log.info('removing %r' % portfile)
                os.unlink(portfile)
        clean.run(self)

cmdclass['clean'] = GWpyClean


# -- build a Portfile for macports --------------------------------------------

class BuildPortfile(Command):
    """Generate a Macports Portfile for this project from the current build
    """
    description = 'Generate Macports Portfile'
    user_options = [
        ('version=', None, 'the X.Y.Z package version'),
        ('portfile=', None, 'target output file, default: \'Portfile\''),
        ('template=', None,
         'Portfile template, default: \'Portfile.template\''),
    ]

    def initialize_options(self):
        self.version = None
        self.portfile = 'Portfile'
        self.template = 'Portfile.template'
        self._template = None

    def finalize_options(self):
        from jinja2 import Template
        with open(self.template, 'r') as t:
            self._template = Template(t.read())

    def run(self):
        # get version from distribution
        if self.version is None:
            self.version = __version__
        # find dist file
        dist = os.path.join(
            'dist',
            '%s-%s.tar.gz' % (self.distribution.get_name(),
                              self.distribution.get_version()))
        # run sdist if needed
        if not os.path.isfile(dist):
            self.run_command('sdist')
        # get checksum digests
        log.info('reading distribution tarball %r' % dist)
        with open(dist, 'rb') as fobj:
            data = fobj.read()
        log.info('recovered digests:')
        digest = dict()
        digest['rmd160'] = self._get_rmd160(dist)
        for algo in [1, 256]:
            digest['sha%d' % algo] = self._get_sha(data, algo)
        for key, val in digest.iteritems():
            log.info('    %s: %s' % (key, val))
        # write finished portfile to file
        with open(self.portfile, 'w') as fport:
            fport.write(self._template.render(
                version=self.distribution.get_version(), **digest))
        log.info('portfile written to %r' % self.portfile)

    @staticmethod
    def _get_sha(data, algorithm=256):
        hash_ = getattr(hashlib, 'sha%d' % algorithm)
        return hash_(data).hexdigest()

    @staticmethod
    def _get_rmd160(filename):
        p = subprocess.Popen(['openssl', 'rmd160', filename],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(err)
        else:
            return out.splitlines()[0].rsplit(' ', 1)[-1]

cmdclass['port'] = BuildPortfile


# -- don't use setup_requires if just checking for information ----------------

# (credit: matplotlib/setup.py)
setup_requires = []
if '--help' not in sys.argv and '--help-commands' not in sys.argv:
    dist_ = Distribution({'cmdclass': cmdclass})
    dist_.parse_config_files()
    dist_.parse_command_line()
    if not (any('--' + opt in sys.argv for opt in
            Distribution.display_option_names + ['help']) or
            dist_.commands == ['clean']):
        setup_requires = ['tornado', 'numpy >= 1.7', 'jinja2', 'gitpython']

# -- find files ---------------------------------------------------------------

# Use the find_packages tool to locate all packages and modules
packagenames = find_packages()

# glob for all scripts
if os.path.isdir('bin'):
    scripts = glob.glob(os.path.join('bin', '*'))
else:
    scripts = []

# -- run setup ----------------------------------------------------------------

setup(name=PACKAGENAME,
      provides=[PACKAGENAME],
      version=__version__,
      description="A python package for gravitational-wave astrophysics",
      long_description="""
          GWpy is a collaboration-driven `Python <http://www.python.org>`_
          package providing tools for studying data from ground-based
          gravitational-wave detectors.
      """,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url='https://gwpy.github.io/',
      packages=packagenames,
      include_package_data=True,
      cmdclass=cmdclass,
      scripts=scripts,
      setup_requires=setup_requires,
      requires=[
          'glue',
          'dateutil',
          'numpy',
          'scipy',
          'matplotlib',
          'astropy',
          'six',
      ],
      install_requires=[
          'python-dateutil',
          'numpy >= 1.7',
          'scipy >= 0.16.0',
          'matplotlib >= 1.3.0',
          'astropy >= 1.0',
          'six >= 1.5',
      ] + extra_install_requires,
      tests_require=[
      ] + extra_tests_require,
      extras_require={
          'nds': ['nds2-client'],
          'gwf': ['ldas-tools'],
          'doc': ['sphinx', 'numpydoc', 'sphinx-bootstrap-theme',
                  'sphinxcontrib-doxylink', 'sphinxcontrib-epydoc',
                  'sphinxcontrib-programoutput'],
          'hdf5': ['h5py'],
      },
      dependency_links=[
          'https://www.lsc-group.phys.uwm.edu/daswg/download/'
          'software/source/glue-1.48.tar.gz#egg=glue-1.48',
      ],
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
