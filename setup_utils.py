# -*- coding: utf-8 -*-
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

"""Packaging utilities for the GWpy package
"""

from __future__ import print_function

import contextlib
import datetime
import glob
import hashlib
import os
import subprocess
import shutil
import sys
import tempfile
from distutils.cmd import Command
from distutils.command.clean import (clean as orig_clean, log, remove_tree)
from distutils.command.bdist_rpm import bdist_rpm as distutils_bdist_rpm
from distutils.errors import DistutilsArgError
from distutils.version import LooseVersion

from setuptools.command.bdist_rpm import bdist_rpm as _bdist_rpm
from setuptools.command.sdist import sdist as _sdist

import versioneer

CMDCLASS = versioneer.get_cmdclass()
SETUP_REQUIRES = {
    'test': ['pytest_runner'],
}


# -- utilities ----------------------------------------------------------------

def in_git_clone():
    """Returns `True` if the current directory is a git repository

    Logic is 'borrowed' from :func:`git.repo.fun.is_git_dir`
    """
    gitdir = '.git'
    return os.path.isdir(gitdir) and (
        os.path.isdir(os.path.join(gitdir, 'objects')) and
        os.path.isdir(os.path.join(gitdir, 'refs')) and
        os.path.exists(os.path.join(gitdir, 'HEAD'))
    )


def reuse_dist_file(filename):
    """Returns `True` if a distribution file can be reused

    Otherwise it should be regenerated
    """
    # if target file doesn't exist, we must generate it
    if not os.path.isfile(filename):
        return False

    # if we can interact with git, we can regenerate it, so we may as well
    try:
        import git
    except ImportError:
        return True
    else:
        try:
            git.Repo().tags
        except (TypeError, git.GitError):
            return True
        else:
            return False


def get_gitpython_version():
    """Determine the required version of GitPython

    Because of target systems running very, very old versions of setuptools,
    we only specify the actual version we need when we need it.
    """
    # if not in git clone, it doesn't matter
    if not in_git_clone():
        return 'GitPython'

    # otherwise, call out to get the git version
    try:
        gitv = subprocess.check_output('git --version', shell=True)
    except (OSError, IOError, subprocess.CalledProcessError):
        # no git installation, most likely
        git_version = '0.0.0'
    else:
        if isinstance(gitv, bytes):
            gitv = gitv.decode('utf-8')
        git_version = gitv.rstrip().split()[-1]

    # if git>=2.15, we need GitPython>=2.1.8
    if LooseVersion(git_version) >= '2.15':
        return 'GitPython>=2.1.8'
    return 'GitPython'


@contextlib.contextmanager
def temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


# -- custom commands ----------------------------------------------------------

class changelog(Command):
    """Generate the changelog for this project
    """
    description = 'write the changelog for this project'
    user_options = [
        ('format=', 'f', 'output changelog format'),
        ('start-tag=', 's', 'git tag to start with'),
        ('output=', 'o', 'output file path'),
    ]
    FORMATS = ('rpm', 'deb')

    def initialize_options(self):
        self.format = 'rpm'
        self.start_tag = None
        self.output = 'changelog.txt'

    def finalize_options(self):
        if self.format not in self.FORMATS:
            raise DistutilsArgError(
                "--format should be one of {!r}".format(self.FORMATS))
        if self.start_tag:
            self.start_tag = self.start_tag.lstrip('v')

    def format_changelog_entry(self, tag):
        log.debug('    parsing changelog entry for {}'.format(tag))
        if self.format == 'rpm':
            return self._format_entry_rpm(tag)
        elif self.format == 'deb':
            return self._format_entry_deb(tag)
        raise RuntimeError("unsupported changelog format")

    def _format_entry_rpm(self, tag):
        tago = tag.tag
        date = datetime.date.fromtimestamp(tago.tagged_date)
        dstr = date.strftime('%a %b %d %Y')
        tagger = tago.tagger
        message = tago.message.split('\n')[0]
        return "* {} {} <{}>\n- {}\n".format(dstr, tagger.name, tagger.email,
                                             message)

    def _format_entry_deb(self, tag):
        tago = tag.tag
        date = datetime.datetime.fromtimestamp(tago.tagged_date)
        dstr = date.strftime('%a, %d %b %Y %H:%M:%S')
        tz = int(-tago.tagger_tz_offset / 3600. * 100)
        version = tag.name.strip('v')
        tagger = tago.tagger
        message = tago.message.split('\n')[0]
        name = self.distribution.get_name()
        return ("{} ({}-1) unstable; urgency=low\n\n"
                "  * {}\n\n"
                " -- {} <{}>  {} {:+05d}\n".format(
                    name, version, message,
                    tagger.name, tagger.email, dstr, tz))

    def get_git_tags(self):
        import git
        repo = git.Repo()
        tags = repo.tags
        tags.reverse()
        log.debug('found {} git tags'.format(len(tags)))
        return tags

    def run(self):
        log.info('creating changelog')
        lines = []
        for tag in self.get_git_tags():
            if self.start_tag and tag.name.lstrip('v') < self.start_tag:
                log.debug('reached start tag ({}), stopping'.format(
                    self.start_tag))
                break
            lines.append(self.format_changelog_entry(tag))
        log.info('writing changelog to {}'.format(self.output))
        with open(self.output, 'w') as f:
            for line in lines:
                print(line, file=f)


CMDCLASS['changelog'] = changelog
SETUP_REQUIRES['changelog'] = (get_gitpython_version(),)

orig_bdist_rpm = CMDCLASS.pop('bdist_rpm', _bdist_rpm)
DEFAULT_SPEC_TEMPLATE = os.path.join('etc', 'spec.template')


class bdist_rpm(orig_bdist_rpm):

    def run(self):
        if self.spec_only:
            return distutils_bdist_rpm.run(self)
        return orig_bdist_rpm.run(self)

    def _make_spec_file(self):
        # return already read specfile
        specfile = '{}.spec'.format(self.distribution.get_name())
        if reuse_dist_file(specfile):
            with open(specfile, 'rb') as specf:
                return specf.read()

        # generate changelog
        changelogcmd = self.distribution.get_command_obj('changelog')
        with tempfile.NamedTemporaryFile(delete=True, mode='w+') as f:
            self.distribution._set_command_options(changelogcmd, {
                'format': ('bdist_rpm', 'rpm'),
                'output': ('bdist_rpm', f.name),
            })
            self.run_command('changelog')
            f.seek(0)
            self.changelog = f.read()

        # read template
        from jinja2 import Template
        with open(DEFAULT_SPEC_TEMPLATE, 'r') as t:
            template = Template(t.read())

        # render specfile
        dist = self.distribution
        return template.render(
            name=dist.get_name(),
            version=dist.get_version(),
            description=dist.get_description(),
            long_description=dist.get_long_description(),
            url=dist.get_url(),
            license=dist.get_license(),
            changelog=self.changelog,
        ).splitlines()


CMDCLASS['bdist_rpm'] = bdist_rpm
SETUP_REQUIRES['bdist_rpm'] = SETUP_REQUIRES['changelog'] + ('jinja2',)

orig_sdist = CMDCLASS.pop('sdist', _sdist)


class sdist(orig_sdist):
    """Extension to sdist to build spec file and debian/changelog
    """
    def run(self):
        # generate spec file
        specfile = '{}.spec'.format(self.distribution.get_name())
        if not reuse_dist_file(specfile):
            self.distribution.have_run.pop('bdist_rpm', None)
            speccmd = self.distribution.get_command_obj('bdist_rpm')
            self.distribution._set_command_options(speccmd, {
                'spec_only': ('sdist', True),
            })
            self.run_command('bdist_rpm')
            shutil.move(os.path.join('dist', specfile), specfile)
            log.info('moved {} to {}'.format(
                os.path.join('dist', specfile), specfile))

        # generate debian/changelog
        debianchlog = os.path.join('debian', 'changelog')
        if not reuse_dist_file(debianchlog):
            self.distribution.have_run.pop('changelog')
            changelogcmd = self.distribution.get_command_obj('changelog')
            self.distribution._set_command_options(changelogcmd, {
                'format': ('sdist', 'deb'),
                'output': ('sdist', debianchlog),
            })
            self.run_command('changelog')

        orig_sdist.run(self)


CMDCLASS['sdist'] = sdist
SETUP_REQUIRES['sdist'] = (SETUP_REQUIRES['changelog'] +
                           SETUP_REQUIRES['bdist_rpm'])


class clean(orig_clean):
    """Custom clean command to remove more temporary files and directories
    """
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
            for egg in glob.glob('*.egg') + glob.glob('*.egg-info'):
                if os.path.isdir(egg):
                    remove_tree(egg, dry_run=self.dry_run)
                else:
                    log.info('removing %r' % egg)
                    os.unlink(egg)
            # remove extra files
            for filep in ('Portfile',):
                if os.path.exists(filep) and not self.dry_run:
                    log.info('removing %r' % filep)
                    os.unlink(filep)
        orig_clean.run(self)


CMDCLASS['clean'] = clean

DEFAULT_PORT_TEMPLATE = os.path.join('etc', 'Portfile.template')


class port(Command):
    """Generate a Macports Portfile for this project from the current build
    """
    description = 'generate a Macports Portfile'
    user_options = [
        ('tarball=', None, 'the distribution tarball to use'),
        ('version=', None, 'the X.Y.Z package version'),
        ('portfile=', None, 'target output file, default: \'Portfile\''),
        ('template=', None,
         'Portfile template, default: \'{}\''.format(DEFAULT_PORT_TEMPLATE)),
    ]

    def initialize_options(self):
        self.tarball = None
        self.version = None
        self.portfile = 'Portfile'
        self.template = DEFAULT_PORT_TEMPLATE
        self._template = None

    def finalize_options(self):
        from jinja2 import Template
        with open(self.template, 'r') as t:
            # pylint: disable=attribute-defined-outside-init
            self._template = Template(t.read())
        if self.version is None and self.tarball is not None:
            if self.tarball.endswith('.gz'):
                stub = os.path.splitext(self.tarball[:-3])[0]
            else:
                stub = os.path.splitext(self.tarball)[0]
            self.version = stub.rsplit('-', 1)[-1]
        elif self.version is None:
            self.version = self.distribution.get_version()

    def run(self):
        with temp_directory() as tmpd:
            # download dist file
            if self.tarball is None:
                self.tarball = self._download(self.distribution.get_name(),
                                              self.version, tmpd)

            # get checksum digests
            log.info('reading distribution tarball %r' % self.tarball)
            with open(self.tarball, 'rb') as fobj:
                data = fobj.read()
            log.info('recovered checksums:')
            checksum = dict()
            checksum['rmd160'] = self._get_rmd160(self.tarball)
            checksum['sha256'] = self._get_sha(data)
            checksum['size'] = os.path.getsize(self.tarball)
            for key, val in checksum.iteritems():
                log.info('    %s: %s' % (key, val))

            # write finished portfile to file
            with open(self.portfile, 'w') as fport:
                print(self._template.render(
                    version=self.version, **checksum),
                    file=fport)
            log.info('portfile written to %r' % self.portfile)

    @staticmethod
    def _download(name, version, targetdir):
        from pip.commands.download import DownloadCommand
        dcmd = DownloadCommand()
        rset = dcmd.run(*dcmd.parse_args([
            '{}=={}'.format(name, version),
            '--dest', targetdir, '--no-deps', '--no-binary', ':all:',
        ]))
        log.info('downloaded {}'.format(
            rset.requirements[name].link.url_without_fragment))
        return os.path.join(
            targetdir, rset.requirements[name].link.filename)

    @staticmethod
    def _get_sha(data, algorithm=256):
        hash_ = getattr(hashlib, 'sha%d' % algorithm)
        return hash_(data).hexdigest()

    @staticmethod
    def _get_rmd160(filename):
        out = subprocess.check_output(['openssl', 'rmd160', filename])
        return out.splitlines()[0].rsplit(' ', 1)[-1]


CMDCLASS['port'] = port
SETUP_REQUIRES['port'] = SETUP_REQUIRES['sdist'] + ('jinja2',)


# -- utility functions --------------------------------------------------------

def get_setup_requires():
    """Return the list of packages required for this setup.py run
    """
    # don't force requirements if just asking for help
    if {'--help', '--help-commands'}.intersection(sys.argv):
        return list()

    # otherwise collect all requirements for all known commands
    reqlist = []
    for cmd, dependencies in SETUP_REQUIRES.items():
        if cmd in sys.argv:
            reqlist.extend(dependencies)

    return reqlist


def get_scripts(scripts_dir='bin'):
    """Get relative file paths for all files under the ``scripts_dir``
    """
    scripts = []
    for (dirname, _, filenames) in os.walk(scripts_dir):
        scripts.extend([os.path.join(dirname, fn) for fn in filenames])
    return scripts
