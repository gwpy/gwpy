# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2019)
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
import os
import re
import subprocess
import shutil
import sys
import tempfile
from distutils.cmd import Command
from distutils.command.clean import (clean as orig_clean, log, remove_tree)
from distutils.command.bdist_rpm import bdist_rpm as distutils_bdist_rpm
from distutils.errors import DistutilsArgError
from distutils.version import (LooseVersion, StrictVersion)
from itertools import groupby

from setuptools.command.bdist_rpm import bdist_rpm as _bdist_rpm
from setuptools.command.sdist import sdist as _sdist

import versioneer

CMDCLASS = versioneer.get_cmdclass()
SETUP_REQUIRES = {
    'test': ['pytest_runner'],
}
COPYRIGHT_REGEX = re.compile(r"Copyright[\S ]+(?P<years>\d\d\d\d([, \d-]+)?)")

# -- documentation builder ----------------------------------------------------

SETUP_REQUIRES['build_sphinx'] = [  # list should match requirements-doc.txt
    'sphinx >= 1.6.1',
    'numpydoc >= 0.8.0',
    'sphinx-bootstrap-theme >= 0.6',
    'sphinxcontrib-programoutput',
    'sphinx-automodapi',
    'requests',
]
if {'build_sphinx'}.intersection(sys.argv):
    try:
        from sphinx.setup_command import BuildDoc
    except ImportError as exc:
        exc.msg = 'build_sphinx command requires {0}'.format(
            SETUP_REQUIRES['build_sphinx'][0].replace(' ', ''))
        exc.args = (exc.msg,)
        raise
    CMDCLASS['build_sphinx'] = BuildDoc


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
        git_version = gitv.strip().split()[2]

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
            self.start_tag = self._tag_version(self.start_tag)

    def format_changelog_entry(self, tag):
        import git
        log.debug('    parsing changelog entry for {}'.format(tag))
        if isinstance(tag, git.Tag):
            tago = tag.tag
            date = datetime.datetime.fromtimestamp(tago.tagged_date)
            tz = tago.tagger_tz_offset
            version = tag.name.strip('v')
            author = tago.tagger.name
            email = tago.tagger.email
            message = tago.message.split('\n')[0]
            build = 1
        else:
            repo = git.Repo()
            commit = repo.head.commit
            date = commit.authored_datetime
            tz = commit.author_tz_offset
            version = str(tag)
            author = commit.author.name
            email = commit.author.email
            message = 'Test build'
            build = 1000
        name = self.distribution.get_name()
        if self.format == 'rpm':
            formatter = self._format_entry_rpm
        elif self.format == 'deb':
            formatter = self._format_entry_deb
        else:
            raise RuntimeError("unsupported changelog format")
        return formatter(name, version, build, author, email, message,
                         date, tz)

    @staticmethod
    def _format_entry_rpm(name, version, build, author, email, message,
                          date, tzoffset):
        return (
            "* {} {} <{}> - {}-{}\n"
            "- {}\n".format(
                date.strftime('%a %b %d %Y'),
                author,
                email,
                version,
                build,
                message,
            )
        )

    @staticmethod
    def _format_entry_deb(name, version, build, author, email, message,
                          date, tzoffset):
        return (
            "{} ({}-{}) unstable; urgency=low\n\n"
            "  * {}\n\n"
            " -- {} <{}>  {} {:+05d}\n".format(
                name,
                version,
                build,
                message,
                author,
                email,
                date.strftime('%a, %d %b %Y %H:%M:%S'),
                int(-tzoffset / 3600. * 100),
            )
        )

    @staticmethod
    def _tag_version(tag):
        return StrictVersion(str(tag).lstrip('v'))

    def get_git_tags(self):
        import git
        repo = git.Repo()
        tags = repo.tags
        tags.sort(key=self._tag_version, reverse=True)
        log.debug('found {} git tags'.format(len(tags)))
        return tags

    def is_tag(self):
        import git
        repo = git.Repo()
        return repo.git.describe() in repo.tags

    def run(self):
        log.info('creating changelog')
        lines = []
        tags = self.get_git_tags()
        if not self.is_tag():
            version = self._tag_version(tags[0]).version
            devversion = '{0}.{1}.{2}-dev'.format(version[0], version[1],
                                                  version[2])
            lines.append(self.format_changelog_entry(devversion))
        for tag in tags:
            if self.start_tag and self._tag_version(tag) < self.start_tag:
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
        orig_clean.run(self)


CMDCLASS['clean'] = clean


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


def _parse_years(years):
    """Parse string of ints include ranges into a `list` of `int`

    Source: https://stackoverflow.com/a/6405228/1307974
    """
    result = []
    for part in years.split(','):
        if '-' in part:
            a, b = part.split('-')
            a, b = int(a), int(b)
            result.extend(range(a, b + 1))
        else:
            a = int(part)
            result.append(a)
    return result


def _format_years(years):
    """Format a list of ints into a string including ranges

    Source: https://stackoverflow.com/a/9471386/1307974
    """
    def sub(x):
        return x[1] - x[0]

    ranges = []
    for k, iterable in groupby(enumerate(sorted(years)), sub):
        rng = list(iterable)
        if len(rng) == 1:
            s = str(rng[0][1])
        else:
            s = "{}-{}".format(rng[0][1], rng[-1][1])
        ranges.append(s)
    return ", ".join(ranges)


def update_copyright(path, year):
    """Update a file's copyright statement to include the given year
    """
    with open(path, "r") as fobj:
        text = fobj.read().rstrip()
    match = COPYRIGHT_REGEX.search(text)
    x = match.start("years")
    y = match.end("years")
    if text[y-1] == " ":  # don't strip trailing whitespace
        y -= 1
    yearstr = match.group("years")
    years = set(_parse_years(yearstr)) | {year}
    with open(path, "w") as fobj:
        print(text[:x] + _format_years(years) + text[y:], file=fobj)


def update_all_copyright(year):
    files = subprocess.check_output([
        "git", "grep", "-l", "-E", r"(\#|\*) Copyright",
    ]).strip().splitlines()
    ignore = {
        "gwpy/utils/sphinx/epydoc.py",
        "docs/_static/js/copybutton.js",
    }
    for path in files:
        if path.decode() in ignore:
            continue
        try:
            update_copyright(path, year)
        except AttributeError:
            raise RuntimeError(
                "failed to update copyright for {!r}".format(path),
            )
