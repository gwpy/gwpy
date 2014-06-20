# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
#
# This file is part of GWpy
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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""Git version generator
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Adam Mercer <adam.mercer@ligo.org>'

import os
import time
from distutils.version import (Version, LooseVersion, StrictVersion)

from git import Repo
from jinja2 import Template


VERSION_PY_TEMPLATE = Template(
"""# -*- coding: utf-8 -*-
{% if 'author' in package %}\
# Copyright (C) {{ package['author'] }} ({{ package['year'] }})

# package metadata
__author__ = "{{ package['author'] }} <{{ package['email'] }}>"\
{% endif %}
__version__ = '{{ version.vstring }}'
__date__ = '{{ status.datestr }}'

# package version
version = '{{ version.vstring }}'
major = {{ version.version[0] }}
minor = {{ version.version[1] }}
micro = {{ version.version[2] }}
debug = {{ not version.vstring.replace('.', '').isdigit() }}
release = {{ version.version[0] > 0 and \
             version.vstring.replace('.', '').isdigit() }}

# repository version information
git_hash = '{{ status.commit.hexsha }}'
git_tag = '{{ status.tag and status.tag.name or None }}'
git_author = "{{ status.author }}"
git_committer = "{{ status.committer }}"
git_is_dirty = {{ status.is_dirty() }}
""")


class GitStatus(object):
    def __init__(self, path=os.curdir):
        self.repo = Repo(path=path)

    def is_dirty(self):
        if isinstance(self.repo.is_dirty, bool):
            return self.repo.is_dirty
        else:
            return self.repo.is_dirty()
    is_dirty.__doc__ = Repo.is_dirty.__doc__

    @property
    def commit(self):
        return self.branch.commit

    @property
    def branch(self):
        if isinstance(self.repo.active_branch, str):
            for branch in self.repo.branches:
                if branch.name == self.repo.active_branch:
                    return branch
            raise RuntimeError("Cannot resolve active branch.")
        else:
            return self.repo.active_branch

    @property
    def tag(self):
        for tag in self.repo.tags:
            if tag.commit == self.commit:
                return tag
        return None

    @property
    def revision(self):
        """The number of commits between the HEAD and the last tag.
        """
        try:
            tag = self.repo.tags[-1]
        except IndexError:
            start = ''
        else:
            start = '%s..' % tag.name
        return self.repo.git.rev_list('%sHEAD' % start).count('\n') + 1

    @property
    def version(self):
        if self.tag and not self.is_dirty():
            v = self.tag.name.strip('v')
        else:
            try:
                v = self.repo.tags[-1].name.strip('v')
            except IndexError:
                v = '0.0.0'
            nrev = self.revision
            if nrev:
                v += '.dev%d' % nrev
            elif self.is_dirty():
                v += '.dev'
        return LooseVersion(v)

    @property
    def date(self):
        t = self.commit.committed_date
        if isinstance(t, time.struct_time):
            return t
        else:
            return time.gmtime(t)

    @property
    def datestr(self):
        return time.strftime('%Y-%m-%d %H:%M:%S +0000', self.date)

    @property
    def author(self):
        return "%s <%s>" % (self.commit.author.name, self.commit.author.email)

    @property
    def committer(self):
        return "%s <%s>" % (self.commit.committer.name,
                            self.commit.committer.email)

    # ------------------------------------------------------------------------
    # Write

    def write(self, fobj, version=None, **package_metadata):
        """Write the contents of this `GitStatus` to a file object.
        """
        if version is None:
            version = self.version
        if not isinstance(version, Version):
            version = LooseVersion(str(version))
        if len(version.version) < 3 or not isinstance(version.version[2], int):
            version.version.insert(2, 'None')
        package_metadata.setdefault('year', time.gmtime().tm_year)
        fobj.write(VERSION_PY_TEMPLATE.render(status=self, version=version,
                                              package=package_metadata))

