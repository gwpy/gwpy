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

"""Sphinx extension for an inline autosummary-style class description.
"""

import os

from sphinx import package_dir
from sphinx.ext.autodoc import (ALL, Documenter, ClassDocumenter)
from sphinx.ext.autosummary import get_documenter
from sphinx.jinja2glue import BuiltinTemplateLoader
from sphinx.util.inspect import safe_getattr

from numpydoc.docscrape_sphinx import SphinxClassDoc

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class GWpyClassDocumenter(ClassDocumenter):
    """Sub-class of `ClassDocumenter` to use autosummary template.

    This is specifically for the GWpy documentation. Any other uses
    might not get the desired results, mainly because I haven't
    coded up support for the standard ``autoclass`` arguments, e.g.
    ``:members:``.
    """
    objtype = 'class'

    def add_directive_header(self, sig):
        if self.doc_as_attr:
            self.directivetype = 'attribute'
        Documenter.add_directive_header(self, sig)

        # add inheritance info, if wanted
        if not self.doc_as_attr and self.options.show_inheritance:
            self.add_line(u'', '<autodoc>')
            if (hasattr(self.object, '__bases__') and
                    len(self.object.__bases__)):
                bases = []
                for b in self.object.__bases__:
                    if b.__module__ == '__builtin__':
                        bases.append(u':class:`%s`' % b.__name__)
                    elif b.__module__.startswith('gwpy.'):
                        bases.append(u':class:`%s.%s`'
                                     % (b.__module__.rsplit('.', 1)[0],
                                        b.__name__))
                    else:
                        bases.append(u':class:`%s.%s`'
                                     % (b.__module__, b.__name__))
                self.add_line(u'   Bases: %s' % ', '.join(bases),
                              '<autodoc>')

    def add_content(self, more_content, no_docstring=False):
        if self.doc_as_attr:
            super(GWpyClassDocumenter, self).add_content(
                more_content, no_docstring=no_docstring)
        else:
            name = safe_getattr(self.object, '__name__', None)
            if name:
                # create our own templating environment
                builder = self.env.app.builder or None
                template_dirs = [os.path.join(package_dir, 'ext',
                                              'autosummary', 'templates')]
                if builder is not None:
                    if builder.config.templates_path:
                        template_dirs = (builder.config.templates_path +
                                         template_dirs)
                    # allow the user to override the templates
                    template_loader = BuiltinTemplateLoader()
                    template_loader.init(builder, dirs=template_dirs)
                else:
                    template_loader = FileSystemLoader(template_dirs)
                template_env = SandboxedEnvironment(loader=template_loader)
                template = template_env.get_template('autoclass/class.rst')

                def get_members(obj, typ, include_public=[]):
                    items = []
                    want_all = self.options.inherited_members or \
                               self.options.members is ALL
                    members = zip(*self.get_object_members(want_all)[1])[0]
                    if self.options.exclude_members:
                        members = [m for m in members if
                                   m not in self.options.exclude_members]
                    for name in members:
                        try:
                            documenter = get_documenter(safe_getattr(obj, name),
                                                        obj)
                        except AttributeError:
                            continue
                        if documenter.objtype == typ:
                            items.append(name)
                    public = [x for x in items
                              if x in include_public or not x.startswith('_')]
                    return public, items

                ns = {}
                config = self.env.app.config
                npconfig = dict(
                    use_plots=config.numpydoc_use_plots,
                    show_class_members=config.numpydoc_show_class_members)
                ns['docstring'] = SphinxClassDoc(self.object, config=npconfig)

                ns['members'] = vars(self.object)
                ns['methods'], ns['all_methods'] = get_members(self.object,
                                                               'method',
                                                               ['__init__'])
                ns['attributes'], ns['all_attributes'] = get_members(
                    self.object, 'attribute')

                parts = self.fullname.split('.')
                mod_name, obj_name = '.'.join(parts[:-1]), parts[-1]

                ns['fullname'] = name
                ns['module'] = mod_name
                ns['objname'] = obj_name
                ns['name'] = parts[-1]

                for line in template.render(**ns).split('\n'):
                    if line not in [None, 'None']:
                        self.add_line(line, '<autodoc>')
                self.doc_as_attr = True


def setup(app):
    app.add_autodocumenter(GWpyClassDocumenter)
