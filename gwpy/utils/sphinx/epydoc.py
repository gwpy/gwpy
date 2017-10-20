# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, Sebastian Wiesner <lunaryorn@googlemail.com>
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
    sphinxcontrib.epydoc
    ====================

    Sphinx extension to cross-reference epydoc generated documentation

    .. moduleauthor::  Sebastian Wiesner  <lunaryorn@googlemail.com>
"""


from __future__ import (print_function, division, unicode_literals,
                        absolute_import)

__version__ = '0.6'

import re
import posixpath

from docutils import nodes


def filename_for_object(objtype, name):
    if objtype == 'exception':
        # exceptions are classes for epydoc
        objtype = 'class'
    if not (objtype == 'module' or objtype == 'class'):
        try:
            name, attribute = name.rsplit('.', 1)
        except ValueError:
            anchor = ''
        else:
            anchor = '#{0}'.format(attribute)
        if objtype == 'function' or objtype == 'data':
            objtype = 'module'
        else:
            objtype = 'class'
    else:
        anchor = ''
    return '{0}-{1}.html{2}'.format(name, objtype, anchor)


def resolve_reference_to_epydoc(app, env, node, contnode):
    """
    Resolve a reference to an epydoc documentation.
    """
    domain = node.get('refdomain')
    if domain != 'py':
        # epydoc only holds Python docs
        return

    target = node['reftarget']

    mapping = app.config.epydoc_mapping
    matching_baseurls = (baseurl for baseurl in mapping if
                         any(re.match(p, target) for p in mapping[baseurl]))
    baseurl = next(matching_baseurls, None)
    if not baseurl:
        return

    objtype = env.domains[domain].objtypes_for_role(node['reftype'])[0]
    fn = filename_for_object(objtype, target)
    uri = posixpath.join(baseurl, fn)

    newobjtype = posixpath.splitext(fn)[0].rsplit('-', 1)[-1]
    title = '{} {} at {}'.format(target, newobjtype, baseurl)

    newnode = nodes.reference('', '')
    newnode['class'] = 'external-xref'
    newnode['refuri'] = uri
    newnode['reftitle'] = title
    newnode.append(contnode)
    return newnode


def setup(app):
    app.require_sphinx('1.0')
    app.add_config_value('epydoc_mapping', {}, 'env')
    app.connect(str('missing-reference'), resolve_reference_to_epydoc)
