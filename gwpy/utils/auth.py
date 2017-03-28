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

"""This module provides LIGO.ORG authenticated HTML queries
"""

import os
import stat
import warnings
import tempfile
import getpass
import socket

from six.moves import http_cookiejar
from six.moves.urllib import request as http_request
from six.moves.urllib.error import HTTPError

from glue.auth.saml import HTTPNegotiateAuthHandler

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

COOKIE_JAR = os.path.join(tempfile.gettempdir(), getpass.getuser())
LIGO_LOGIN_URL = 'login.ligo.org'


def request(url, debug=False, timeout=None):
    """Request the given URL using LIGO.ORG SAML authentication.

    This requires an active Kerberos ticket for the user, to get one:

    .. code-block:: bash

       kinit albert.einstein@LIGO.ORG

    Parameters
    ----------
    url : `str`
        URL path for request
    debug : `bool`, optional
        Query in verbose debugging mode, default `False`
    """
    # set debug to 1 to see all HTTP(s) traffic
    debug = int(debug)

    # need an instance of HTTPS handler to do HTTPS
    httpshandler = http_request.HTTPSHandler(debuglevel=debug)

    # use a cookie jar to store session cookies
    jar = http_cookiejar.LWPCookieJar()

    # if a cookie jar exists open it and read the cookies
    # and make sure it has the right permissions
    if os.path.exists(COOKIE_JAR):
        os.chmod(COOKIE_JAR, stat.S_IRUSR | stat.S_IWUSR)
        # set ignore_discard so that session cookies are preserved
        try:
            jar.load(COOKIE_JAR, ignore_discard=True)
        except http_cookiejar.LoadError as e:
            warnings.warn('LoadError caught: %s' % str(e))

    # create a cookie handler from the cookie jar
    cookiehandler = http_request.HTTPCookieProcessor(jar)
    # need a redirect handler to follow redirects
    redirecthandler = http_request.HTTPRedirectHandler()

    # need an auth handler that can do negotiation.
    # input parameter is the Kerberos service principal.
    auth_handler = HTTPNegotiateAuthHandler(
        service_principal='HTTP@%s' % LIGO_LOGIN_URL)

    # create the opener.
    opener = http_request.build_opener(auth_handler, cookiehandler,
                                       httpshandler, redirecthandler)

    # prepare the request object
    req = http_request.Request(url)

    # use the opener and the request object to make the request.
    if timeout is None:
        timeout = socket._GLOBAL_DEFAULT_TIMEOUT
    response = opener.open(req, timeout=timeout)

    # save the session cookies to a file so that they can
    # be used again without having to authenticate
    jar.save(COOKIE_JAR, ignore_discard=True)

    return response
