# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module provides LIGO.ORG authenticated HTML queries
"""

import os
import sys
import stat
import urllib2
import cookielib

from glue.auth.saml import HTTPNegotiateAuthHandler

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

COOKIE_JAR = '/tmp/%s_cookies' % os.getenv('USER')
LIGO_LOGIN_URL = 'login.ligo.org'

def request(url, debug=False):
    """Request the given URL using LIGO.ORG SAML authentication.

    This requires an active Kerberos ticket for the user, to get one:

    >>> kinit albert.einstein@LIGO.ORG

    Parameters
    ----------
    url : `str`
        URL path for request
    debug : `bool`, optional
        Query in verbose debuggin mode, default `False`
    """
    # set debug to 1 to see all HTTP(s) traffic
    debug = int(debug)

    # need an instance of HTTPS handler to do HTTPS
    httpsHandler = urllib2.HTTPSHandler(debuglevel = debug)

    # use a cookie jar to store session cookies
    jar = cookielib.LWPCookieJar()

    # if a cookier jar exists open it and read the cookies
    # and make sure it has the right permissions
    if os.path.exists(COOKIE_JAR):
        os.chmod(COOKIE_JAR, stat.S_IRUSR | stat.S_IWUSR)

        # set ignore_discard so that session cookies are preserved
        jar.load(COOKIE_JAR, ignore_discard = True)

    # create a cookie handler from the cookier jar
    cookie_handler = urllib2.HTTPCookieProcessor(jar)
    # need a redirect handler to follow redirects
    redirectHandler = urllib2.HTTPRedirectHandler()

    # need an auth handler that can do negotiation.
    # input parameter is the Kerberos service principal.
    auth_handler = HTTPNegotiateAuthHandler(service_principal='HTTP@%s'
                                                            % (LIGO_LOGIN_URL))

    # create the opener.
    opener = urllib2.build_opener(auth_handler, cookie_handler, httpsHandler,
                                  redirectHandler)

    # prepare the request object
    request = urllib2.Request(url)

    # use the opener and the request object to make the request.
    response = opener.open(request)

    # save the session cookies to a file so that they can
    # be used again without having to authenticate
    jar.save(COOKIE_JAR, ignore_discard = True)

    return response

