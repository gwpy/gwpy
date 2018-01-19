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

"""Construct HTML maps on top of images.
"""

from six.moves import zip

import numpy

from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..types import Series

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

JQUERY_URL = '//ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js'

HTML_HEADER = """<!doctype html>
<html lang="en">
<head>
    <title>{title}</title>
    <script src="{jquery}"></script>
</head>
<body>
"""
HTML_FOOTER = "</body>\n</html>"

HTML_MAP = """
<img src="{png}" usemap="#{map}" width="{width}" height="{height}"/>
<map name="{map}">
    {data}
</map>
<script>
    var id_ = 'tmptooltip';
    $('area').mouseenter(function() {{
        var position = $(this).attr('coords').split(',');
        x = +position[0];
        y = +position[1];
        var $popup = $('<div />').appendTo('body');
        $popup.attr('id', id_);
        $popup.addClass('gwpy-tooltip');
        $popup.html('<p>'+$(this).attr('alt')+'</p>');
        $popup.css({{position: 'absolute', top: y, left: x+20}});
    }});
    $('area').mouseleave(function() {{
        $('#'+id_).remove();
    }});
</script>
"""


def html_area(x, y, href='#', alt=None, shape='circle', **kwargs):
    """Format the HTML <area /> tag for this (x, y) <map> element

    Parameters
    ----------
    x : `int`
        x pixel coordinate
    y : `int`
        y pixel coordinate
    href : `str`, optional
        URL for onclick
    alt : `str`, optional
        hover popup information, defaults to '(x, y)'
    shape : `str`, optional
        shape of <area />
    **kwargs
        any other attr="value" pairs to append to <area /> tag

    Returns
    -------
    area : `str`
        formatted line of HTML defining <area />
    """
    if alt is None:
        alt = '(%s, %s)' % (x, y)
    out = ('<area shape="{shape}" coords="{x:d},{y:d},5" href="{href}" '
           'alt="{alt}" '.format(shape=shape, x=x, y=y, href=href, alt=alt))
    for attr, value in kwargs.items():
        out += '%s="%s" ' % (attr, value)
    out += '/>'
    return out


def map_artist(artist, filename, mapname='points', shape='circle',
               popup=None, title=None, standalone=True, jquery=JQUERY_URL):
    """Construct an HTML <map> to annotate the given `artist`

    Parameters
    ----------
    artist : `~matplotlib.artist.Artist`
        the plotted object (as returned from
        :meth:`~matplotlib.axes.Axes.plot`, for example)
    filename : `str`
        path to image file on disk
    mapname : `str`, optional
        ID to connect <img> tag and <map> tags, default: ``'points'``. This
        should be unique if multiple maps are to be written to a single HTML
        file.
    shape : `str`, optional
        shape for <area> tag, default: ``'circle'``
    popup : `function`, `iterable`, optional
        content for alt tooltip popup, either a function that can be
        called with the (x, y) pixel coords of each data point, or an
        iterable with one `str` text per data point
    standalone : `bool`, optional
        wrap map HTML with required HTML5 header and footer tags,
        default: `True`
    title : `str`, optional
        title name for standalone HTML page
    jquery : `str`, optional
        URL of jquery script

    Returns
    -------
    html : `str`
        formatted HTML page

    Notes
    -----
    This method should be run *after* the top-level figure is saved to disk,
    so that the relevant DPI and figure size information is fixed.
    """
    axes = artist.axes

    # get artist data
    if isinstance(artist, Collection):
        data = artist.get_offsets()
        z = artist.get_array()
        if z is not None:
            data = numpy.vstack((data[:, 0], data[:, 1], z)).T
    elif isinstance(artist, Line2D):
        data = numpy.asarray(artist.get_data()).T
    else:
        data = numpy.asarray(artist.get_data())

    return _map(data, axes, filename, mapname=mapname, shape=shape,
                popup=popup, title=title, standalone=standalone, jquery=jquery)


def map_data(data, axes, filename, mapname='points', shape='circle',
             popup=None, title=None, standalone=True, jquery=JQUERY_URL):
    """Construct an HTML <map> to annotate the given `artist`

    Parameters
    ----------
    data : `~gwpy.data.Series`, `~numpy.ndarray`
        data to map
    axes : `~matplotlib.axes.Axes`
        `Axes` to map onto
    filename : `str`
        path to image file on disk
    mapname : `str`, optional
        ID to connect <img> tag and <map> tags, default: ``'points'``. This
        should be unique if multiple maps are to be written to a single HTML
        file.
    shape : `str`, optional
        shape for <area> tag, default: ``'circle'``
    popup : `function`, `iterable`, optional
        content for alt tooltip popup, either a function that can be
        called with the (x, y) pixel coords of each data point, or an
        iterable with one `str` text per data point
    standalone : `bool`, optional
        wrap map HTML with required HTML5 header and footer tags,
        default: `True`
    title : `str`, optional
        title name for standalone HTML page
    jquery : `str`, optional
        URL of jquery script

    Returns
    -------
    html : `str`
        formatted HTML page

    Notes
    -----
    This method should be run *after* the top-level figure is saved to disk,
    so that the relevant DPI and figure size information is fixed.
    """
    if isinstance(data, Series):
        data = numpy.vstack((data.xindex.value, data.value)).T
    else:
        data = numpy.asarray(data)

    if isinstance(axes, Figure):
        axes = axes.gca()

    return _map(data, axes, filename, mapname=mapname, title=title,
                popup=popup, shape=shape, standalone=standalone, jquery=jquery)


def _map(data, axes, filename, href='#', mapname='points', popup=None,
         title='', shape='circle', standalone=True, jquery=JQUERY_URL):
    """Build an HTML <map> for a data set on some axes.
    """
    fig = axes.figure
    transform = axes.transData

    # get 2-d pixels
    pixels = numpy.round(transform.transform(data[:, :2])).astype(int)

    # get figure size
    dpi = fig.dpi
    width = int(fig.get_figwidth() * dpi)
    height = int(fig.get_figheight() * dpi)

    npix = pixels.shape[0]
    # configure hrefs
    if isinstance(href, str):
        hrefs = [href] * npix
    elif callable(href):
        hrefs = map(href, data)
    else:
        hrefs = numpy.asarray(href)
    if not len(hrefs) == npix:
        raise ValueError("href map with %d elements doesn't match %d data "
                         "points" % (hrefs.shape[0], npix))

    # build map
    areas = []
    for i, (datum, pixel, href) in enumerate(
            zip(data[::-1], pixels[::-1], hrefs[::-1])):
        if callable(popup):
            alt = popup(*datum)
        elif popup is None:
            alt = '(%s)' % (', '.join(map(str, datum)))
        else:
            alt = popup[-i-1]
        x = pixel[0]
        y = height - pixel[1]
        areas.append(html_area(x, y, href=href, alt=alt, shape=shape))

    hmap = HTML_MAP.format(title=title, png=filename, map=mapname, width=width,
                           height=height, data='\n    '.join(areas))
    if standalone:
        return (HTML_HEADER.format(title=title, jquery=jquery) + hmap +
                HTML_FOOTER)
    return hmap
