# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
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

"""GWDataFind UI for FFL cache files.

This module is used to replace the proper GWDataFind interface
on-the-fly when FFL data access is inferred.
As such this module is required to emulate those functions
from `gwdatafind` used in :mod:`gwpy.io.datafind`.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import os
import re
from warnings import warn
from functools import lru_cache

from ligo.segments import (
    segment,
    segmentlist,
)

from .cache import (
    _iter_cache,
    cache_segments,
    read_cache_entry,
)

_SITE_REGEX = re.compile(r"\A(\w+)-")
_DEFAULT_TYPE_MATCH = re.compile(r"^(?!lastfile|spectro|\.).*")


# -- generic utilities ------

def _read_last_line(path, bufsize=2, encoding="utf-8"):
    """Read the last line of a file.
    """
    with open(path, "rb") as fobj:
        # go to end of file
        fobj.seek(-bufsize, os.SEEK_END)

        # rewind until we hit a line break
        while fobj.read(1) != b"\n":
            try:
                fobj.seek(-bufsize, os.SEEK_CUR)
            except OSError:
                # if we've rewound to the start of the file, just stop
                if fobj.tell() < bufsize:
                    fobj.seek(0)
                    break
                # otherwise this is a different error
                raise

        # read the current line
        return fobj.readline().rstrip().decode(encoding)


# -- ffl utilities ----------

def _get_ffl_basedir():
    """Return the base directory in which to find FFL files

    Raises
    ------
    KeyError
        If neither the ``FFLPATH`` or ``VIRGODATA`` environment variables
        are set.
    """
    if 'FFLPATH' in os.environ:
        return os.environ['FFLPATH']
    if 'VIRGODATA' in os.environ:
        return os.path.join(os.environ['VIRGODATA'], 'ffl')
    raise KeyError(
        "failed to parse FFLPATH from environment, please set "
        "FFLPATH to point to the directory containing FFL files",
    )


def _is_ffl_file(path):
    """Return `True` if this file looks (naively) like an FFL file.
    """
    return str(path).endswith(".ffl")


def _get_site_tag(path):
    """Return the ``(site, tag)`` for a given FFL file.
    """
    # tag is just name of file minus extension
    tag = os.path.splitext(os.path.basename(path))[0]

    # need to read first file from FFL to get site (IFO)
    last = _read_last_line(path).split()[0]
    site = _SITE_REGEX.match(os.path.basename(last)).groups()[0]

    return site, tag


def _find_ffl_files(basedir=None):
    """Find all FFL files under a given base directory.
    """
    for root, _, files in os.walk(basedir or _get_ffl_basedir()):
        for name in filter(_is_ffl_file, files):
            yield os.path.join(root, name)


@lru_cache()
def _find_ffls(basedir=None):
    """Find all readable FFL files.
    """
    ffls = {}
    for path in _find_ffl_files(basedir=basedir):
        try:
            ffls[_get_site_tag(path)] = path
        except (
            OSError,  # file is empty (or cannot be read at all)
            AttributeError,  # last entry didn't match _SITE_REGEX
        ):
            continue
    return ffls


def _ffl_path(site, tag, basedir=None):
    """Return the path of the FFL file for a given site and tag.
    """
    try:
        return _find_ffls(basedir=basedir)[(site, tag)]
    except KeyError:
        raise ValueError(
            f"no FFL file found for ('{site}', '{tag}')",
        )


@lru_cache()
def _read_ffl(site, tag, basedir=None):
    """Read an FFL file as a list of `CacheEntry` objects
    """
    ffl = _ffl_path(site, tag, basedir=basedir)
    with open(ffl, "r") as fobj:
        return [
            type(entry)(site, tag, entry.segment, entry.path)
            for entry in _iter_cache(fobj, gpstype=float)
        ]


def _handle_error(action, message):
    """Handle error, warn, or ignore for the given state.

    Parameters
    ----------
    action : `str`
        The action to perform, one of
        ``'warn'`` (emit a `UserWarning`, default),
        ``'ignore'`` (do nothing),
        or anything else (raise a `RuntimeError`).

    message : `str`
        The message to emit with warnings or errors.

    Raises
    ------
    RuntimeError
        If action is not ``'warn'`` or ``'ignore'``.
    """
    # if ignore, do nothing
    if action == "ignore":
        return
    # if warn, emit a warning
    if action == "warn":
        warn(message, stacklevel=2)
        return

    # otherwise, raise an error
    raise RuntimeError(message)


# -- ui ---------------------

def find_types(site=None, match=_DEFAULT_TYPE_MATCH):
    """Return the list of known data types.

    Parameters
    ----------
    site : `str`, optional
        Observatory ID (e.g. ``'A'`)) to retrict types, if `None` (default)
        is given, all types are returned.

    match : `str`, optional
        Regular expression to use to restrict types.

    Returns
    -------
    types : `list` of `str`
        The list of data types matching the criteria.
    """
    ffls = _find_ffls()
    types = [tag for (site_, tag) in ffls if site in (None, site_)]
    if match is not None:
        match = re.compile(match)
        return list(filter(match.search, types))
    return types


def find_urls(
    site,
    tag,
    gpsstart,
    gpsend,
    match=None,
    on_gaps="warn",
):
    """Return the list of all files of the given type in the [start, end)
    GPS interval.

    Parameters
    ----------
    site : `str`
        Observatory ID to search for.

    tag : `str`
        Data type tag to search for.

    gpsstart : `int`
        GPS start time of query.

    gpsend : `int`
        GPS end time of query.

    match : `str`, optional
        Regular expression to use to retrict returned data URLs.

    on_gaps : `str`, optional
        What to do if the full GPS interval is not covered, one of
        ``'warn'`` (emit a `UserWarning`, default),
        ``'ignore'`` (do nothing),
        anything else (raise a `RuntimeError`).

    Returns
    -------
    urls : `list` of `str`
        A list of URLs representing discovered data.
    """
    if match:
        match = re.compile(match)

    span = segment(gpsstart, gpsend)

    cache = [
        e for e in _read_ffl(site, tag) if (
            e.observatory == site
            and e.description == tag
            and e.segment.intersects(span)
            and (match.search(e.path) if match else True)
        )
    ]
    urls = [e.path for e in cache]

    # handle missing data
    missing = segmentlist([span]) - cache_segments(cache)
    if missing:
        _handle_error(
            on_gaps,
            "Missing segments: \n" + "\n".join(map(str, missing)),
        )

    return urls


def find_latest(site, tag, on_missing="warn"):
    """Return the most recent file of a given type.

    Parameters
    ----------
    site : `str`
        Observatory ID to search for.

    tag : `str`
        Data type tag to search for.

    on_missing : `str`, optional
        What to do if a URL is not found for the given site and tag, one of
        ``'warn'`` (emit a `UserWarning`, default),
        ``'ignore'`` (do nothing),
        anything else (raise a `RuntimeError`).

    Returns
    -------
    urls : `list` of `str`
        A list (typically of one item) of URLs representing the latest data
        for a specific site and tag.
    """
    try:
        fflfile = _ffl_path(site, tag)
    except ValueError:  # no readable FFL file
        urls = []
    else:
        urls = [read_cache_entry(_read_last_line(fflfile), gpstype=float)]

    if not urls:
        _handle_error(on_missing, "No files found")

    return urls
