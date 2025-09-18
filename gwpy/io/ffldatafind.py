# Copyright (c) 2022-2025 Cardiff University
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

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from igwn_segments import (
    segment,
    segmentlist,
)

from .cache import (
    _CacheEntry,
    _iter_cache,
    cache_segments,
    file_segment,
    read_cache_entry,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from .utils import FileSystemPath

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

logger = logging.getLogger(__name__)

_SITE_REGEX = re.compile(r"\A(\w+)-")
_DEFAULT_TYPE_MATCH = re.compile(r"^(?!lastfile|spectro|\.).*")


# -- generic utilities ---------------

@cache
def _read_last_line(
    path: FileSystemPath,
    bufsize: int = 2,
    encoding: str = "utf-8",
) -> str:
    """Read and return the last line of a text file."""
    with open(path, "rb") as fobj:  # noqa: PTH123
        # start from the end and read until we find a full line, or reach the beginning
        fobj.seek(0, os.SEEK_END)
        count = 0
        try:
            while fobj.read(1) != b"\n":
                fobj.seek(-bufsize, os.SEEK_CUR)
                count += 1
        except OSError:
            if not count:  # file is empty or unreadable
                raise
            # if we've rewound to the start of the file, just stop
            if fobj.tell() < bufsize:
                fobj.seek(0)
            # otherwise this is a different error
            else:
                raise
        return fobj.read().decode(encoding).strip()


# -- ffl utilities -------------------

def _get_ffl_basedir() -> Path:
    """Return the base directory containing FFL cache files."""
    # FFLPATH
    try:
        fflpath = Path(os.environ["FFLPATH"])
    except KeyError:
        pass
    else:
        logger.debug("Parsed FFLPATH='%s'", fflpath)
        return fflpath

    # VIRGODATA
    try:
        virgodata = Path(os.environ["VIRGODATA"])
    except KeyError:
        msg = (
            "failed to parse FFLPATH from environment, please set "
            "FFLPATH to point to the directory containing FFL files"
        )
        raise KeyError(msg) from None
    else:
        logger.debug("Parsed VIRGODATA='%s'", virgodata)
        return virgodata / "ffl"


def _is_ffl_file(path: FileSystemPath) -> bool:
    """Return True if the given path looks like an FFL file path."""
    return Path(path).suffix == ".ffl"


def _get_site_tag(path: FileSystemPath) -> tuple[str, str]:
    """Return the (site, tag) tuple given the path to an FFL file."""
    path = Path(path)
    tag = path.stem
    last = _read_last_line(path).split()[0]
    match = _SITE_REGEX.match(Path(last).name)
    if match is None:
        msg = f"Site tag not found in last line of FFL file: {last}"
        raise ValueError(msg)
    site = match.groups()[0]
    return site, tag


def _find_ffl_files(
    basedir: FileSystemPath | None = None,
) -> Generator[str, None, None]:
    """Yield all FFL file paths from the given base directory."""
    if basedir is None:
        basedir = _get_ffl_basedir()
    basedir = Path(basedir)
    for root, _, names in os.walk(basedir):
        for name in filter(_is_ffl_file, names):
            yield str(Path(root) / name)


@cache
def _find_ffls(basedir: str | Path | None = None) -> dict[tuple[str, str], list[str]]:
    """Find all readable FFL files."""
    ffls = defaultdict(list)
    for path in _find_ffl_files(basedir=basedir):
        try:
            site, tag = _get_site_tag(path)
        except (
            OSError,  # file is empty (or cannot be read at all)
            ValueError,  # last entry didn't match _SITE_REGEX
        ):
            continue
        logger.debug("Found FFL for %s-%s: '%s'", site, tag, path)
        ffls[(site, tag)].append(path)
    return ffls


def _ffl_paths(
    site: str,
    tag: str,
    basedir: str | Path | None = None,
) -> list[str]:
    """Return the paths of all FFL files for a given site and tag."""
    try:
        return _find_ffls(basedir=basedir)[(site, tag)]
    except KeyError as exc:
        msg = f"no FFL file found for ('{site}', '{tag}')"
        raise ValueError(msg) from exc


@cache
def _read_ffls(
    site: str,
    tag: str,
    basedir: str | Path | None = None,
) -> list[_CacheEntry]:
    """Read all FFL files for a given site and tag as a list of CacheEntry objects.

    Parameters
    ----------
    site : `str`
        Observatory ID to search for.
    tag : `str`
        Data type tag to search for.
    basedir : `str`, `Path`, or `None`, optional
        Base directory to search in. If `None`, uses default FFL directory.

    Returns
    -------
    entries : `list` of `_CacheEntry`
        List of cache entries found in the FFL files.
    """
    entries: list[_CacheEntry] = []
    for ffl in _ffl_paths(site, tag, basedir=basedir):
        with Path(ffl).open() as fobj:
            entries.extend(
                _CacheEntry(site, tag, entry.segment, entry.path)
                for entry in _iter_cache(fobj, gpstype=float)
            )
    return entries


def _handle_error(action: str, message: str) -> None:
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
    msg = message
    raise RuntimeError(msg)


# -- ui ---------------------

def _check_gwdatafind_kwargs(
    urltype: str,
    ext: str | None,
) -> None:
    """Validate the arguments that are supported for compatibility with GWDataFind."""
    if urltype != "file":
        msg = "Data discovery via FFL only supports urltype='file'"
        raise ValueError(msg)
    if ext not in {"gwf", None}:
        msg = "Data discovery via FFL only supports ext='gwf'"
        raise ValueError(msg)


def find_types(
    site: str | None = None,
    match: str | re.Pattern | None = _DEFAULT_TYPE_MATCH,
    ext: str | None = "gwf",
) -> list[str]:
    """Return the list of known data types.

    Parameters
    ----------
    site : `str`, optional
        Observatory ID (e.g. ``'A'``) to restrict types, if `None` (default)
        is given, all types are returned.

    match : `str`, optional
        Regular expression to use to restrict types.

    ext : `str`, optional
        The file extension for which to search.

    Returns
    -------
    types : `list` of `str`
        The list of data types matching the criteria.
    """
    _check_gwdatafind_kwargs("file", ext)
    ffls = _find_ffls()
    types = [tag for (site_, tag) in ffls if site in (None, site_)]
    if match is not None:
        match = re.compile(match)
        return list(filter(match.search, types))
    return types


def find_urls(
    site: str,
    tag: str,
    gpsstart: int,
    gpsend: int,
    match: str | re.Pattern | None = None,
    on_gaps: str = "warn",
    urltype: str = "file",
    ext: str | None = "gwf",
) -> list[str]:
    """Return the list of all files for ``tag`` in the ``[start, end)`` GPS interval.

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
        Regular expression to use to restrict returned data URLs.

    on_gaps : `str`, optional
        Action to take when the requested all or some of the GPS interval
        is not covereed by the dataset, one of:

        - ``'error'``: raise a `RuntimeError` (default)
        - ``'warn'``: print a warning but return all available URLs
        - ``'ignore'``: return the list of URLs with no warnings

    urltype : `str`, optional
        Type of URL for which to search. Only ``"file"`` is acceptable.

    ext : `str`, optional
        The file extension for which to search.

    Returns
    -------
    urls : `list` of `str`
        A list of URLs representing discovered data.
    """
    _check_gwdatafind_kwargs(urltype, ext)

    if match:
        match = re.compile(match)

    span = segment(gpsstart, gpsend)

    def _keep(e: _CacheEntry) -> bool:
        """Return `True` if this `_CacheEntry` is to be kept."""
        return bool(
            e.observatory == site
            and e.path.endswith(f".{ext}")
            and e.description == tag
            and e.segment.intersects(span)
            and (match.search(e.path) if match else True),
        )

    cache = list(filter(_keep, _read_ffls(site, tag)))
    urls = [e.path for e in cache]

    # handle missing data
    missing = segmentlist([span]) - cache_segments(cache)
    if missing:
        msg = "Missing segments: \n" + "\n".join(map(str, missing))
        _handle_error(on_gaps, msg)

    return urls


def find_latest(
    site: str,
    tag: str,
    urltype: str = "file",
    ext: str | None = "gwf",
    on_missing: str = "warn",
) -> list[str]:
    """Return the most recent file of a given type.

    Parameters
    ----------
    site : `str`
        Observatory ID to search for.

    tag : `str`
        Data type tag to search for.

    urltype : `str`, optional
        Type of URL for which to search. Only ``"file"`` is acceptable.

    ext : `str`, optional
        The file extension for which to search.

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
    _check_gwdatafind_kwargs(urltype, ext)

    try:
        fflfiles = _ffl_paths(site, tag)
    except ValueError:  # no readable FFL file
        urls = []
    else:
        urls = [
            read_cache_entry(_read_last_line(fflfile), gpstype=float)
            for fflfile in fflfiles
        ]
        if urls:  # if multiple, find the latest one
            urls = sorted(urls, key=file_segment)[-1:]

    if not urls:
        _handle_error(on_missing, "No files found")

    return urls
