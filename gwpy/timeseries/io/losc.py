# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Read and write HDF5 files in GWOSC format.

For more details, see :ref:`gwpy-table-io`.
"""

from __future__ import annotations

import logging
import re
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from astropy.units import Quantity
from astropy.utils.data import get_readable_fileobj
from gwosc.locate import get_urls

from ...detector.units import parse_unit
from ...io import (
    gwf as io_gwf,
    hdf5 as io_hdf5,
)
from ...io.cache import (
    file_segment,
    sieve as sieve_cache,
)
from ...io.utils import file_path
from ...segments import Segment
from ...time import to_gps
from ...utils.env import bool_env
from .. import StateVector, TimeSeries

# Module logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from contextlib import AbstractContextManager
    from typing import (
        IO,
        BinaryIO,
    )

    import h5py

    from ...detector import Channel
    from ...typing import GpsLike
    from .. import TimeSeriesBase

DQMASK_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*DQMASK\Z")
STRAIN_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*STRAIN\Z")

GWOSC_LOCATE_KWARGS = (
    "sample_rate",
    "version",
    "host",
    "format",
    "dataset",
)


# -- utilities -----------------------

def _download_file(
    url: str,
    *,
    cache: bool | None = None,
    verbose: bool = False,
    timeout: float | None = None,
    **kwargs,
) -> AbstractContextManager[BinaryIO]:
    """Download a file with optional caching."""
    if cache is None:
        cache = bool_env("GWPY_CACHE", default=False)
    return get_readable_fileobj(
        url,
        cache=cache,
        show_progress=verbose,
        remote_timeout=timeout,
        **kwargs,
    )


def _get_file_extension(url: str) -> str:
    """Get the file extension from a URL, handling compressed files."""
    if url.endswith(".gz"):
        path = Path(url[:-3])
    else:
        path = Path(url)
    return path.suffix


def _set_format_from_extension(
    ext: str,
    kwargs: dict[str, str],
) -> None:
    """Set format in kwargs based on file extension."""
    if ext == ".hdf5":
        kwargs.setdefault("format", "hdf5.gwosc")
    elif ext == ".txt":
        kwargs.setdefault("format", "ascii.gwosc")
    elif ext == ".gwf":
        kwargs.setdefault("format", "gwf")


def _parse_bits_from_gwf_unit(series: StateVector) -> None:
    """Parse bit definitions from GWF unit string."""
    try:
        bits = {}
        for bit in str(series.unit).split():
            a, b = bit.split(":", 1)
            bits[int(a)] = b
        series.bits = bits
        series.override_unit("")
    except (TypeError, ValueError):
        # Don't care, bad GWOSC format
        pass


def _fetch_gwosc_data_file(
    url: str,
    *args: str | None,
    cls: type[TimeSeriesBase] = TimeSeries,
    cache: bool | None = None,
    verbose: bool = False,
    timeout: float | None = None,
    **kwargs,
) -> TimeSeriesBase:
    """Fetch a single GWOSC file and return a `Series`."""
    # Match file format
    ext = _get_file_extension(url)
    _set_format_from_extension(ext, kwargs)

    with _download_file(url, cache=cache, verbose=verbose, timeout=timeout) as rem:
        # Get channel for GWF if not given
        if ext == ".gwf" and (not args or args[0] is None):
            args = (_gwf_channel(rem, cls, verbose=verbose),)

        if verbose:
            logger.info("Reading data...")

        try:
            series = cls.read(rem, *args, **kwargs)
        except Exception as exc:
            if verbose:
                logger.exception("Failed to read data")
            exc.args = (f"Failed to read GWOSC data from {url!r}: {exc}",)
            raise
        else:
            # Parse bits from unit in GWF
            if ext == ".gwf" and isinstance(series, StateVector):
                _parse_bits_from_gwf_unit(series)

            if verbose:
                logger.info("Done reading data")
            return series


def _overlapping(files: Iterable[str]) -> bool:
    """Quick method to see if a file list contains overlapping files."""
    segments: set[Segment] = set()
    for path in files:
        seg = file_segment(path)
        for s in segments:
            if seg.intersects(s):
                return True
        segments.add(seg)
    return False


def _name_from_gwosc_hdf5(f: h5py.HLObject, path: str) -> str:
    """Forge a name from a path in a GWOSC HDF5 file.

    We want to be as close as possible to the GWF channel name.

    New files (starting at O2 circa 2016-2017) contain a dataset called
    GWOSCmeta (in the path) that stores the GWF channel (without the ifo name)
    so we use this to reconstruct the GWF channel.
    It works for strain, DQ and injections.

    For old files, we return the path.
    """
    try:
        # New files store the channel name in GWOSCmeta
        meta_ds = io_hdf5.find_dataset(f, f"{path}/GWOSCmeta")
    except KeyError:
        # GWOSCmeta isn't stored in old files
        return path
    channel = meta_ds[()].decode("utf-8")
    # We can then find the observatory
    # This is just the letter code, not the number so we assume 1
    ifo_ds = io_hdf5.find_dataset(f, "meta/Observatory")
    ifo = ifo_ds[()].decode("utf-8")
    return f"{ifo}1:{channel}"


# -- remote data access (the main event)

def fetch_gwosc_data(
    detector: str,
    start: GpsLike,
    end: GpsLike,
    cls: type[TimeSeriesBase] = TimeSeries,
    **kwargs,
) -> TimeSeriesBase:
    """Fetch GWOSC data for a given detector.

    This function is for internal purposes only, all users should instead
    use the interface provided by `TimeSeries.fetch_open_data` (and similar
    for `StateVector.fetch_open_data`).
    """
    # format arguments
    start = to_gps(start)
    end = to_gps(end)
    span: Segment[float] = Segment(start, end)

    # find URLs (requires python-gwosc)
    url_kw = {key: kwargs.pop(key) for key in GWOSC_LOCATE_KWARGS if key in kwargs}
    if "sample_rate" in url_kw:  # format as Hertz
        url_kw["sample_rate"] = Quantity(url_kw["sample_rate"], "Hz").value
    urls = get_urls(
        detector,
        int(start),
        ceil(end),
        **url_kw,
    )
    cache = sieve_cache(urls, segment=span)

    # if event dataset, pick shortest file that covers the request
    # -- this is a bit hacky, and presumes that only an event dataset
    # -- would be produced with overlapping files.
    # -- This should probably be improved to use dataset information
    if len(cache) and _overlapping(cache):
        cache.sort(key=lambda x: abs(file_segment(x)))
        for url in cache:
            a, b = file_segment(url)
            if a <= start and b >= end:
                cache = [url]
                break
    if kwargs.get("verbose", False):  # get_urls() guarantees len(cache) >= 1
        host = urlparse(cache[0]).netloc
        logger.info(
            "Fetched %d URLs from %s for [%s .. %s])",
            len(cache),
            host,
            start,
            ceil(end),
        )

    is_gwf = cache[0].endswith(".gwf")
    args: tuple[str | Channel | None, ...]
    if is_gwf and cache:
        args = (kwargs.pop("channel", None),)
    else:
        args = ()

    # read data
    out = None
    kwargs["cls"] = cls
    for url in cache:
        keep = file_segment(url) & span
        kwargs["start"], kwargs["end"] = keep
        new = _fetch_gwosc_data_file(url, *args, **kwargs)
        if is_gwf and (not args or args[0] is None):
            args = (new.name,)
        if out is None:
            out = new.copy()
        else:
            out.append(new, resize=True)
    return out


# -- I/O -----------------------------

@io_hdf5.with_read_hdf5
def read_gwosc_hdf5(
    h5f: str | h5py.HLObject,
    path: str = "strain/Strain",
    start: GpsLike | None = None,
    end: GpsLike | None = None,
    *,
    copy: bool = False,
) -> TimeSeries:
    """Read a `TimeSeries` from a GWOSC-format HDF file.

    Parameters
    ----------
    h5f : `str`, `h5py.HLObject`
        path of HDF5 file, or open `H5File`

    path : `str`
        name of HDF5 dataset to read.

    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data

    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data

    copy : `bool`, default: `False`
        create a fresh-memory copy of the underlying array

    Returns
    -------
    data : `~gwpy.timeseries.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    dataset = io_hdf5.find_dataset(h5f, path)
    # read data
    nddata = dataset[()]
    # read metadata
    xunit = parse_unit(dataset.attrs["Xunits"])
    epoch = dataset.attrs["Xstart"]
    dt = Quantity(dataset.attrs["Xspacing"], xunit)
    unit = dataset.attrs["Yunits"]
    # build and return
    return TimeSeries(
        nddata,
        epoch=epoch,
        sample_rate=(1/dt).to("Hertz"),
        unit=unit,
        name=path.rsplit("/", 1)[1],
        copy=copy,
    ).crop(start=start, end=end)


@io_hdf5.with_read_hdf5
def read_gwosc_hdf5_state(
    f: str | h5py.HLObject,
    path: str = "quality/simple",
    start: GpsLike | None = None,
    end: GpsLike | None = None,
    *,
    copy: bool = False,
    value_dataset: str = "DQmask",
    bits_dataset: str = "DQDescriptions",
) -> StateVector:
    """Read a `StateVector` from a GWOSC-format HDF file.

    Parameters
    ----------
    f : `str`, `h5py.HLObject`
        path of HDF5 file, or open `H5File`

    path : `str`
        path of HDF5 datasets to read (will be used as name of the dataset).

    start : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        start GPS time of desired data

    end : `Time`, `~gwpy.time.LIGOTimeGPS`, optional
        end GPS time of desired data

    copy : `bool`, default: `False`
        create a fresh-memory copy of the underlying array

    value_dataset : `str`
        HDF5 dataset where to read the statevector values

    bits_dataset : `str`
        HDF5 dataset where to read the definition of each bits

    Returns
    -------
    data : `~gwpy.timeseries.StateVector`
        a new `StateVector` containing the data read from disk
    """
    # find data
    bits_ds = io_hdf5.find_dataset(f, f"{path}/{value_dataset}")
    def_ds = io_hdf5.find_dataset(f, f"{path}/{bits_dataset}")
    # read data
    bits = bits_ds[()]
    bit_def = [bytes.decode(bytes(b), "utf-8") for b in def_ds[()]]
    # read metadata
    epoch = bits_ds.attrs["Xstart"]
    try:
        dt = bits_ds.attrs["Xspacing"]
    except KeyError:
        dt = Quantity(1, "s")
    else:
        xunit = parse_unit(bits_ds.attrs["Xunits"])
        dt = Quantity(dt, xunit)
    # Name
    name = _name_from_gwosc_hdf5(f, path)
    return StateVector(bits, bits=bit_def, t0=epoch, name=name,
                       dx=dt, copy=copy).crop(start=start, end=end)


def _gwf_channel(
    source: IO | str,
    series_class: type[TimeSeriesBase] = TimeSeries,
    *,
    verbose: bool = False,
) -> str:
    """Find the right channel name for a GWOSC GWF file."""
    channels = list(io_gwf.iter_channel_names(file_path(source)))
    if issubclass(series_class, StateVector):
        regex = DQMASK_CHANNEL_REGEX
    else:
        regex = STRAIN_CHANNEL_REGEX
    found, = list(filter(regex.match, channels))
    if verbose:
        logger.debug("Using channel %r", found)
    return found


# register
TimeSeries.read.registry.register_reader(
    "hdf5.gwosc",
    TimeSeries,
    read_gwosc_hdf5,
)
StateVector.read.registry.register_reader(
    "hdf5.gwosc",
    StateVector,
    read_gwosc_hdf5_state,
)
