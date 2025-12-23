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
import warnings
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from itertools import product
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from astropy.units import Quantity
from gwosc.api import DEFAULT_URL as DEFAULT_GWOSC_URL
from gwosc.locate import get_urls

from ...detector import Channel
from ...detector.units import parse_unit
from ...io import (
    gwf as io_gwf,
    hdf5 as io_hdf5,
)
from ...io.cache import (
    file_segment,
    sieve as sieve_cache,
)
from ...io.remote import open_remote_file
from ...io.utils import file_path
from ...segments import Segment
from ...time import to_gps
from .. import (
    StateVector,
    TimeSeries,
    TimeSeriesBase,
    TimeSeriesBaseDict,
)

# Module logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Iterable,
    )
    from typing import (
        IO,
        Any,
        TypeVar,
    )

    import h5py

    from ...time import SupportsToGps

    T = TypeVar("T", bound=TimeSeriesBase)


KHZ_4 = 4096
KHZ_16 = 16384

DQMASK_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*DQMASK\Z")
STRAIN_CHANNEL_REGEX = re.compile(r"\A[A-Z]\d:(GW|L)OSC-.*STRAIN\Z")

GWOSC_HDF5_FORMAT = "hdf5.gwosc"
GWOSC_LOCATE_KWARGS = (
    "sample_rate",
    "version",
    "host",
    "format",
    "dataset",
)

# Keyword arguments that we may inherit from TimeSeries.get
# that are unsafe to use in to fetch_gwosc_data
IGNORE_GET_KWARGS = {
    "frametype",
    "frametype_match",
    "urltype",
}

NUM_THREADS = min(4, cpu_count() or 1)


# -- utilities -----------------------

def _is_gwosc_channel(name: str | Channel) -> bool:
    """Check if a channel name looks like a GWOSC channel."""
    return ":GWOSC-" in str(name)


def _all_gwosc_channels(channels: Iterable[str | Channel]) -> bool:
    """Check if all channel names look like GWOSC channels."""
    return all(map(_is_gwosc_channel, channels))


def _any_gwosc_channels(channels: Iterable[str | Channel]) -> bool:
    """Check if any channel names look like GWOSC channels."""
    return any(map(_is_gwosc_channel, channels))


def _get_file_extension(url: str) -> str:
    """Get the file extension from a URL, handling compressed files."""
    if url.endswith(".gz"):
        path = Path(url[:-3])
    else:
        path = Path(url)
    return path.suffix


def _default_format(ext: str) -> str | None:
    """Set format in kwargs based on file extension."""
    if ext == ".hdf5":
        return GWOSC_HDF5_FORMAT
    if ext == ".gwf":
        return "gwf"
    return None


def _gwf_channel(
    source: IO | str,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> str:
    """Find the right channel name for a GWOSC GWF file."""
    channels = list(io_gwf.iter_channel_names(file_path(source)))
    if issubclass(series_class, StateVector):
        regex = DQMASK_CHANNEL_REGEX
    else:
        regex = STRAIN_CHANNEL_REGEX
    found, = list(filter(regex.match, channels))
    logger.debug("Using channel %r", found)
    return found


def _parse_bits_from_gwf_unit(series: StateVector) -> None:
    """Parse bit definitions from GWF unit string."""
    try:
        bits: dict[int, str | None] = {}
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
    series_class: type[TimeSeriesBase] = TimeSeries,
    cache: bool | None = None,
    timeout: float | None = None,
    format: str | None = None,  # noqa: A002
    **kwargs,
) -> TimeSeriesBase:
    """Fetch a single GWOSC file and return a `Series`."""
    # Match file format
    ext = _get_file_extension(url)
    fmt = format or _default_format(ext)

    logger.debug("Downloading GWOSC data from %s", url)
    with open_remote_file(
        url,
        cache=cache,
        remote_timeout=timeout,
    ) as rem:
        # Get channel for GWF if not given
        if ext == ".gwf" and (not args or args[0] is None):
            args = (_gwf_channel(rem, series_class),)

        logger.debug("Reading %s", url.rsplit("/", maxsplit=1)[-1])

        try:
            series = series_class.read(rem, *args, format=fmt, **kwargs)
        except Exception as exc:
            exc.args = (f"Failed to read GWOSC data from {url!r}: {exc}",)
            raise
        else:
            # Parse bits from unit in GWF
            if ext == ".gwf" and isinstance(series, StateVector):
                _parse_bits_from_gwf_unit(series)
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


def _name_from_gwosc_hdf5(
    h5ds: h5py.Dataset,
    name: str = "GWOSCmeta",
) -> str:
    """Construct a channel-like name for a dataset in a GWOSC HDF5 file.

    New files (starting at O2 circa 2016-2017) contain a dataset called
    ``GWOSCmeta`` alongside the main dataset that stores the GWF channel
    (without the ifo name) so we use this to reconstruct the GWF channel.
    It works for strain, DQ and injections.

    For old files, we return the name of the dataset prefixed with the
    detector prefix from the ``/meta/Detector`` dataset.
    """
    h5g = h5ds.parent

    # New files store the channel name in GWOSCmeta
    try:
        meta_ds = h5g[name]
    except KeyError:  # no GWOSCmeta
        channel = h5ds.name.rsplit("/", maxsplit=1)[-1]
    else:
        channel = meta_ds[()].decode("utf-8")
    # We can then find the detector prefix
    ifo_ds = io_hdf5.find_dataset(h5g.file, "meta/Detector")
    ifo = ifo_ds[()].decode("utf-8")
    return f"{ifo}:{channel}"


# -- remote data access (the main event)

def fetch_gwosc_data(
    detector: str,
    start: SupportsToGps,
    end: SupportsToGps,
    dataset: str | None = None,
    version: int | None = None,
    sample_rate: int = 4096,
    format: str = "hdf5",
    host: str = DEFAULT_GWOSC_URL,
    series_class: type[T] = TimeSeries,
    *,
    verbose: bool | None = None,
    **kwargs,
) -> T:
    """Fetch open-access data from GWOSC.

    Parameters
    ----------
    detector : `str`
        The two-character prefix of the IFO in which you are interested,
        e.g. `'L1'`.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    sample_rate : `float`, optional,
        The sample rate of desired data; most data are stored
        by GWOSC at 4096 Hz, however there may be event-related
        data releases with a 16384 Hz rate, default: `4096`.

    version : `int`, optional
        Version of files to download, defaults to highest discovered
        version.

    format : `str`, optional
        The data format to download and parse, default: ``'h5py'``

        - ``'hdf5'``
        - ``'gwf'`` - requires `lalframe`

    host : `str`, optional
        HTTP host name of GWOSC server to access.

    cache : `bool`, optional
        Save/read a local copy of the remote URL, default: `False`;
        useful if the same remote data are to be accessed multiple times.
        Set ``GWPY_CACHE=1`` in the environment to auto-cache.

    timeout : `float`, optional
        The time to wait for a response from the GWOSC server.

    verbose : `bool`, optional
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    kwargs
        Any other keyword arguments are passed to the `TimeSeries.read`
        method that parses the file that was downloaded.

    Examples
    --------
    >>> from gwpy.timeseries import (TimeSeries, StateVector)
    >>> print(TimeSeries.get('H1', 1126259446, 1126259478))
    TimeSeries([  2.17704028e-19,  2.08763900e-19,  2.39681183e-19,
                ...,   3.55365541e-20,  6.33533516e-20,
                  7.58121195e-20]
               unit: Unit(dimensionless),
               t0: 1126259446.0 s,
               dt: 0.000244140625 s,
               name: Strain,
               channel: None)
    >>> print(StateVector.get('H1', 1126259446, 1126259478))
    StateVector([127,127,127,127,127,127,127,127,127,127,127,127,
                 127,127,127,127,127,127,127,127,127,127,127,127,
                 127,127,127,127,127,127,127,127]
                unit: Unit(dimensionless),
                t0: 1126259446.0 s,
                dt: 1.0 s,
                name: quality/simple,
                channel: None,
                bits: Bits(0: data present
                           1: passes cbc CAT1 test
                           2: passes cbc CAT2 test
                           3: passes cbc CAT3 test
                           4: passes burst CAT1 test
                           5: passes burst CAT2 test
                           6: passes burst CAT3 test,
                           channel=None,
                           epoch=1126259446.0))

    For the `StateVector`, the naming of the bits will be
    ``format``-dependent, because they are recorded differently by GWOSC
    in different formats.

    Notes
    -----
    `StateVector` data are not available in ``txt.gz`` format.
    """
    if verbose is not None:
        warnings.warn(
            "The 'verbose' argument is deprecated and will be removed in a future "
            "release, please consider using DEBUG-level logging instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # strip out arguments for other formats
    for key in IGNORE_GET_KWARGS:
        kwargs.pop(key, None)

    # format arguments
    start = to_gps(start)
    end = to_gps(end)
    span: Segment[float] = Segment(start, end)

    # find URLs (requires python-gwosc)
    sample_rate = int(Quantity(sample_rate, "Hz").value)
    urls = get_urls(
        detector,
        int(start),
        ceil(end),
        dataset=dataset,
        version=version,
        sample_rate=sample_rate,
        format=format,
        host=host,
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
    logger.debug(
        "Fetched %d URLs from %s for [%s .. %s])",
        len(cache),
        urlparse(cache[0]).netloc,
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
    kwargs["series_class"] = series_class
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
    if out is None:
        msg = f"No data found for {detector} in [{start} .. {end})"
        raise ValueError(msg)
    return out


def fetch_dict(
    detectors: Collection[str | Channel],
    start: SupportsToGps,
    end: SupportsToGps,
    dataset: str | None = None,
    version: int | None = None,
    sample_rate: int = 4096,
    format: str = "hdf5",
    host: str = DEFAULT_GWOSC_URL,
    parallel: int = NUM_THREADS,
    series_class: type[T] = TimeSeries,
    **kwargs,
) -> dict[str | Channel, T]:
    """Fetch open-access data from GWOSC for multiple detectors.

    Parameters
    ----------
    detectors : `list` of `str`
        List of two-character prefices of the IFOs in which you
        are interested, e.g. `['H1', 'L1']`.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine.

    sample_rate : `float`, `Quantity`,
        The sample rate (Hertz) of desired data; most data are stored
        by GWOSC at 4096 Hz, however there may be event-related
        data releases with a 16384 Hz rate.

    version : `int`
        Version of files to download, defaults to highest discovered
        version.

    format : `str`
        The data format to download and parse.
        One of

        "hdf5"
            HDF5 data files, read using `h5py`.

        "gwf"
            Gravitational-Wave Frame files, requires |LDAStools.frameCPP|_.

    host : `str`
        Host name of GWOSC server to access.

    cache : `bool`
        Save/read a local copy of the remote URL, default: `False`;
        useful if the same remote data are to be accessed multiple times.
        Set `GWPY_CACHE=1` in the environment to auto-cache.

    parallel : `int`
        Number of parallel threads to use when downloading data for
        multiple detectors. Default is ``1``.

    kwargs
        Any other keyword arguments are passed to the `TimeSeries.read`
        method that parses the file that was downloaded.

    See Also
    --------
    TimeSeries.fetch_open_data
        For more examples.

    TimeSeries.read
        For details of how files are read.

    Examples
    --------
    >>> from gwpy.timeseries import TimeSeriesDict
    >>> print(TimeSeriesDict.get(['H1', 'L1'], 1126259446, 1126259478))
    TimeSeriesDict({'H1': <TimeSeries([2.17704028e-19, 2.08763900e-19, 2.39681183e-19, ...,
                 3.55365541e-20, 6.33533516e-20, 7.58121195e-20]
                unit=Unit(dimensionless),
                t0=<Quantity 1.12625945e+09 s>,
                dt=<Quantity 0.00024414 s>,
                name='Strain',
                channel=None)>, 'L1': <TimeSeries([-1.04289994e-18, -1.03586274e-18, -9.89322445e-19,
                 ..., -1.01767748e-18, -9.82876816e-19,
                 -9.59276974e-19]
                unit=Unit(dimensionless),
                t0=<Quantity 1.12625945e+09 s>,
                dt=<Quantity 0.00024414 s>,
                name='Strain',
                channel=None)>})
    """  # noqa: E501
    names = {str(x).split(":", maxsplit=1)[0]: x for x in detectors}
    parallel = min(len(detectors), parallel or 1)

    def _fod(ifo: str) -> tuple[str, TimeSeriesBase]:
        """Fetch data for a single detector."""
        return ifo, fetch_gwosc_data(
            ifo,
            start,
            end,
            dataset=dataset,
            version=version,
            sample_rate=sample_rate,
            format=format,
            host=host,
            parallel=1,
            series_class=series_class,
            **kwargs,
        )

    # fetch all data in a thread pool
    out = series_class.DictClass()
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(_fod, ifo) for ifo in names]
        for future in as_completed(futures):
            ifo, data = future.result()
            out[name := names[ifo]] = data
            logger.debug("Fetched data for %s", name)

    return out


# -- I/O -----------------------------

@io_hdf5.with_read_hdf5
def read_gwosc_hdf5(
    h5f: str | h5py.HLObject,
    path: str = "strain/Strain",
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
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
    name = _name_from_gwosc_hdf5(dataset)
    return TimeSeries(
        nddata,
        epoch=epoch,
        sample_rate=(1/dt).to("Hertz"),
        unit=unit,
        name=name,
        copy=copy,
    ).crop(start=start, end=end)


@io_hdf5.with_read_hdf5
def read_gwosc_hdf5_state(
    f: str | h5py.HLObject,
    path: str = "quality/simple",
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
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
    name = _name_from_gwosc_hdf5(bits_ds)
    return StateVector(
        bits,
        bits=bit_def,
        t0=epoch,
        name=name,
        dx=dt,
        copy=copy,
    ).crop(start=start, end=end)


TimeSeries.read.registry.register_reader(
    GWOSC_HDF5_FORMAT,
    TimeSeries,
    read_gwosc_hdf5,
)
StateVector.read.registry.register_reader(
    GWOSC_HDF5_FORMAT,
    StateVector,
    read_gwosc_hdf5_state,
)


# -- TimeSeries.get integration ------

def _as_tuple(
    value: str | int | Iterable[str | int] | None,
    default: Iterable[str | int],
) -> tuple[str | int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str | int):
        return (value,)
    return tuple(value)


def identify_gwosc_sources(
    origin: str,
    channels: str | Channel | Iterable[str | Channel],
    *args,  # noqa: ARG001, ANN002
    host: str | None = None,
    format: str | Iterable[str] | None = None,  # noqa: A002
    sample_rate: int | Iterable[int] | None = None,
    **kwargs,  # noqa: ARG001
) -> Iterable[dict[str, Any]] | None:
    """Identify GWOSC sources for these arguments."""
    if origin != "get":
        return None

    sources: list[dict[str, object]] = []

    # If host was given, and looks like nds2/gwdatafind, stop
    if str(host).startswith((
        "datafind",
        "gwdatafind",
        "nds",
    )):
        return sources

    channels = (channels,) if isinstance(channels, str | Channel) else channels
    names = {str(c) for c in channels}

    # Set priority (and format) based on channel names
    priority = 10
    if format is None and _all_gwosc_channels(names):
        format = "gwf"  # noqa: A001
        priority = 5  # user mentioned GWOSC, use it if we can
    elif any(":" in str(c) for c in channels):
        priority = 1000  # GWOSC almost certainly won't be able to help

    hosts = _as_tuple(host, (DEFAULT_GWOSC_URL,))
    formats = _as_tuple(format, ("hdf5", "gwf"))

    # Check if sample rate can be inferred from channel names
    if sample_rate is None:
        if all("4KHZ" in n for n in names):
            sample_rate = KHZ_4
        elif all("16KHZ" in n for n in names):
            sample_rate = KHZ_16
    sample_rates = _as_tuple(sample_rate, (KHZ_4, KHZ_16))

    for host_, format_, sample_rate_ in product(
        hosts,
        formats,
        sample_rates,
    ):
        sources.append({
            "host": host_,
            "format": format_,
            "sample_rate": sample_rate_,
            "priority": priority,
        })

    return sources


for klass, fetch in (
    (TimeSeriesBase, fetch_gwosc_data),
    (TimeSeriesBaseDict, fetch_dict),
):
    # register identifier
    klass.get.registry.register_identifier(
        "gwosc",
        klass,
        identify_gwosc_sources,
    )
    # register fetch
    klass.get.registry.register_reader(
        "gwosc",
        klass,
        fetch,
    )
