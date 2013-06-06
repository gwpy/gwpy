# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Extensions to the GLUE Cache and CacheEntry objects for reading
multiple frames of GW data
"""

from astropy.io import registry

from glue.lal import (LIGOTimeGPS, Cache, CacheEntry)
from glue import segments

from ... import version
from ...time import Time
from ...data import TimeSeries
from ...segments import (Segment, SegmentList)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def is_contiguous(cache):
    """Check whether this cache represents a single, contiguous
    stretch of data
    """
    segs = SegmentList([e.segment for e in cache]).coalesce()
    return len(segs) == 1


def read_cache(cache, channel, epoch=None, duration=None, verbose=False):
    """Read a TimeSeries from the list of files in the cache
    """
    # check contiguity
    if not is_contiguous(cache):
        raise ValueError("Cannot load TimeSeries from discontiguous data "
                         "cache")
    cache.sort(key=lambda e: e.segment[0])

    # set times
    if not epoch:
        epoch = cache[0].segment[0]
    epoch = LIGOTimeGPS(isinstance(epoch, Time) and epoch.gps or epoch)
    if not duration:
        end = LIGOTimeGPS(cache[-1].segment[-1])
        duration = float(end - epoch)
    end = epoch+duration
    span = segments.segment(epoch, end)
    cache = cache.sieve(segment=span)

    # return on empty
    if not len(cache):
        return TimeSeries([], channel=channel)
    # read first file
    out = TimeSeries.read(cache[0].path, channel, epoch=epoch,
                          duration=duration, format='gwf', verbose=verbose)
    # read other files
    for fp in cache[1:]:
        if span.disjoint(fp.segment):
            break
        frstart = max(epoch, fp.segment[0])
        frend = min(end, fp.segment[1])
        frdur = frend - frstart
        tmp = TimeSeries.read(fp.path, channel, format='gwf',
                              epoch=frstart, duration=frdur,
                              verbose=verbose)
        N = out.data.shape[0]
        out.data.resize((N + tmp.data.size,))
        out.data[N:] = tmp.data

    return out

def identify_gwf_cache(*args, **kwargs):
    cache = args[1][0]
    if isinstance(cache, Cache):
        return True
    else:
        return False

registry.register_reader("lcf", TimeSeries, read_cache, force=True)
registry.register_identifier("lcf", TimeSeries, identify_gwf_cache)
