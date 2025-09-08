# Stubs for the SWIG-generated `lalframe` module used by GWpy.

# ruff: noqa: N802, N803

import lal

# -- FrameU interface ----------------

class FrameUFrFile: ...

def FrameUFrFileOpen(filename: str, mode: str) -> FrameUFrFile: ...

class FrameUFrameH: ...

def FrameNew(
    epoch: lal.LIGOTimeGPS, duration: float, project: str,
    run: int, frnum: int, detectorFlags: int,
) -> FrameUFrameH: ...

def FrameUFrameHWrite(stream: FrameUFrFile, frame: FrameUFrameH) -> int: ...

class FrameUFrTOC: ...

def FrameUFrTOCRead(frfile: FrameUFrFile) -> FrameUFrTOC: ...
def FrameUFrTOCQueryAdcN(frtoc: FrameUFrTOC) -> int: ...
def FrameUFrTOCQueryAdcName(frtoc: FrameUFrTOC, adc: int) -> str: ...
def FrameUFrTOCQueryDetectorN(frtoc: FrameUFrTOC) -> int: ...
def FrameUFrTOCQueryDetectorName(frtoc: FrameUFrTOC, adc: int) -> str: ...
def FrameUFrTOCQueryDt(frtoc: FrameUFrTOC, pos: int) -> float: ...
def FrameUFrTOCQueryGTimeModf(frtoc: FrameUFrTOC, pos: int) -> tuple[int, float]: ...
def FrameUFrTOCQueryNFrame(frtoc: FrameUFrTOC) -> int: ...
def FrameUFrTOCQueryProcN(frtoc: FrameUFrTOC) -> int: ...
def FrameUFrTOCQueryProcName(frtoc: FrameUFrTOC, proc: int) -> str: ...
def FrameUFrTOCQuerySimN(frtoc: FrameUFrTOC) -> int: ...
def FrameUFrTOCQuerySimName(frtoc: FrameUFrTOC, sim: int) -> str: ...

class FrameUFrChan: ...

def FrameUFrChanQueryName(chan: FrameUFrChan) -> str: ...
def FrameUFrChanQueryTimeOffset(chan: FrameUFrChan) -> float: ...
def FrameUFrChanRead(stream: FrameUFrFile, name: str, pos: int) -> FrameUFrChan: ...

# -- Stream interface --------------

class FrFile: ...

def FrFileQueryChanType(frfile: FrFile, chname: str, pos: int) -> int: ...
def FrFileQueryChanVectorLength(frfile: FrFile, chname: str, pos: int) -> int: ...
def FrFileQueryDt(frfile: FrFile, pos: int) -> float: ...
def FrFileQueryGTime(start: lal.LIGOTimeGPS, frfile: FrFile, pos: int) -> int: ...
def FrFileQueryNFrame(frfile: FrFile) -> int: ...

class FrStream:
    cache: lal.Cache
    epoch: lal.LIGOTimeGPS
    file: FrFile
    fnum: int
    mode: int
    pos: int

def FrStreamCacheOpen(cache: lal.Cache) -> FrStream: ...
def FrStreamOpen(path: str, name: str) -> FrStream: ...
def FrStreamRewind(stream: FrStream) -> None: ...
def FrStreamNext(stream: FrStream) -> None: ...
def FrStreamSeek(stream: FrStream, epoch: lal.LIGOTimeGPS) -> None: ...

def FrStreamGetTimeSeriesType(channel: str, stream: FrStream) -> int: ...

def FrStreamReadINT2TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.INT2TimeSeries: ...

def FrStreamReadINT4TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.INT4TimeSeries: ...

def FrStreamReadINT8TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.INT8TimeSeries: ...

def FrStreamReadUINT2TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.UINT2TimeSeries: ...

def FrStreamReadUINT4TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.UINT4TimeSeries: ...

def FrStreamReadUINT8TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.UINT8TimeSeries: ...

def FrStreamReadREAL4TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.REAL4TimeSeries: ...

def FrStreamReadREAL8TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.REAL8TimeSeries: ...

def FrStreamReadCOMPLEX8TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.COMPLEX8TimeSeries: ...

def FrStreamReadCOMPLEX16TimeSeries(
    stream: FrStream,
    channel: str,
    start: lal.LIGOTimeGPS,
    duration: float,
    zero: int,
) -> lal.COMPLEX16TimeSeries: ...
