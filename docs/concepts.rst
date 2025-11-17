.. _gwpy-concepts:

############
Key Concepts
############

This guide introduces the fundamental concepts you need to understand to work
effectively with GWpy.

----

Core Data Structures
====================

GWpy provides three main data structures for representing scientific data:

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: **Time-domain data**
        :link: /reference/gwpy.timeseries.TimeSeries
        :link-type: doc

        :class:`~gwpy.timeseries.TimeSeries`
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Detector strain, auxiliary channels, and any signal that varies with time.

    .. grid-item-card:: **Frequency-domain data**
        :link: /reference/gwpy.frequencyseries.FrequencySeries
        :link-type: doc

        :class:`~gwpy.frequencyseries.FrequencySeries`
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Power spectra, amplitude spectral densities, transfer functions.

    .. grid-item-card:: **Time-frequency data**
        :link: /reference/gwpy.spectrogram.Spectrogram
        :link-type: doc

        :class:`~gwpy.spectrogram.Spectrogram`
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        How frequency content evolves over time.

TimeSeries: Time-Domain Data
-----------------------------

A :class:`~gwpy.timeseries.TimeSeries` represents data sampled at regular
intervals in time.
Think of it as a :class:`NumPy array <numpy.ndarray>` with metadata:

.. code-block:: python

    from gwpy.timeseries import TimeSeries
    import numpy as np

    # Create a TimeSeries
    data = TimeSeries([1, 2, 3, 4, 5], sample_rate=10, unit="m")

    print(f"Data: {data.value}")         # [1 2 3 4 5]
    print(f"Times: {data.times}")        # [0.0, 0.1, 0.2, 0.3, 0.4] s
    print(f"Sample rate: {data.sample_rate}")  # 10.0 Hz
    print(f"Duration: {data.duration}")  # 0.5 s

**Key Attributes:**

.. autosummary::

    ~gwpy.timeseries.TimeSeries.sample_rate
    ~gwpy.timeseries.TimeSeries.dt
    ~gwpy.timeseries.TimeSeries.t0
    ~gwpy.timeseries.TimeSeries.span
    ~gwpy.timeseries.TimeSeries.unit
    ~gwpy.timeseries.TimeSeries.channel

FrequencySeries: Frequency-Domain Data
---------------------------------------

A :class:`~gwpy.frequencyseries.FrequencySeries` represents data in the
frequency domain, typically from a Fourier transform:

.. code-block:: python

    from gwpy.timeseries import TimeSeries

    # Create time-domain signal
    ts = TimeSeries.get("L1", 1126259446, 1126259478)

    # Transform to frequency domain
    asd = ts.asd(fftlength=4)

    print(f"Frequency spacing: {asd.df}")  # 0.25 Hz
    print(f"Frequency range: {asd.f0} to {asd.frequencies[-1]}")

**Key Attributes:**

.. autosummary::

    ~gwpy.frequencyseries.FrequencySeries.df
    ~gwpy.frequencyseries.FrequencySeries.f0
    ~gwpy.frequencyseries.FrequencySeries.frequencies

Spectrogram: Time-Frequency Data
---------------------------------

A :class:`~gwpy.spectrogram.Spectrogram` is a 2D array showing how frequency
content changes over time:

.. code-block:: python

    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.get("L1", 1126259446, 1126259478)

    # Create spectrogram with 1-second time bins
    specgram = ts.spectrogram(stride=1, fftlength=4)

    print(f"Shape: {specgram.shape}")  # (time_bins, frequency_bins)
    print(f"Time resolution: {specgram.dt}")  # 1.0 s
    print(f"Frequency resolution: {specgram.df}")  # 0.25 Hz

Think of it as a stack of frequency spectra, one for each time bin.

**Key Attributes:**

.. autosummary::

    ~gwpy.spectrogram.Spectrogram.t0
    ~gwpy.spectrogram.Spectrogram.dt
    ~gwpy.spectrogram.Spectrogram.f0
    ~gwpy.spectrogram.Spectrogram.df
    ~gwpy.spectrogram.Spectrogram.times
    ~gwpy.spectrogram.Spectrogram.frequencies

----

GPS Time
========

Gravitational-wave detectors use **GPS time**, which counts seconds since
January 6, 1980, 00:00:00 UTC (the GPS epoch).

Why GPS Time?
-------------

- **No leap seconds**: Unlike UTC, GPS time is continuous
- **High precision**: Integer seconds with sub-second fractions
- **Universal**: Same at all detector sites

Converting Between Time Systems
--------------------------------

GWpy's :mod:`gwpy.time` module helps convert between time systems:

.. code-block:: python

    from gwpy.time import to_gps, from_gps

    # Convert from human-readable time
    gps = to_gps("September 14, 2015 09:50:45 UTC")
    print(gps)  # 1126259462

    # Convert back
    date = from_gps(1126259462)
    print(date)  # 2015-09-14 09:50:45 UTC

.. tip::

    GWpy's :func:`~gwpy.time.to_gps` function is very flexible:

    .. code-block:: python

        to_gps("now")
        to_gps("Jan 1, 2020")
        to_gps("2015-09-14 09:50:45")
        to_gps(1126259462)  # GPS already, returns unchanged

----

Units and Quantities
====================

GWpy uses :ref:`Astropy's units <astropy-units>` framework to
handle physical units safely:

.. code-block:: python

    from gwpy.timeseries import TimeSeries

    # Units are preserved in operations
    ts = TimeSeries([1, 2, 3], sample_rate=10, unit="m")

    print(ts.unit)  # m (meters)

    # Convert units
    ts_mm = ts.to("mm")
    print(ts_mm.value)  # [1000, 2000, 3000]
    print(ts_mm.unit)   # mm

Why Units Matter
----------------

Units catch errors at runtime:

.. code-block:: python

    from astropy import units as u

    distance = 1 * u.m
    time = 2 * u.s

    speed = distance / time
    print(speed)  # 0.5 m / s  ✓

    # This would raise an error:
    # result = distance + time  # Can't add meters and seconds!

Common Units in GWpy
--------------------

- **Strain**: dimensionless (detector output)
- **ASD**: ``strain / sqrt(Hz)`` (noise amplitude)
- **PSD**: ``strain^2 / Hz`` (noise power)
- **Time**: ``s`` (seconds)
- **Frequency**: ``Hz`` (Hertz)

----

Channels
========

In gravitational-wave detectors, a **channel** is a named data stream from
a sensor or derived calculation.

Channel Naming
--------------

Channels follow the pattern ``IFO:SUBSYSTEM-SIGNAL_NAME``:

.. code-block:: python

    # Examples:
    "L1:GDS-CALIB_STRAIN"      # LIGO Livingston calibrated strain
    "H1:GDS-CALIB_STRAIN"      # LIGO Hanford calibrated strain
    "V1:Hrec_hoft_16384Hz"     # Virgo strain
    "L1:PSL-ISS_PDA_OUT_DQ"    # Auxiliary channel

Components:

- ``L1`` / ``H1`` / ``V1`` - Interferometer name
- ``GDS`` - Subsystem (Global Diagnostics System)
- ``CALIB_STRAIN`` - Signal name (calibrated strain)

.. seealso::

    :doc:`/detector/channel` for more about channel names

----

Data Quality and Segments
==========================

Not all detector data are suitable for analysis.
**Data quality flags** indicate when data meets certain criteria.

Segments and SegmentLists
--------------------------

A :class:`~gwpy.segments.Segment` is a semi-open GPS time interval ``[start, stop)``:

.. code-block:: python

    from gwpy.segments import Segment, SegmentList

    # Single segment
    seg = Segment(1126259447, 1126259479)
    print(f"Duration: {abs(seg)} seconds")  # 32 seconds

    # List of segments
    seglist = SegmentList([
        Segment(0, 10),
        Segment(20, 30),
        Segment(25, 35),  # Overlaps with previous
    ])

    # Coalesce overlapping segments
    seglist.coalesce()
    print(seglist)  # [(0, 10), (20, 35)]

Data Quality Flags
------------------

A :class:`~gwpy.segments.DataQualityFlag` has:

- :attr:`~DataQualityFlag.known` segments - when data exists
- :attr:`~DataQualityFlag.active` segments - when the condition is true

.. code-block:: python

    from gwpy.segments import DataQualityFlag

    # Example: detector in "analysis ready" state
    dqflag = DataQualityFlag.query(
        "L1:DMT-ANALYSIS_READY:1",
        1126259447,
        1126259479,
    )

    print(f"Known time: {abs(dqflag.known)}")
    print(f"Active time: {abs(dqflag.active)}")
    print(f"Efficiency: {dqflag.active / dqflag.known * 100:.1f}%")

.. seealso::

    :doc:`/segments/index`: for a complete guide to segments.

----

Signal Processing Basics
=========================

Filtering
---------

**Filters** remove unwanted frequencies from signals:

.. code-block:: python

    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.get("L1", 1126259446, 1126259478)

    # Highpass: remove frequencies below 30 Hz
    hp = ts.highpass(30)

    # Lowpass: remove frequencies above 300 Hz
    lp = ts.lowpass(300)

    # Bandpass: keep only 30-300 Hz
    bp = ts.bandpass(30, 300)

    # Notch: remove a single frequency (e.g., 60 Hz power line)
    notched = ts.notch(60)

Whitening
---------

**Whitening** normalizes the frequency-domain amplitude,
making all frequencies equally important:

.. code-block:: python

    # Whiten using a 4-second ASD estimate
    whitened = ts.whiten(fftlength=4)

This is crucial for matched filtering and viewing weak signals.

Spectral Estimation
-------------------

Compute **power spectral density (PSD)** or **amplitude spectral density (ASD)**:

.. code-block:: python

    # PSD: power per frequency bin
    psd = ts.psd(fftlength=4, method="median")

    # ASD: sqrt(PSD), more intuitive units
    asd = ts.asd(fftlength=4, method="median")

Common methods:

- ``'welch'`` - mean of overlapping FFTs (default)
- ``'median'`` - median of overlapping FFTs (robust to outliers)
- ``'bartlett'`` - mean of non-overlapping FFTs

----

Arrays and Metadata
===================

GWpy objects are built on NumPy arrays but add metadata:

Accessing the Array
-------------------

.. code-block:: python

    from gwpy.timeseries import TimeSeries
    import numpy as np

    ts = TimeSeries([1, 2, 3, 4, 5], sample_rate=1)

    # Get the underlying array
    arr = ts.value  # NumPy array

    # Or use as array directly
    mean = np.mean(ts)
    std = np.std(ts)

Operations Preserve Metadata
-----------------------------

.. code-block:: python

    # Arithmetic operations preserve metadata
    ts2 = ts * 2
    print(ts2.sample_rate)  # Still 1.0 Hz

    # Slicing preserves metadata
    segment = ts[2:5]
    print(segment.t0)  # Adjusted to slice start time

NumPy Compatibility
-------------------

Use GWpy objects with NumPy functions:

.. code-block:: python

    import numpy as np

    # NumPy functions work directly
    maximum = np.max(ts)
    smoothed = np.convolve(ts, np.ones(5)/5, mode="same")

    # But note: some operations may lose metadata

----

Next Steps
==========

Now that you understand these concepts:

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: 📖 Browse User Guide
        :link: /guide
        :link-type: doc

        Learn to accomplish specific tasks

    .. grid-item-card:: 🔍 Explore API
        :link: /reference/index
        :link-type: doc

        Detailed reference for all classes

    .. grid-item-card:: 🎨 See Examples
        :link: /examples/index
        :link-type: doc

        Working code examples
