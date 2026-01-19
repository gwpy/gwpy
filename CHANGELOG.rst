#########
Changelog
#########

.. _v3.0.14:

===================
3.0.14 - 2026-01-16
===================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.14>`__

Bug Fixes
~~~~~~~~~

- Fix compatibility with Scipy 1.17
  [:mr:`1981`]
- Improve the GWpyFormat unit parser
  [:mr:`1954`]
- Fix usage of chained classmethod decorator
  [:mr:`1906`]

Other Changes
~~~~~~~~~~~~~

- Add support for Python 3.13
  [:mr:`1905`]

.. _v3.0.13:

===================
3.0.13 - 2025-07-02
===================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.13>`__

Bug Fixes
~~~~~~~~~

- Fix incompatibility with numpy 2.3.x
  [:mr:`1896`]
- Work around bug in :func:`scipy.signal.csd`
  [:mr:`1893`]
- Fix unit parsing with Astropy 7.1
  [:mr:`1877`]
- Fix a bug with `~gwpy.segments.DataQualityFlag` subtraction & revise a few tests
  [:mr:`1703`]
- Add missing ``detection_snr`` argument in call to `gwpy.astro.inspiral_range`
  [:mr:`1615`]

Other Changes
~~~~~~~~~~~~~

- Pin to python-ligo-lw <2.0.0
  [:mr:`1899`]
- Don’t set number attribute for `gwpy.plot.Plot`
  [:mr:`1834`]
- Typos in doc
  [:mr:`1778`]

.. _v3.0.12:

===================
3.0.12 - 2025-02-06
===================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.12>`__

API Changes
~~~~~~~~~~~

- Add support for timeout in `~gwpy.timeseries.TimeSeries.fetch_open_data`
  [:mr:`1832`]

.. _v3.0.11:

===================
3.0.11 - 2025-01-14
===================

This release fixes a few bugs and solves compatibility issues with the
latest release of Matplotlib.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.11>`__

API Changes
~~~~~~~~~~~

- Migrate from ligo-segments to igwn-segments
  [:mr:`1819`]

Bug Fixes
~~~~~~~~~

- Fix version comparisons in `gwpy.plot`
  [:mr:`1799`]

Other Changes
~~~~~~~~~~~~~

Changes:

- Drop support for Python 3.8
  [:mr:`1802`]
- Remove all use of CVMFS for data in tests
  [:mr:`1821`]

.. _v3.0.10:

===================
3.0.10 - 2024-08-30
===================

This release fixes a few bugs and solves compatibility issues with the
latest release of NumPy.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.10>`__

API Changes
~~~~~~~~~~~

- Modify coherence unit to be an alias for ``dimensionless_unscaled``
  [:mr:`1779`]
- Raise `NotImplementedError` from
  `StateTimeSeries.override_unit <gwpy.timeseries.StateTimeSeries.override_unit>`__
  when attempting to change to an unsupported unit
  [:mr:`1787`]

Bug Fixes
~~~~~~~~~

- Fix use of numpy types for v2.0
  [:mr:`1782`]
- Fix implicit array to scalar conversion in filter design
  [:mr:`1780`]
- Update import of ``interactive_bk`` for Matplotlib 3.9
  [:mr:`1770`]

Other Changes
~~~~~~~~~~~~~

- Skip `ReadTimeout` errors as standard for network tests
  [:mr:`1775`]
- Fix a few minor warnings emitted by Sphinx builds
  [:mr:`1772`]
- Fix man warnings in tests
  [:mr:`1784`]
- Use ID tokens for PyPI upload
  [:mr:`1783`]
- Skip tests that fail with 502 Bad Gateway errors from Zenodo.org
  [:mr:`1774`]
- Use ``release-branch-semver`` scheme with setuptools-scm
  [:mr:`1786`]
- Update all references to github.com
  [:mr:`1773`]
- Add ``K1_HOFT_C00`` frametype description in documentation
  [:mr:`1789`]

.. _v3.0.9:

==================
3.0.9 - 2024-07-13
==================

Patch release.

This release fixes a few bugs and resolves compatibility issues with the
latest releases of NumPy and SciPy.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.9>`__

Bug Fixes
~~~~~~~~~

- Fix argument reuse bug in `gwpy.cli`
  [:mr:`1747`]
- Fix usage of Hann window for Scipy 1.13.0
  [:mr:`1753`]
- Fix test failure with Zenodo rate limit
  [:mr:`1755`]
- Fix array copy compatibility with numpy 2
  [:mr:`1760`]
- Fix usage of Scipy firwin
  [:mr:`1762`]
- Fix usage of Scipy trapezoid in gwpy.astro
  [:mr:`1763`]

Other Changes
~~~~~~~~~~~~~

Changes:

- Support parsing multiple FFL files for a single dataset
  [:mr:`1616`]
- Remove redundant usetex test function
  [:mr:`1637`]
- Add tests for :mod:`gwpy.timeseries.io.cache`
  [:mr:`1641`]
- Add Virgo 'HoftOnline' as a preferred frametype
  [:mr:`1749`]
- Improve error handling when reading GWF with LALFrame
  [:mr:`1764`]
- Add extra unit alias for 's'
  [:mr:`1765`]

.. _v3.0.8:

==================
3.0.8 - 2024-01-12
==================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.8>`__

Bug Fixes
~~~~~~~~~

- Fix bug in :meth:`TimeSeriesDict.find <gwpy.timeseries.TimeSeriesDict.find>`
  [:mr:`1672`]
- Fix array indexing of ``Series.xindex``
  [:mr:`1685`]
- Fix missing sample when reading with non-exact sample times using
  LALFrame
  [:mr:`1694`]
- Fix bugs in inverse FFT normalisation
  [:mr:`1728`]

Other Changes
~~~~~~~~~~~~~

- Sieve URLs from gwosc client before fetch
  [:mr:`1665`]
- Handle reading ``LIGO_LW`` XML files that are missing the ``_ns``
  timing columns.
  [:mr:`1669`]
- Silence LAL ``LIGOTimeGPS`` warnings when parsing a cache
  [:mr:`1688`]
- Use class fixtures to download GWOSC data that is used many times
  [:mr:`1691`]
- Replace `maya` with `dateparser` in :mod:`gwpy.time`
  [:mr:`1695`]
- Fix coherence test fixture
  [:mr:`1698`]
- Fix the Zenodo RST utility module
  [:mr:`1699`]
- Drop ``HAS_TEX`` for faster startup time
  [:mr:`1712`]
- Track LALSuite rebrand in docs
  [:mr:`1713`]
- Add support for Python 3.12
  [:mr:`1714`]
- Avoid python-framel 8.46.0
  [:mr:`1715`]
- Migrate test requirement to pytest-freezer
  [:mr:`1716`]
- Remove all use of distutils
  [:mr:`1718`]
- Replace `datetime.utcnow` with ``datetime.now(UTC)``
  [:mr:`1719`]
- Add aliases for LIGO and Virgo's 'time' units
  [:mr:`1721`]
- Read the unitX from a GWF with frameCPP
  [:mr:`1722`]
- Allow localhost sockets during dist testing
  [:mr:`1723`]
- Propagate source keyword in `DataQualityDict.populate <gwpy.segments.DataQualityDict.populate>`
  [:mr:`1725`, :mr:`1730`]
- Update documentation references for lscsoft-glue and LALSuite
  [:mr:`1733`]
- Replace ``sphinx-panels`` tabs with ``sphinx-immaterial`` tabs
  [:mr:`1734`]

.. _v3.0.7:

==================
3.0.7 - 2023-10-05
==================

Build fix release.

The distributions for GWpy 3.0.6 were corrupted, so 3.0.7 replaces that
release.

There are no code changes in 3.0.7 relative to 3.0.6

.. _v3.0.6:

==================
3.0.6 - 2023-10-05
==================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.6>`__

Bug Fixes
~~~~~~~~~

- Fix type comparison lint
  [:mr:`1654`]
- Fix compatibility with matplotlib 3.8.0
  [:mr:`1661`]

Other Changes
~~~~~~~~~~~~~

- Drop support for Python 3.7
  [:mr:`1636`]
- Don’t round ``TimeSeries.dt`` when setting via ``sample_rate``
  [:mr:`1649`]
- Improve tests of ``TimeSeriesDict.find`` error handling
  [:mr:`1676`]
- Fix all executable lint issues
  [:mr:`1677`]

.. _v3.0.5:

==================
3.0.5 - 2023-06-02
==================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.5>`__

Bug Fixes
~~~~~~~~~

- Fix issues writing ``StateVector.bits`` to HDF5
  [:mr:`1608`]
- Fix plotting ``TimeSeriesDict`` on separate Axes
  [:mr:`1610`]
- Fix issues with ``Series.crop`` precision
  [:mr:`1621`]
- Address inconsistency in ``transfer_function`` between CSD and PSD
  averaging methods
  [:mr:`1619`]

Other Changes
~~~~~~~~~~~~~

- Update all GWOSC URLs to gwosc.org
  [:mr:`1613`]
- Resample time series for coherence calculation when sampling rates
  differ
  [:mr:`1611`]
- Update ``LatexInlineDimensional`` unit formatter for Astropy 5.3
  [:mr:`1622`]
- Document the ``copy`` keyword for ``TimeSeriesBaseDict.crop``
  [:mr:`1617`]
- Add H0 and L0 to dynamic_scaled exclusion list
  [:mr:`1628`]

.. _v3.0.4:

==================
3.0.4 - 2023-04-12
==================

Bug fix release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.4>`__

Bug Fixes
~~~~~~~~~

- Fix incompatibility with gwosc 0.7.0
  [:mr:`1606`]

.. _v3.0.3:

==================
3.0.3 - 2023-04-09
==================

Bug fix release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.3>`__

Bug Fixes
~~~~~~~~~

- Fix incompatibilities with matplotlib 3.7.0
  [:mr:`1586`]
- Fix incorrect phase for transfer function calculation
  [:mr:`1590`]
- Address floating point errors for index creation
  [:mr:`1600`]

Other Changes
~~~~~~~~~~~~~

- Update usage of ``pytest.raises`` throughout the codebase
  [:mr:`1542`]
- Improve handling/propagation of maya/dateparser errors and refactor
  ``gwpy.time`` tests
  [:mr:`1559`]
- Improve documentation of GWF I/O libraries
  [:mr:`1563`]
- Update import of ``matplotlib.docstrings``
  [:mr:`1566`]
- Update a number of CLI examples to use GWTC-3 data
  [:mr:`1567`]
- Update GWF writing for frameCPP 3.0.0
  [:mr:`1575`]
- Add wrapper around ``scipy.signal.get_window`` to handle pre-computed
  window arrays
  [:mr:`1576`]
- Gracefully skip ``requests.exceptions.ConnectionError`` in testing
  [:mr:`1577`]
- Improve NDSWarning message emitted by ``TimeSeriesDict.fetch``
  [:mr:`1578`]
- Replace usage of deprecated ``matplotlib.ticker.is_close_to_int``
  [:mr:`1579`]
- Enhance LaTeX detection
  [:mr:`1585`]
- Update qscan example
  [:mr:`1587`]
- Replace deprecated ``interp2d`` with ``RectBivariateSpline``
  [:mr:`1602`]

.. _v3.0.2:

==================
3.0.2 - 2022-11-24
==================

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.2>`__

Bug Fixes
~~~~~~~~~

- Fix reading ILWD chars from LIGO_LW XML with python-ligo-lw
  [:mr:`1570`]

Other Changes
~~~~~~~~~~~~~

- Declare support for Python 3.11
  [:mr:`1555`]

.. _v3.0.1:

==================
3.0.1 - 2022-11-08
==================

Bug-fix release for 3.0.x.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.1>`__

Bug Fixes
~~~~~~~~~

- Fix Python 3.11 compatibility in ``gwpy.io.nds2``
  [:mr:`1558`]

Other Changes
~~~~~~~~~~~~~

- Handle optional kerberos with requests-ecp
  [:mr:`1554`]

.. _v3.0.0:

==================
3.0.0 - 2022-10-04
==================

Major feature release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v3.0.0>`__

Deprecations
~~~~~~~~~~~~

- ``gwpy.testing.utils.skip_missing_dependency`` is now deprecated and
  will be removed in a future release
  [:mr:`1531`]

Expired Deprecations
~~~~~~~~~~~~~~~~~~~~

- Remove support for S6-style SegDB server
  [:mr:`1356`]
- Remove support for ``hdf5.losc`` file format key
  [:mr:`1465`]
- Remove ``gwpy.table.EventColum``
  [:mr:`1466`]
- Remove deprecated ``Plot`` methods
  [:mr:`1467`]
- Remove deprecated colorbar behaviour
  [:mr:`1468`]
- Remove deprecated ``SegmentAxes`` methods
  [:mr:`1469`]
- Remove deprecated ``TimeSeries.read`` options for GWF
  [:mr:`1470`]
- Remove deprecated ``Spectrogram.plot`` keyword
  [:mr:`1472`]

New Features
~~~~~~~~~~~~

- Add support for reading GstLAL trigger files
  [:mr:`1265`]
- Add support for filter functions to use multiple columns with
  ``EventTable.filter``
  [:mr:`1391`]
- Add ``TimeSeries.transfer_function``
  [:mr:`1406`]
- Support writing ROOT files with ``uproot4``
  [:mr:`1419`]

API Changes
~~~~~~~~~~~

- Refactor gwpy.astro to depend on inspiral-range
  [:mr:`1284`]
- Update API for ``gwpy.utils.lal.to_lal_unit`` function
  [:mr:`1479`]
- Remove equivalence between ``strain`` and ``dimensionless_unscaled``
  units [:mr:`1477`]
- Update ``Axes.tile`` to use ``sortbycolor`` keyword
  [:mr:`1483`]

Other Changes
~~~~~~~~~~~~~

- Improve handling of empty arrays when writing GWF with LALFrame
  [:mr:`1501`]
- Use ``pytest.mark.requires`` in all tests
  [:mr:`1531`]
- Improve PyCBC HDF5 table reading
  [:mr:`1543`]

A number of bugs were fixed, and compatibility improved with advanced
versions of the requirements.

.. _v2.1.5:

==================
2.1.5 - 2022-08-01
==================

Patch release for GWpy 2.1.x.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.5>`__

Bug Fixes
~~~~~~~~~

- Fix compatibility with scipy 1.9.0
  [:mr:`1521`]
- Deprecate the ``gwpy.utils.null_context`` function
  [:mr:`1523`]

Other Changes
~~~~~~~~~~~~~

- Fix some lint
  [:mr:`1530`]

.. _v2.1.4:

==================
2.1.4 - 2022-06-27
==================

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.4>`__

Bug Fixes
~~~~~~~~~

- handle cases in ``EventTable.cluster()`` where table is empty or
  already clustered
  [:mr:`1363`]
- fix bug in ``Series`` index comparison
  [:mr:`1428`]
- fix bug reading ASCII Files
  [:mr:`1474`]
- resolve ``DeprecationWarning`` from SciPy 1.8.0
  [:mr:`1484`]
- fix reading a contiguous set of files with LALFrame
  [:mr:`1487`]
- migrate GWDataFind interface to new UI
  [:mr:`1499`]
- update name of Gravity Spy server for glitch tables
  [:mr:`1512`]
- work around a bug in frameCPP
  [:mr:`1517`]

Other Changes
~~~~~~~~~~~~~

- Python >=3.7
  [:mr:`1441`]
- h5py >=2.8.0
  [:mr:`1441`]
- Astropy >=4.0
  [:mr:`1463`]
- NumPy >=1.16
  [:mr:`1463`]
- GWDataFind >=1.1.0
  [:mr:`1499`]

.. _v2.1.3:

==================
2.1.3 - 2021-12-17
==================

Patch release for GWpy 2.1.x.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.3>`__

Bug Fixes
~~~~~~~~~

- update test suite to use tmp_path fixture in pytest
  [:mr:`1401`]
- add support for Python 3.10
  [:mr:`1418`]
- fix DeprecationWarning emitted in Python 3.10
  [:mr:`1439`]
- update timeseries/public example
  [:mr:`1438`]
- decorate test to skip gracefully on network error
  [:mr:`1456`]
- fix compatibility with python-ligo-lw 1.8.0
  [:mr:`1457`]
- fix compatibility with scipy 1.8.0
  [:mr:`1458`]

.. _v2.1.2:

==================
2.1.2 - 2021-11-25
==================

Patch release for GWpy 2.1.x.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.2>`__

Deprecations
~~~~~~~~~~~~

- ``gwpy.utils.shell.which()`` is now deprecated
  [:mr:`1411`]

Bug Fixes
~~~~~~~~~

- writing a ``TimeSeries`` with no ``name`` to GWF now has consistent
  behaviour across all GWF APIs
  [:mr:`1425`]
- fixed compatibility with Astropy 5.0
  [:mr:`1435`]

.. _v2.1.1:

==================
2.1.1 - 2021-10-18
==================

Patch release.

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.1>`__

Deprecations
~~~~~~~~~~~~

- ``gwpy.utils.shell.call()`` is now deprecated
  [:mr:`1412`]

API Changes
~~~~~~~~~~~

- a ``strict`` keyword was added to ``gwpy.io.cache.sieve()``
  [:mr:`1417`]

Bug Fixes
~~~~~~~~~

- ``Series.crop()`` can now handle irregular indices (thanks :ghu:`mattpitkin`!)
  [:mr:`1414`]
- ``TimeSeries.read`` no longer checks the ``tRange`` of an
  ``FrProcData`` when reading GWF using FrameCPP
  [:mr:`1420`]

.. _v2.1.0:

==================
2.1.0 - 2021-08-30
==================

`Full details <https://gitlab.com/gwpy/gwpy/-/releases/v2.1.0>`__

Deprecations
~~~~~~~~~~~~

- the ``gwpy.testing.utils.TemporaryFile`` context manager is now
  deprecated
  [:mr:`1353`]
- ``gwpy.astro.inspiral_range`` function renamed to ``sensemon_range``
  [:mr:`1293`]

New Features
~~~~~~~~~~~~

- ROOT I/O now uses ``uproot`` as the backend
  [:mr:`1263`,
  `#1309 <https://github.com/gwpy/gwpy/issues/1309>`__]
- ``pathlib.Path`` objects are now supported everywhere file names are
  passed around (hopefully)
  [:mr:`1362`,
  `#1380 <https://github.com/gwpy/gwpy/issues/1380>`__]
- ``gwpy.signal.filter_design.notch`` now supports an ``output`` keyword
  [:mr:`1347`]
- ``gwpy-plot qtransform`` now supports FFT options on the command-line
  [:mr:`1330`]
- new ``--average-method`` command-line option for ``gwpy-plot``
  [:mr:`1329`]
- ``TimeSeries.rayleigh_spectrum`` now supports a ``window`` keyword
  [:mr:`1285`]

API Changes
~~~~~~~~~~~

- passing a ``channel`` name is now optional when reading files in the
  ``hdf5.snax`` format
  [:mr:`1275`]
- the default spectral averaging method is now 'median' (was 'mean')
  [:mr:`1282`]
