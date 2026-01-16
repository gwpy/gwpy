<!-- markdownlint-disable MD022 -->

# Changelog

(v3.0.14)=
## 3.0.14 - 2026-01-16

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.14)

### Bug Fixes

-   Fix compatibility with Scipy 1.17
    [[!1981](https://gitlab.com/gwpy/gwpy/-/merge_requests/1981)]
-   Improve the GWpyFormat unit parser
    [[!1954](https://gitlab.com/gwpy/gwpy/-/merge_requests/1954)]
-   Fix usage of chained classmethod decorator
    [[!1906](https://gitlab.com/gwpy/gwpy/-/merge_requests/1906)]

### Other Changes

-   Add support for Python 3.13
    [[!1905](https://gitlab.com/gwpy/gwpy/-/merge_requests/1905)]

(v3.0.13)=
## 3.0.13 - 2025-07-02

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.13)

### Bug Fixes

-   Fix incompatibility with numpy 2.3.x
    [[!1896](https://gitlab.com/gwpy/gwpy/-/merge_requests/1896)]
-   Work around bug in `scipy.signal.csd`
    [[!1893](https://gitlab.com/gwpy/gwpy/-/merge_requests/1893)]
-   Fix unit parsing with Astropy 7.1
    [[!1877](https://gitlab.com/gwpy/gwpy/-/merge_requests/1877)]
-   Fix a bug with `DataQualityFlag` subtraction & revise a few tests
    [[!1703](https://gitlab.com/gwpy/gwpy/-/merge_requests/1703)]
-   Add missing `detection_snr` argument
    [[!1615](https://gitlab.com/gwpy/gwpy/-/merge_requests/1615)]

### Other Changes

-   Pin to python-ligo-lw <2.0.0
    [[!1899](https://gitlab.com/gwpy/gwpy/-/merge_requests/1899)]
-   Don't set number attribute for `gwpy.plot.Plot`
    [[!1834](https://gitlab.com/gwpy/gwpy/-/merge_requests/1834)]
-   Typos in doc
    [[!1778](https://gitlab.com/gwpy/gwpy/-/merge_requests/1778)]

(v3.0.12)=
## 3.0.12 - 2025-02-06

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.12)

### API Changes

-   Add support for timeout in `fetch_open_data`
    [[!1832](https://gitlab.com/gwpy/gwpy/-/merge_requests/1832)]

(v3.0.11)=
## 3.0.11 - 2025-01-14

This release fixes a few bugs and solves compatibility issues with the latest
release of Matplotlib.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.11)

### API Changes

-   Migrate from ligo-segments to igwn-segments
    [[!1819](https://gitlab.com/gwpy/gwpy/-/merge_requests/1819)]

### Bug Fixes

-   Fix version comparisons in `gwpy.plot`
    [[!1799](https://gitlab.com/gwpy/gwpy/-/merge_requests/1799)]

### Other Changes

Changes:

-   Drop support for Python 3.8
    [[!1802](https://gitlab.com/gwpy/gwpy/-/merge_requests/1802)]
-   Remove all use of CVMFS for data in tests
    [[!1821](https://gitlab.com/gwpy/gwpy/-/merge_requests/1821)]

(v3.0.10)=
## 3.0.10 - 2024-08-30

This release fixes a few bugs and solves compatibility issues with the latest
release of NumPy.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.10)

### API Changes

-   Modify coherence unit to be an alias for `dimensionless_unscaled`
    [[!1779](https://gitlab.com/gwpy/gwpy/-/merge_requests/1779)]
-   Raise `NotImplementedError` from `StateTimeSeries.override_unit`
    [[!1787](https://gitlab.com/gwpy/gwpy/-/merge_requests/1787)]

### Bug Fixes

-   Fix use of numpy types for v2.0
    [[!1782](https://gitlab.com/gwpy/gwpy/-/merge_requests/1782)]
-   Fix implicit array to scalar conversion in filter design
    [[!1780](https://gitlab.com/gwpy/gwpy/-/merge_requests/1780)]
-   Update import of `interactive_bk` for Matplotlib 3.9
    [[!1770](https://gitlab.com/gwpy/gwpy/-/merge_requests/1770)]

### Other Changes

-   Skip `ReadTimeout` errors as standard for network tests
    [[!1775](https://gitlab.com/gwpy/gwpy/-/merge_requests/1775)]
-   Fix a few minor warnings emitted by Sphinx builds
    [[!1772](https://gitlab.com/gwpy/gwpy/-/merge_requests/1772)]
-   Fix man warnings in tests
    [[!1784](https://gitlab.com/gwpy/gwpy/-/merge_requests/1784)]
-   Use ID tokens for PyPI upload
    [[!1783](https://gitlab.com/gwpy/gwpy/-/merge_requests/1783)]
-   Skip tests that fail with 502 Bad Gateway errors from Zenodo.org
    [[!1774](https://gitlab.com/gwpy/gwpy/-/merge_requests/1774)]
-   Use `release-branch-semver` scheme with setuptools-scm
    [[!1786](https://gitlab.com/gwpy/gwpy/-/merge_requests/1786)]
-   Update all references to github.com
    [[!1773](https://gitlab.com/gwpy/gwpy/-/merge_requests/1773)]
-   Add `K1_HOFT_C00` frametype description in documentation
    [[!1789](https://gitlab.com/gwpy/gwpy/-/merge_requests/1789)]

(v3.0.9)=
## 3.0.9 - 2024-07-13

Patch release.

This release fixes a few bugs and resolves compatibility issues with the
latest releases of NumPy and SciPy.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.9)

### Bug Fixes

-   Fix argument reuse bug in `gwpy.cli`
    [[!1747](https://gitlab.com/gwpy/gwpy/-/merge_requests/1747)]
-   Fix usage of Hann window for Scipy 1.13.0
    [[!1753](https://gitlab.com/gwpy/gwpy/-/merge_requests/1753)]
-   Fix test failure with Zenodo rate limit
    [[!1755](https://gitlab.com/gwpy/gwpy/-/merge_requests/1755)]
-   Fix array copy compatibility with numpy 2
    [[!1760](https://gitlab.com/gwpy/gwpy/-/merge_requests/1760)]
-   Fix usage of Scipy firwin
    [[!1762](https://gitlab.com/gwpy/gwpy/-/merge_requests/1762)]
-   Fix usage of Scipy trapezoid in gwpy.astro
    [[!1763](https://gitlab.com/gwpy/gwpy/-/merge_requests/1763)]

### Other Changes

Changes:

-   Support parsing multiple FFL files for a single dataset
    [[!1616](https://gitlab.com/gwpy/gwpy/-/merge_requests/1616)]
-   Remove redundant usetex test function
    [[!1637](https://gitlab.com/gwpy/gwpy/-/merge_requests/1637)]
-   Add tests for `gwpy.timeseries.io.cache`
    [[!1641](https://gitlab.com/gwpy/gwpy/-/merge_requests/1641)]
-   Add Virgo 'HoftOnline' as a preferred frametype
    [[!1749](https://gitlab.com/gwpy/gwpy/-/merge_requests/1749)]
-   Improve error handling when reading GWF with LALFrame
    [[!1764](https://gitlab.com/gwpy/gwpy/-/merge_requests/1764)]
-   Add extra unit alias for 's'
    [[!1765](https://gitlab.com/gwpy/gwpy/-/merge_requests/1765)]

(v3.0.8)=
## 3.0.8 - 2024-01-12

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.8)

### Bug Fixes

-   Fix bug in `TimeSeriesDict.find`
    [[!1672](https://gitlab.com/gwpy/gwpy/-/merge_requests/1672)]
-   Fix array indexing of `Series.xindex`
    [[!1685](https://gitlab.com/gwpy/gwpy/-/merge_requests/1685)]
-   Fix missing sample when reading with non-exact sample times using LALFrame
    [[!1694](https://gitlab.com/gwpy/gwpy/-/merge_requests/1694)]
-   Fix bugs in inverse FFT normalisation
    [[!1728](https://gitlab.com/gwpy/gwpy/-/merge_requests/1728)]

### Other Changes

-   Sieve URLs from gwosc client before fetch
    [[!1665](https://gitlab.com/gwpy/gwpy/-/merge_requests/1665)]
-   Handle reading `LIGO_LW` XML files that are missing the `_ns` timing columns.
    [[!1669](https://gitlab.com/gwpy/gwpy/-/merge_requests/1669)]
-   Silence LAL `LIGOTimeGPS` warnings when parsing a cache
    [[!1688](https://gitlab.com/gwpy/gwpy/-/merge_requests/1688)]
-   Use class fixtures to download GWOSC data that is used many times
    [[!1691](https://gitlab.com/gwpy/gwpy/-/merge_requests/1691)]
-   Replace `maya` with `dateparser` in `gwpy.time`
    [[!1695](https://gitlab.com/gwpy/gwpy/-/merge_requests/1695)]
-   Fix coherence test fixture
    [[!1698](https://gitlab.com/gwpy/gwpy/-/merge_requests/1698)]
-   Fix the Zenodo RST utility module
    [[!1699](https://gitlab.com/gwpy/gwpy/-/merge_requests/1699)]
-   Drop `HAS_TEX` for faster startup time
    [[!1712](https://gitlab.com/gwpy/gwpy/-/merge_requests/1712)]
-   Track LALSuite rebrand in docs
    [[!1713](https://gitlab.com/gwpy/gwpy/-/merge_requests/1713)]
-   Add support for Python 3.12
    [[!1714](https://gitlab.com/gwpy/gwpy/-/merge_requests/1714)]
-   Avoid python-framel 8.46.0
    [[!1715](https://gitlab.com/gwpy/gwpy/-/merge_requests/1715)]
-   Migrate test requirement to pytest-freezer
    [[!1716](https://gitlab.com/gwpy/gwpy/-/merge_requests/1716)]
-   Remove all use of distutils
    [[!1718](https://gitlab.com/gwpy/gwpy/-/merge_requests/1718)]
-   Replace `datetime.utcnow()` with `datetime.now(UTC)`
    [[!1719](https://gitlab.com/gwpy/gwpy/-/merge_requests/1719)]
-   Add aliases for LIGO and Virgo's 'time' units
    [[!1721](https://gitlab.com/gwpy/gwpy/-/merge_requests/1721)]
-   Read the unitX from a GWF with frameCPP
    [[!1722](https://gitlab.com/gwpy/gwpy/-/merge_requests/1722)]
-   Allow localhost sockets during dist testing
    [[!1723](https://gitlab.com/gwpy/gwpy/-/merge_requests/1723)]
-   Propagate source keyword in `DataQualityDict.populate`
    [[!1730](https://gitlab.com/gwpy/gwpy/-/merge_requests/1730)]
-   Update documentation references for lscsoft-glue and LALSuite
    [[!1733](https://gitlab.com/gwpy/gwpy/-/merge_requests/1733)]
-   Replace `sphinx-panels` tabs with `sphinx-immaterial` tabs
    [[!1734](https://gitlab.com/gwpy/gwpy/-/merge_requests/1734)]

(v3.0.7)=
## 3.0.7 - 2023-10-05

Build fix release.

The distributions for GWpy 3.0.6 were corrupted, so 3.0.7 replaces that release.

There are no code changes in 3.0.7 relative to 3.0.6

(v3.0.6)=
## 3.0.6 - 2023-10-05

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.6)

### Bug Fixes

-   Fix type comparison lint
    [[!1654](https://gitlab.com/gwpy/gwpy/-/merge_requests/1654)]
-   Fix compatibility with matplotlib 3.8.0
    [[!1661](https://gitlab.com/gwpy/gwpy/-/merge_requests/1661)]

### Other Changes

-   Drop support for Python 3.7
    [[!1636](https://gitlab.com/gwpy/gwpy/-/merge_requests/1636)]
-   Don't round `TimeSeries.dt` when setting via `sample_rate`
    [[!1649](https://gitlab.com/gwpy/gwpy/-/merge_requests/1649)]
-   Improve tests of `TimeSeriesDict.find` error handling
    [[!1676](https://gitlab.com/gwpy/gwpy/-/merge_requests/1676)]
-   Fix all executable lint issues
    [[!1677](https://gitlab.com/gwpy/gwpy/-/merge_requests/1677)]

(v3.0.5)=
## 3.0.5 - 2023-06-02

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.5)

### Bug Fixes

-   Fix issues writing `StateVector.bits` to HDF5
    [[!1608](https://gitlab.com/gwpy/gwpy/-/merge_requests/1608)]
-   Fix plotting `TimeSeriesDict` on separate Axes
    [[!1610](https://gitlab.com/gwpy/gwpy/-/merge_requests/1610)]
-   Fix issues with `Series.crop` precision
    [[!1621](https://gitlab.com/gwpy/gwpy/-/merge_requests/1621)]
-   Address inconsistency in `transfer_function` between CSD and PSD averaging methods
    [[!1619](https://gitlab.com/gwpy/gwpy/-/merge_requests/1619)]

### Other Changes

-   Update all GWOSC URLs to gwosc.org
    [[!1613](https://gitlab.com/gwpy/gwpy/-/merge_requests/1613)]
-   Resample time series for coherence calculation when sampling rates differ
    [[!1611](https://gitlab.com/gwpy/gwpy/-/merge_requests/1611)]
-   Update `LatexInlineDimensional` unit formatter for Astropy 5.3
    [[!1622](https://gitlab.com/gwpy/gwpy/-/merge_requests/1622)]
-   Document the `copy` keyword for `TimeSeriesBaseDict.crop`
    [[!1617](https://gitlab.com/gwpy/gwpy/-/merge_requests/1617)]
-   Add H0 and L0 to dynamic_scaled exclusion list
    [[!1628](https://gitlab.com/gwpy/gwpy/-/merge_requests/1628)]

(v3.0.4)=
## 3.0.4 - 2023-04-12

Bug fix release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.4)

### Bug Fixes

-   Fix incompatibility with gwosc 0.7.0
    [[!1606](https://gitlab.com/gwpy/gwpy/-/merge_requests/1606)]

(v3.0.3)=
## 3.0.3 - 2023-04-09

Bug fix release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.3)

### Bug Fixes

-   Fix incompatibilities with matplotlib 3.7.0
    [[!1586](https://gitlab.com/gwpy/gwpy/-/merge_requests/1586)]
-   Fix incorrect phase for transfer function calculation
    [[!1590](https://gitlab.com/gwpy/gwpy/-/merge_requests/1590)]
-   Address floating point errors for index creation
    [[!1600](https://gitlab.com/gwpy/gwpy/-/merge_requests/1600)]

### Other Changes

-   Update usage of `pytest.raises` throughout the codebase
    [[!1542](https://gitlab.com/gwpy/gwpy/-/merge_requests/1542)]
-   Improve handling/propagation of maya/dateparser errors
    and refactor `gwpy.time` tests
    [[!1559](https://gitlab.com/gwpy/gwpy/-/merge_requests/1559)]
-   Improve documentation of GWF I/O libraries
    [[!1563](https://gitlab.com/gwpy/gwpy/-/merge_requests/1563)]
-   Update import of `matplotlib.docstrings`
    [[!1566](https://gitlab.com/gwpy/gwpy/-/merge_requests/1566)]
-   Update a number of CLI examples to use GWTC-3 data
    [[!1567](https://gitlab.com/gwpy/gwpy/-/merge_requests/1567)]
-   Update GWF writing for frameCPP 3.0.0
    [[!1575](https://gitlab.com/gwpy/gwpy/-/merge_requests/1575)]
-   Add wrapper around `scipy.signal.get_window` to handle pre-computed window arrays
    [[!1576](https://gitlab.com/gwpy/gwpy/-/merge_requests/1576)]
-   Gracefully skip `requests.exceptions.ConnectionError` in testing
    [[!1577](https://gitlab.com/gwpy/gwpy/-/merge_requests/1577)]
-   Improve NDSWarning message emitted by `TimeSeriesDict.fetch`
    [[!1578](https://gitlab.com/gwpy/gwpy/-/merge_requests/1578)]
-   Replace usage of deprecated `matplotlib.ticker.is_close_to_int`
    [[!1579](https://gitlab.com/gwpy/gwpy/-/merge_requests/1579)]
-   Enhance LaTeX detection
    [[!1585](https://gitlab.com/gwpy/gwpy/-/merge_requests/1585)]
-   Update qscan example
    [[!1587](https://gitlab.com/gwpy/gwpy/-/merge_requests/1587)]
-   Replace deprecated `interp2d` with `RectBivariateSpline`
    [[!1602](https://gitlab.com/gwpy/gwpy/-/merge_requests/1602)]

(v3.0.2)=
## 3.0.2 - 2022-11-24

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.2)

### Bug Fixes

-   Fix reading ILWD chars from LIGO_LW XML with python-ligo-lw
    [[!1570](https://gitlab.com/gwpy/gwpy/-/merge_requests/1570)]

### Other Changes

-   Declare support for Python 3.11
    [[!1555](https://gitlab.com/gwpy/gwpy/-/merge_requests/1555)]

(v3.0.1)=
## 3.0.1 - 2022-11-08

Bug-fix release for 3.0.x.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.1)

### Bug Fixes

-   Fix Python 3.11 compatibility in `gwpy.io.nds2`
    [[!1558](https://gitlab.com/gwpy/gwpy/-/merge_requests/1558)]

### Other Changes

-   Handle optional kerberos with requests-ecp
    [[!1554](https://gitlab.com/gwpy/gwpy/-/merge_requests/1554)]

(v3.0.0)=
## 3.0.0 - 2022-10-04

Major feature release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v3.0.0)

### Deprecations

-   `gwpy.testing.utils.skip_missing_dependency` is now
    deprecated and will be removed in a future release
    [[!1531](https://gitlab.com/gwpy/gwpy/-/merge_requests/1531)]

### Expired Deprecations

-   Remove support for S6-style SegDB server
    [[!1356](https://gitlab.com/gwpy/gwpy/-/merge_requests/1356)]
-   Remove support for `hdf5.losc` file format key
    [[!1465](https://gitlab.com/gwpy/gwpy/-/merge_requests/1465)]
-   Remove `gwpy.table.EventColum`
    [[!1466](https://gitlab.com/gwpy/gwpy/-/merge_requests/1466)]
-   Remove deprecated `Plot` methods
    [[!1467](https://gitlab.com/gwpy/gwpy/-/merge_requests/1467)]
-   Remove deprecated colorbar behaviour
    [[!1468](https://gitlab.com/gwpy/gwpy/-/merge_requests/1468)]
-   Remove deprecated `SegmentAxes` methods
    [[!1469](https://gitlab.com/gwpy/gwpy/-/merge_requests/1469)]
-   Remove deprecated `TimeSeries.read` options for GWF
    [[!1470](https://gitlab.com/gwpy/gwpy/-/merge_requests/1470)]
-   Remove deprecated `Spectrogram.plot` keyword
    [[!1472](https://gitlab.com/gwpy/gwpy/-/merge_requests/1472)]

### New Features

-   Add support for reading GstLAL trigger files
    [[!1265](https://gitlab.com/gwpy/gwpy/-/merge_requests/1265)]
-   Add support for filter functions to use multiple
    columns with `EventTable.filter`
    [[!1391](https://gitlab.com/gwpy/gwpy/-/merge_requests/1391)]
-   Add `TimeSeries.transfer_function`
    [[!1406](https://gitlab.com/gwpy/gwpy/-/merge_requests/1406)]
-   Support writing ROOT files with `uproot4`
    [[!1419](https://gitlab.com/gwpy/gwpy/-/merge_requests/1419)]

### API Changes

-   Refactor gwpy.astro to depend on inspiral-range
    [[!1284](https://gitlab.com/gwpy/gwpy/-/merge_requests/1284)]
-   Update API for `gwpy.utils.lal.to_lal_unit` function
    [[!1479](https://gitlab.com/gwpy/gwpy/-/merge_requests/1479)]
-   Remove equivalence between `strain` and
    `dimensionless_unscaled` units
    [[!1477](https://gitlab.com/gwpy/gwpy/-/merge_requests/1477)]
-   Update `Axes.tile` to use `sortbycolor` keyword
    [[!1483](https://gitlab.com/gwpy/gwpy/-/merge_requests/1483)]

### Other Changes

-   Improve handling of empty arrays when writing GWF with LALFrame
    [[!1501](https://gitlab.com/gwpy/gwpy/-/merge_requests/1501)]
-   Use `pytest.mark.requires` in all tests
    [[!1531](https://gitlab.com/gwpy/gwpy/-/merge_requests/1531)]
-   Improve PyCBC HDF5 table reading
    [[!1543](https://gitlab.com/gwpy/gwpy/-/merge_requests/1543)]

A number of bugs were fixed, and compatibility improved with advanced
versions of the requirements.

(v2.1.5)=
## 2.1.5 - 2022-08-01

Patch release for GWpy 2.1.x.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.5)

### Bug Fixes

-   Fix compatibility with scipy 1.9.0
    [[!1521](https://gitlab.com/gwpy/gwpy/-/merge_requests/1521)]
-   Deprecate the `gwpy.utils.null_context` function
    [[!1523](https://gitlab.com/gwpy/gwpy/-/merge_requests/1523)]

### Other Changes

-   Fix some lint
    [[!1530](https://gitlab.com/gwpy/gwpy/-/merge_requests/1530)]

(v2.1.4)=
## 2.1.4 - 2022-06-27

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.4)

### Bug Fixes

-   handle cases in `EventTable.cluster()` where table is empty or already clustered
    [[!1363](https://gitlab.com/gwpy/gwpy/-/merge_requests/1363)]
-   fix bug in `Series` index comparison
    [[!1428](https://gitlab.com/gwpy/gwpy/-/merge_requests/1428)]
-   fix bug reading ASCII Files
    [[!1474](https://gitlab.com/gwpy/gwpy/-/merge_requests/1474)]
-   resolve `DeprecationWarning` from SciPy 1.8.0
    [[!1484](https://gitlab.com/gwpy/gwpy/-/merge_requests/1484)]
-   fix reading a contiguous set of files with LALFrame
    [[!1487](https://gitlab.com/gwpy/gwpy/-/merge_requests/1487)]
-   migrate GWDataFind interface to new UI
    [[!1499](https://gitlab.com/gwpy/gwpy/-/merge_requests/1499)]
-   update name of Gravity Spy server for glitch tables
    [[!1512](https://gitlab.com/gwpy/gwpy/-/merge_requests/1512)]
-   work around a bug in frameCPP
    [[!1517](https://gitlab.com/gwpy/gwpy/-/merge_requests/1517)]

### Other Changes

-   Python >=3.7
    [[!1441](https://gitlab.com/gwpy/gwpy/-/merge_requests/1441)]
-   h5py >=2.8.0
    [[!1441](https://gitlab.com/gwpy/gwpy/-/merge_requests/1441)]
-   Astropy >=4.0
    [[!1463](https://gitlab.com/gwpy/gwpy/-/merge_requests/1463)]
-   NumPy >=1.16
    [[!1463](https://gitlab.com/gwpy/gwpy/-/merge_requests/1463)]
-   GWDataFind >=1.1.0
    [[!1499](https://gitlab.com/gwpy/gwpy/-/merge_requests/1499)]

(v2.1.3)=
## 2.1.3 - 2021-12-17

Patch release for GWpy 2.1.x.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.3)

### Bug Fixes

-   update test suite to use tmp_path fixture in pytest
    [[!1401](https://gitlab.com/gwpy/gwpy/-/merge_requests/1401)]
-   add support for Python 3.10
    [[!1418](https://gitlab.com/gwpy/gwpy/-/merge_requests/1418)]
-   fix DeprecationWarning emitted in Python 3.10
    [[!1439](https://gitlab.com/gwpy/gwpy/-/merge_requests/1439)]
-   update timeseries/public example
    [[!1438](https://gitlab.com/gwpy/gwpy/-/merge_requests/1438)]
-   decorate test to skip gracefully on network error
    [[!1456](https://gitlab.com/gwpy/gwpy/-/merge_requests/1456)]
-   fix compatibility with python-ligo-lw 1.8.0
    [[!1457](https://gitlab.com/gwpy/gwpy/-/merge_requests/1457)]
-   fix compatibility with scipy 1.8.0
    [[!1458](https://gitlab.com/gwpy/gwpy/-/merge_requests/1458)]

(v2.1.2)=
## 2.1.2 - 2021-11-25

Patch release for GWpy 2.1.x.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.2)

### Deprecations

-   `gwpy.utils.shell.which()` is now deprecated
    [[!1411](https://gitlab.com/gwpy/gwpy/-/merge_requests/1411)]

### Bug Fixes

-   writing a `TimeSeries` with no `name` to GWF now has
    consistent behaviour across all GWF APIs
    [[!1425](https://gitlab.com/gwpy/gwpy/-/merge_requests/1425)]
-   fixed compatibility with Astropy 5.0
    [[!1435](https://gitlab.com/gwpy/gwpy/-/merge_requests/1435)]

(v2.1.1)=
## 2.1.1 - 2021-10-18

Patch release.

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.1)

### Deprecations

-   `gwpy.utils.shell.call()` is now deprecated
    [[!1412](https://gitlab.com/gwpy/gwpy/-/merge_requests/1412)]

### API Changes

-   a `strict` keyword was added to `gwpy.io.cache.sieve()`
    [[!1417](https://gitlab.com/gwpy/gwpy/-/merge_requests/1417)]

### Bug Fixes

-   `Series.crop()` can now handle irregular indices (thanks @mattpitkin!)
    [[!1414](https://gitlab.com/gwpy/gwpy/-/merge_requests/1414)]
-   `TimeSeries.read` no longer checks the `tRange` of an `FrProcData`
    when reading GWF using FrameCPP
    [[!1420](https://gitlab.com/gwpy/gwpy/-/merge_requests/1420)]

(v2.1.0)=
## 2.1.0 - 2021-08-30

[Full details](https://gitlab.com/gwpy/gwpy/-/releases/v2.1.0)

### Deprecations

-   the `gwpy.testing.utils.TemporaryFile` context manager is now deprecated
    [[!1353](https://gitlab.com/gwpy/gwpy/-/merge_requests/1353)]
-   `gwpy.astro.inspiral_range` function renamed to `sensemon_range`
    [[!1293](https://gitlab.com/gwpy/gwpy/-/merge_requests/1293)]

### New Features

-   ROOT I/O now uses `uproot` as the backend
    [[!1263](https://gitlab.com/gwpy/gwpy/-/merge_requests/1263), [#1309](https://github.com/gwpy/gwpy/issues/1309)]
-   `pathlib.Path` objects are now supported everywhere file names
    are passed around (hopefully)
    [[!1362](https://gitlab.com/gwpy/gwpy/-/merge_requests/1362), [#1380](https://github.com/gwpy/gwpy/issues/1380)]
-   `gwpy.signal.filter_design.notch` now supports an `output` keyword
    [[!1347](https://gitlab.com/gwpy/gwpy/-/merge_requests/1347)]
-   `gwpy-plot qtransform` now supports FFT options on the command-line
    [[!1330](https://gitlab.com/gwpy/gwpy/-/merge_requests/1330)]
-   new `--average-method` command-line option for `gwpy-plot`
    [[!1329](https://gitlab.com/gwpy/gwpy/-/merge_requests/1329)]
-   `TimeSeries.rayleigh_spectrum` now supports a `window` keyword
    [[!1285](https://gitlab.com/gwpy/gwpy/-/merge_requests/1285)]

### API Changes

-   passing a `channel` name is now optional when reading files
    in the `hdf5.snax` format
    [[!1275](https://gitlab.com/gwpy/gwpy/-/merge_requests/1275)]
-   the default spectral averaging method is now 'median' (was 'mean')
    [[!1282](https://gitlab.com/gwpy/gwpy/-/merge_requests/1282)]
