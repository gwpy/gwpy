GWpy is a collaboration‑driven Python package providing tools for studying data from ground‑based gravitational‑wave detectors. It offers a user‑friendly interface to time‑ and frequency‑domain data from the LIGO, Virgo, and KAGRA observatories, with step‑by‑step tutorials.

The GWpy package contains classes and utilities for astrophysical and instrumental analysis. It targets users who want to “do the science” without worrying about low‑level details, through simple, well‑documented APIs and sensible examples.

GWpy builds on, and extends, the functionality of Astropy and the LVK LALSuite libraries (via SWIG Python bindings), combining rich astronomy abstractions with the completeness and speed of mature C99 routines.

<https://gwpy.github.io>

# Release status

[![PyPI version](https://badge.fury.io/py/gwpy.svg)](http://badge.fury.io/py/gwpy)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/gwpy.svg)](https://anaconda.org/conda-forge/gwpy/)

[![License](https://img.shields.io/pypi/l/gwpy.svg)](https://choosealicense.com/licenses/gpl-3.0/)
![Supported Python versions](https://img.shields.io/pypi/pyversions/gwpy.svg)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

# Development status

[![Build status](https://gitlab.com/gwpy/gwpy/badges/main/pipeline.svg)](https://gitlab.com/gwpy/gwpy/-/pipelines/?ref=main)
[![Coverage status](https://gitlab.com/gwpy/gwpy/badges/main/coverage.svg)](https://gitlab.com/gwpy/gwpy/-/pipelines/?ref=main)

# Installation

To install, you can do:

```
conda install -c conda-forge gwpy
```

or

```
python -m pip install gwpy
```

You can test your installation, and its version by

```
python -c "import gwpy; print(gwpy.__version__)"
```

# Quick example

Here is how you can fetch, filter, whiten, and visualize real LIGO data from GWOSC in just a few lines:

```python
from gwpy.timeseries import TimeSeries
from gwosc import datasets
gps = datasets.event_gps("GW150914")
h = TimeSeries.fetch_open_data("H1", gps-4, gps+4, sample_rate=4096)
w = h.bandpass(20, 1024).whiten(4, 2)
spec = w.spectrogram2(fftlength=0.25, overlap=0.20)
spec.plot(norm="log").show()
```

# License

GWpy is released under the GNU General Public License v3.0 or later, see
[here](https://choosealicense.com/licenses/gpl-3.0/) for a description of
this license, or see the
[LICENSE](https://gitlab.com/gwpy/gwpy/-/blob/main/LICENSE) file for the
full text.
