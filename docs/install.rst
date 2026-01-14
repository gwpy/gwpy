.. _gwpy-installation:

############
Installation
############

.. image:: https://img.shields.io/pypi/v/gwpy.svg
    :target: https://pypi.org/project/gwpy/
    :alt: GWpy PyPI version badge

.. image:: https://img.shields.io/pypi/pyversions/gwpy.svg
    :alt: Supported Python versions

.. _gwpy-install-quick:

Quick Install
=============

For most users, we recommend installing via conda:

.. tab-set::

    .. tab-item:: Conda (Recommended)

        .. code-block:: bash

           conda install -c conda-forge gwpy

    .. tab-item:: Pip

        .. code-block:: bash

           pip install gwpy

    .. tab-item:: uv

        .. code-block:: bash

           uv pip install gwpy

    .. tab-item:: Development Version

        .. code-block:: bash

           pip install git+https://gitlab.com/gwpy/gwpy.git

.. admonition:: New to Python?
    :class: tip

    We recommend `uv <https://docs.astral.sh/uv/>`,
    an extremely fast Python package and project manager.
    For 99% of operations, just replace `pip install <packages>` with
    `uv pip install <packages>`.

    For environments that need more than just Python packages,
    we recommend `conda-forge <https://conda-forge.org/>`_,
    a community-driven Conda package channel and distribution
    (without the licensing issues of Anaconda).

----

.. _gwpy-install-detailed:

Detailed Installation
=====================

Creating a Virtual Environment
-------------------------------

It's good practice to install GWpy in a virtual environment:

.. tab-set::

    .. tab-item:: Conda

        .. code-block:: bash

           conda create -n gwpy-env -c conda-forge gwpy

    .. tab-item:: uv

        .. code-block:: bash

           uv venv
           uv pip install gwpy

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash
    :caption: Install GWpy from source

    git clone https://gitlab.com/gwpy/gwpy.git
    cd gwpy
    pip install .

For development with editable install:

.. code-block:: bash
    :caption: Install GWpy from source in editable mode

    pip install -e . --group dev

----

.. _gwpy-install-optional:

Optional Dependencies
=====================

GWpy has several optional dependencies for additional functionality:

.. list-table::
    :header-rows: 1
    :widths: 20 30 50

    * - Feature
      - Install command
      - Description
    * - GWF I/O
      - ``pip install gwpy[gwf]``
      - Read/write GWF (gravitational wave frame) files
    * - Authentication
      - ``pip install gwpy[auth]``
      - Access proprietary data with credentials
    * - Advanced SQL
      - ``pip install gwpy[sql]``
      - Enhanced database query capabilities
    * - Astronomical calculations
      - ``pip install gwpy[astro]``
      - Compute inspiral ranges and other astrophysical quantities
    * - Pelican data access
      - ``pip install gwpy[pelican,scitokens]``
      - Read data from Pelican servers
    * - Type checking
      - ``pip install gwpy[typing]``
      - Install type checking dependencies (e.g., ``optype``)

.. note::

    Conda doesn't support optional dependencies, so you may have to
    install them manually, or use pip within your conda environment.

For details on some of the gravitational-wave specific optional dependencies,
see :doc:`/external/index`.

----

.. _gwpy-install-verify:

Verifying Installation
======================

Check that GWpy is installed correctly:

.. code-block:: bash
    :caption: Verify GWpy installation

    python -c "import gwpy; print(gwpy.__version__)"

You should see the version number printed, e.g., ``3.0.0``.

Test with a simple example:

.. code-block:: python
    :caption: Simple GWpy test

    from gwpy.timeseries import TimeSeries
    import numpy as np

    # Create a simple time series
    data = TimeSeries(np.random.random(1000), sample_rate=1)
    print(data)

If this runs without errors, you're all set!

----

.. _gwpy-install-help:

Need Help?
==========

If you're still having trouble:

- Check the `issue tracker <https://gitlab.com/gwpy/gwpy/-/issues>`__ for known problems
- Ask on our `Slack channel <https://gwpy.slack.com>`__
- Open a `new issue <https://gitlab.com/gwpy/gwpy/-/issues/new>`__ with details about your problem
