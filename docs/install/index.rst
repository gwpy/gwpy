.. _gwpy-install:

############
Installation
############


===============
Installing GWpy
===============

.. _gwpy-install-conda:

-----
Conda
-----

The recommended way of installing GWpy is with `Conda <https://conda.io>`__:

.. code-block:: bash

   $ conda install -c conda-forge gwpy


.. _gwpy-install-pip:

---
Pip
---

.. code-block:: bash

    $ python -m pip install gwpy

Supported python versions: 2.7, 3.4+.


.. _gwpy-install-debian:

------------
Debian Linux
------------

.. code-block:: bash

    $ apt-get install python3-gwpy

Supported python versions: 2.7 (all), 3.4 (Jessie), 3.5 (Stretch), 3.6 (Buster),
`click here <https://wiki.ligo.org/DASWG/SoftwareOnDebian>`__ for
instructions on how to add the required repositories.


.. _gwpy-install-el:

----------------
Scientific Linux
----------------

.. code-block:: bash

    $ yum install python-gwpy

Supported python versions: 2.7,
`click here <https://wiki.ligo.org/DASWG/ScientificLinux>`__ for
instructions on how to add the required yum repositories.


.. _gwpy-install-macports:

--------
Macports
--------

.. code-block:: bash

    $ port install py37-gwpy

Supported python versions: 2.7, 3.6+.


.. _gwpy-install-requirements:

============
Requirements
============

GWpy has the following strict requirements:

- `Python <https://python.org>`__ 2.7, or 3.4 or greater
- |six|_ `>= 1.5.0`
- |dateutil|_
- |enum34|_ (Python 2.7 only)
- |numpy|_ `>= 1.7.1`
- |scipy|_ `>= 0.12.1`
- |astropy|_ `>= 1.1.1`
- |h5py|_ `>= 1.3.0`
- |matplotlib|_ `>= 1.2.0`
- |ligo-segments|_ `>= 1.0.0`
- |tqdm|_ `>= 4.10.0`
- |ligotimegps|_ `>= 1.2.1`

All of these will be installed using any of the above install methods.

GWpy also depends on the following other packages for optional features:

- |dqsegdb|_: to query data from DQSEGDB (see :ref:`gwpy-segments-dqsegdb`)
- |LDAStools.frameCPP|_ or |lalframe|_: to read/write data in GWF format
- |LDAStools.frameCPP|_: to enable data discovery in GWF files
- |lal|_: to power some FFT-based PSD estimation
- |nds2|_: to provide remote data access for `TimeSeries` (see :ref:`gwpy-timeseries-remote`)
- |pycbc|_: to power some FFT-based PSD estimation
- |root_numpy|_: to read/write :class:`~gwpy.table.EventTable` with ROOT format (see :ref:`gwpy-table-io-root`)
