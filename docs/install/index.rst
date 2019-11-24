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

Supported python versions: 2.7, 3.5+.


.. _gwpy-install-debian:

------------
Debian Linux
------------

.. code-block:: bash

    $ apt-get install python3-gwpy

Supported python versions: 2.7 (all), 3.5 (Stretch), 3.6 (Buster),
`click here <https://wiki.ligo.org/Computing/DASWG/SoftwareOnDebian>`__ for
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


============
Requirements
============

GWpy has the following strict requirements:

.. table:: Requirements for GWpy
   :align: left
   :name: requirements-table

   ==================  ===========================
   Name                Constraints
   ==================  ===========================
   |astropy|_          ``>=1.3.0``
   |dqsegdb2|_
   |gwdatafind|_
   |gwosc-mod|_        ``>=0.4.0``
   |h5py|_             ``>=1.3``
   |ligo-segments|_    ``>=1.0.0``
   |ligotimegps|_      ``>=1.2.1``
   |matplotlib|_       ``!=2.1.0,!=2.1.1,>=1.2.0``
   |numpy|_            ``>=1.7.1``
   |dateutil|_
   |scipy|_            ``>=0.12.1``
   |six|_              ``>=1.5``
   |tqdm|_             ``>=4.10.0``
   ==================  ===========================

All of these will be installed using any of the above install methods.

GWpy also depends on the following other packages for optional features:

- :mod:`glue.ligolw`: to read/write :class:`~gwpy.table.EventTable` with
  LIGO_LW XML format (see :ref:`gwpy-table-io-ligolw`)
- |LDAStools.frameCPP|_: to read/write data in GWF format
- |nds2|_: to provide remote data access for `TimeSeries`
  (see :ref:`gwpy-timeseries-remote`)
- |root_numpy|_: to read/write :class:`~gwpy.table.EventTable` with ROOT
  format (see :ref:`gwpy-table-io-root`)
