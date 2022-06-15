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

   conda install -c conda-forge gwpy


.. _gwpy-install-pip:

---
Pip
---

.. code-block:: bash

    python -m pip install gwpy


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
   |astropy|_          ``>=4.0``
   |dqsegdb2|_
   |gwdatafind|_       ``>=1.1.0``
   |gwosc-mod|_        ``>=0.5.3``
   |h5py|_             ``>=2.8.0``
   |ligo-segments|_    ``>=1.0.0``
   |ligotimegps|_      ``>=1.2.1``
   |matplotlib|_       ``>=3.1.0``
   |numpy|_            ``>=1.16.0``
   |dateutil|_
   |scipy|_            ``>=1.2.0``
   |tqdm|_             ``>=4.10.0``
   ==================  ===========================

All of these will be installed using any of the above install methods.

GWpy also depends on the following other packages for optional features:

- |python-ligo-lw| to read/write :class:`~gwpy.table.EventTable` with
  LIGO_LW XML format (see :ref:`gwpy-table-io-ligolw`)
- |LDAStools.frameCPP|_: to read/write data in GWF format
- |nds2|_: to provide remote data access for `TimeSeries`
  (see :ref:`gwpy-timeseries-remote`)
- |uproot|_: to read/write :class:`~gwpy.table.EventTable` with ROOT
  format (see :ref:`gwpy-table-io-root`)
