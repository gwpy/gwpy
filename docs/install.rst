.. _pydischarge-install:

############
Installation
############


===============
Installing pyDischarge
===============

.. _pydischarge-install-conda:

-----
Conda
-----

The recommended way of installing pyDischarge is with `Conda <https://conda.io>`__:

.. code-block:: bash

   conda install -c conda-forge pydischarge


.. _pydischarge-install-pip:

---
Pip
---

.. code-block:: bash

    python -m pip install pydischarge


============
Requirements
============

pyDischarge has the following strict requirements:

.. table:: Requirements for pyDischarge
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

pyDischarge also depends on the following other packages for optional features:

.. toctree::
   :glob:

   external/*
