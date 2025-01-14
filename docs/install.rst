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
   |dateparser|_       ``>=1.1.4``
   |dqsegdb2|_
   |gwdatafind|_       ``>=1.1.0``
   |gwosc-mod|_        ``>=0.5.3``
   |h5py|_             ``>=3.0.0``
   |igwn-segments|_    ``>=2.0.0``
   |ligotimegps|_      ``>=1.2.1``
   |matplotlib|_       ``>=3.1.0``
   |numpy|_            ``>=1.19.5``
   |dateutil|_
   |scipy|_            ``>=1.2.0``
   |tqdm|_             ``>=4.52.0``
   ==================  ===========================

All of these will be installed using any of the above install methods.

Gwpy also depends on the following other packages for optional features:

.. toctree::
   :glob:

   external/*
