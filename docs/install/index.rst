.. include:: ../references.txt

.. _gwpy-install:

############
Installation
############


.. _gwpy-install-pip:

=============
Quick install
=============

GWpy can be installed using `pip <//pip.pypa.io/>`_:

.. code-block:: bash

   pip install gwpy

This will install GWpy itself and all required dependencies for minimal functionality.

Those dependencies are:

- |dateutil|_
- |numpy|_
- |scipy|_
- |matplotlib|_
- |astropy|_
- |glue|_
- |ligotimegps|_ (if |lal|_ is not already installed)

See the GWpy ``setup.py`` file for minimum version requirements for each of those packages.

.. _gwpy-install-extras:

=================
Installing Extras
=================

Additional (optional) functionality can be installed by specifying one or more of the extra group names, as follows

+--------------+-----------------------------+--------------------------------+
| Group name   | Purpose                     | Packages included              |
+==============+=============================+================================+
| ``hdf5``     | Reading/writing HDF5 files  | |h5py|_                        |
|              |                             |                                |
|              | Querying for open-access    |                                |
|              | LIGO data                   |                                |
+--------------+-----------------------------+--------------------------------+
| ``root``     | Reading/writing ROOT files  | |root_numpy|_                  |
+--------------+-----------------------------+--------------------------------+
| ``segments`` | Querying for LIGO           | |dqsegdb|_                     |
|              | operations and data-quality |                                |
|              | segments                    |                                |
|              | (requires LIGO.ORG          |                                |
|              | credentials)                |                                |
+--------------+-----------------------------+--------------------------------+
| ``hacr``     | Querying for HACR event     | |MySQLdb|_                     |
|              | files                       |                                |
+--------------+-----------------------------+--------------------------------+
| ``docs``     | Generating package          | |sphinx|_, |numpydoc|_,        |
|              | documentation               | |sphinx-bootstrap-theme|_,     |
|              |                             | |sphinxcontrib-programoutput|_ |
+--------------+-----------------------------+--------------------------------+
| ``all``      | All of the above extras     | All of the above               |
+--------------+-----------------------------+--------------------------------+

Any of these extras can be installed with `pip`:

.. code-block:: bash

   pip install gwpy[all]

.. _gwpy-install-non-python-extras:

=================
Non-python extras
=================

GWpy functionality can also be extended with a number of non-python extensions, most of which provide python bindings via `SWIG <//swig.org>`_:

=====================  =====================================
Package                Purpose
=====================  =====================================
|LDAStools.frameCPP|_  Reading/writing GWF files
|lal|_                 Signal-processing
|lalframe|_            Reading/writing GWF files
|nds2|_                Querying for LIGO data over a network
=====================  =====================================

Each of the above is provided by the LIGO Scientific Collaboration, and can be installed using the recommended package manager for your operating system. See below for useful links provided by the LSC Computing group:

==============  ================  ===========================================
OS              Package manager   Useful link
==============  ================  ===========================================
macOS           MacPorts          https://wiki.ligo.org/DASWG/MacPorts
Debian/Ubuntu   Apt               https://wiki.ligo.org/DASWG/DebianJessie
Red Hat/Fedora  Yum               https://wiki.ligo.org/DASWG/ScientificLinux
==============  ================  ===========================================

.. _gwpy-install-available:

#############################################################
Available installations for the LIGO Scientific Collaboration
#############################################################

If you are a member of the LIGO Scientific Collaboration, a `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ is available for you to use on the LIGO Data Grid, providing an isolated environment including GWpy and its dependencies.

How you enter this environment depends on which shell you are using:

**Bash**

.. code-block:: bash

   . ~detchar/opt/gwpysoft/bin/activate

**Csh**

.. code-block:: csh

   . ~detchar/opt/gwpysoft/bin/activate.csh

In either case, once you are finished with your work, if you want to return to your original environment, you can `deactivate` the virtualenv:

.. code-block:: bash

   deactivate

