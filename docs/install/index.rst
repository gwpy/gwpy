.. include:: ../references.txt

.. _gwpy-install:

############
Installation
############

GWpy is currently not packaged for the 'standard' package manager on any operating system, however, it is easily installed via `pip <//pip.pypa.io/>`_.
See the sections below to install dependencies using the system package manager, if appropriate, or skip to :ref:`gwpy-install-pip`.

================
Install on macOS
================

GWpy dependencies are easiest to install using `MacPorts <//www.macports.org/>`_, `follow these instructions <https://www.macports.org/install.php>`_ to install the MacPorts package on your Mac.

The core dependencies can be installed via:

.. code-block:: bash

   sudo port install python27 python_select py27-numpy py27-scipy py27-matplotlib +latex +dvipng texlive-latex-extra py27-astropy glue
   sudo port select --set python python27

Optional extras can be installed via:

.. code-block:: bash

   sudo port install py27-ipython nds2-client kerberos5 py27-pykerberos py27-h5py py27-lal py27-ldas-tools-framecpp

Finally, to install GWpy, skip to :ref:`gwpy-install-pip`.

.. _gwpy-install-linux:

================
Install on Linux
================

The GWpy package has the following build-time dependencies (i.e. required for installation):

* |numpy|_ `>= 1.5`
* |scipy|_ `>= 0.16`
* |matplotlib|_ `>= 1.4.1`
* |astropy|_ `>= 1.2.1`
* |dateutil|_
* |lal|_ `>= 6.18.0`
* |glue|_ `>= 1.53`

You should install these packages on your system using the instructions provided by their authors.

You can also install the following optional dependencies - packages that, when present, enhance the functionality of GWpy:

* |lalframe|_
* |nds2|_

Finally, to install GWpy, skip to :ref:`gwpy-install-pip`.

.. _gwpy-install-pip:

===================
Installing with pip
===================

.. warning::

   GWpy is dependent on |lal|_, which cannot be installed via `pip`, see :ref:`gwpy-install-lal` before continuing to install GWpy.

Once LAL is installed, the easiest way to install GWpy is via `pip <//pip.pypa.io/>`_:

.. code-block:: bash

   pip install gwpy

This will install GWpy and a minimal set of required dependencies that provide basic functionality.

A more complete set of packages that include optional extras can be installed via

.. code-block:: bash

   pip install gwpy[all]

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

