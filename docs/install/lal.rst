.. _gwpy-install-lal:

##############
Installing LAL
##############

The LIGO Algorithm Library is a C-language library with optional python bindings built using SWIG.
LAL is officially supported on the latest stable releases of Debian and Red-hat Linux, with best-effort support for macOS.
See the relevant section below for the best instructions on installing LAL on your system, or from source (e.g. for an isolated virtualenv)

- :ref:`gwpy-install-lal-debian`
- :ref:`gwpy-install-lal-sl`
- :ref:`gwpy-install-lal-macos`
- :ref:`gwpy-install-lal-source`

.. _gwpy-install-lal-debian:

======
Debian
======

**Official instructions can be found at** https://wiki.ligo.org/DASWG/SoftwareDownloads

To install on Debian 7 (codename 'jessie'), first add the LSCSoft repository to your system by adding creating ``/etc/apt/sources.list/lscsoft.list`` with the following contents:

.. code-block:: none

   deb http://software.ligo.org/lscsoft/debian jessie contrib
   deb-src http://software.ligo.org/lscsoft/debian jessie contrib

Then update your package lists as follows:

.. code-block:: bash

   $ apt-get update
   $ apt-get install lscsoft-archive-keyring

Finally, you can install LAL for python2 via

.. code-block:: bash

   $ apt-get install lal-python

OR, for python3:

.. code-block:: bash

   $ apt-get install lal-python3


.. _gwpy-install-lal-sl:

================
Scientific linux
================

**Official instructions can be found at** https://wiki.ligo.org/DASWG/SoftwareDownloads

To install on Scientific Linux 7, firat download and install the RPM configuration for the LSCSoft repository:

.. code-block:: bash

   $ http://software.ligo.org/lscsoft/scientific/7.2/x86_64/production/lscsoft-production-config-1.3-1.el7.noarch.rpm
   $ rpm -ivh lscsoft-production-config-1.3-1.el7.noarch.rpm
   $ yum clean all
   $ yum makecache

Finally, you can install LAL (for python2 or python3) via

.. code-block:: bash

   $ yum install lal-python


.. _gwpy-install-lal-macos:

=====
macOS
=====

macOS support is provided via `MacPorts <//www.macports.org>`_.
To install LAL, simply install the ``pyXY-lal`` port based on your ``X.Y`` version of Python.
For example, for python 2.7, simply run

.. code-block:: bash

   $ sudo port install py27-lal

.. _gwpy-install-lal-source:

============
Source build
============

.. note::

   Building LAL from source requires numpy to be present on your system, either
   install numpy using your sytem package manager, or via `pip`:

   .. code-block:: bash

      $ pip install numpy

To build LAL from source, first identify the latest release tarball by visiting `http://software.ligo.org/lscsoft/source/lalsuite/ <http://software.ligo.org/lscsoft/source/lalsuite/?C=M;O=D>`_ and identifying the most recent release tarball of the form ``lal-X.Y.Z.tar.xz``.
Then you can download and install this package via

.. code-block:: bash

   $ LAL_VERSION="6.18.0"  # change as appropriate
   $ LAL_INSTALL_PREFIX="${VIRTUAL_ENV}"  # change as appropriate

   $ builddir=`mktemp -d`
   $ cd $builddir
   $ wget http://software.ligo.org/lscsoft/source/lalsuite/lal-${LAL_VERSION}.tar.xz
   $ tar -xf lal-${LAL_VERSION}.tar.xz
   $ cd lal-${LAL_VERSION}
   $ ./configure --prefix ${LAL_PREFIX} --quiet --enable-swig-python
   $ make
   $ make install
   $ cd
   $ rm -rf ${builddir}
