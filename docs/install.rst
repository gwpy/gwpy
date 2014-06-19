***************
Installing GWpy
***************

============
Dependencies
============

**Build dependencies**

The GWpy package has the following build-time dependencies (i.e. required for installation):

* `glue <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_
* `python-dateutil <https://pypi.python.org/pypi/python-dateutil/>`_
* `NumPy <http://www.numpy.org>`_ >= 1.5
* `SciPy <http://www.scipy.org>`_
* `Matplotlib <http://matplotlib.org>`_ >= 1.3.0
* `Astropy <http://astropy.org>`_ >= 0.3

.. note::

   The `GLUE <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_ package isn't available through PyPI, meaning you will have to install it manually from the link.

**Runtime dependencies**

Additionally, in order for much of the code to import and run properly, users are required to have the following packages:

* `lal <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ and `lalframe <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ (same URL)
* `NDS2 <https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html>`_ (including SWIG-wrappings for python)

===================
Installing with pip
===================

The easiest way to install GWpy is to use `pip <https://pip.pypa.io/en/latest/index.html>`_:

.. code-block:: bash

   pip install --pre --user gwpy

Currently, GWpy is only in a pre-release phase, and so ``pip`` requires the ``--pre`` option to install GWpy.
The ``--user`` option tells the installer to copy codes into the standard user library path, on linux machines this is

.. code-block:: bash

    ~/.local/lib

while on Mac OS this is

.. code-block:: bash

    ~/Library/Python/X.Y/lib

where ``X.Y`` is the python major and minor version numbers, e.g. ``2.7``.
For either operating system, python will automatically know about these directories, so you don't have to fiddle with any environment variables.

.. warning::

   GWpy is still under major version ``0``, meaning a completely stable state has no been reached. Until that time, backwards-incompatible changes may be made without much warning, but developers will strive to keep these events to a minimum.

=========================================
Installing the latest development version
=========================================

Any user can install the latest development version of GWpy by directing ``pip`` to the GWpy GitHub repository:

.. code-block:: bash

   pip install --user git+https://github.com/gwpy/gwpy

.. warning::

   The latest developments are not guaranteed to be free of bugs, and so you should only install from GitHub if you really need to.

======================
Cloning the repository
======================

The source code for GWpy is under ``git`` version control, hosted by http://github.com.
You can clone the repository from the Terminal as follows:

.. code-block:: bash

    git clone https://github.com/gwpy/gwpy.git

You can then, if you wish, install the package by running the ``setup.py`` script as follows:

.. code-block:: bash

    cd gwpy
    python setup.py install --user

.. warning::

   Users have reported an issue with installation on Mac OS using the anaconda python distribution. The GWpy install might raise the following exception:

   .. code::

      ValueError: unknown locale: UTF-8

   In this instance, you can resolve the issue by setting the following environment variables in your bash shell:

   .. code:: bash

      export LANG=en_US.UTF-8
      export LC_ALL=en_US.UTF-8

   or in csh:

   .. code:: csh

      setenv LANG en_US.UTF-8
      setenv LC_ALL en_US.UTF-8

=======================
Available installations
=======================

If you are a member of the LIGO Scientific Collaboration, both the GWpy and astropy packages are installed on all shared computing centres.

If you use the ``bash`` shell, you can source the following script to set up the environment for the GWpy package

.. code-block:: bash

    source /home/detchar/opt/gwpysoft/etc/gwpy-user-env.sh

If anyone wants to write an equivalent shell script for the ``csh`` shell, please e-mail it to `Duncan <duncan.macleod@ligo.org>`_.
