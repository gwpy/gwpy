***************
Installing GWpy
***************

===================
Installing from git
===================

The source code for GWpy is under ``git`` version control, hosted by http://github.com.

You can install the package by first cloning the repository

.. code-block:: bash

    git clone https://github.com/duncanmmacleod/gwpy.git

and then running the ``setup.py`` script as follows:

.. code-block:: bash

    cd gwpy
    python setup.py install --user

The ``--user`` option tells the installer to copy codes into the standard user library paths, on linux machines this is

.. code-block:: bash

    ~/.local/lib

while on Mac OS this is

.. code-block:: bash

    ~/Library/Python/X.Y/lib

where ``X.Y`` is the python major and minor version numbers, e.g. ``2.7``. In either case, python will autmatically know about these directories, so you don't have to fiddle with any environment variables.

============
Dependencies
============

The GWpy package is dependent on `astropy <http://astropy.org>`_ for a number of the core classes, and on the `LIGO Algorithm Library (LAL) <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ for a lot of core functionality associated with the classes.

Additionally, the `gwpy.io` module is dependent on the `Grid LIGO User Environment (GLUE) <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_ package for connections to the segment database infrastructure, amongst other things.

`numpy` version 1.4 or greater is required for installation. This is an inherited requirement from astropy.

=======================
Available installations
=======================

If you are a member of the LIGO Scientific Collaboration, both the GWpy and astropy packages are installed on all shared computing centres.

If you use the ``bash`` shell, you can source the following script to set up the environment for the GWpy package

.. code-block:: bash

    source /home/duncan.macleod/etc/gwpy-env.sh

If anyone wants to write an equivalent shell script for the ``csh`` shell, please e-mail it to `Duncan <duncan.macleod@ligo.org>`_.
