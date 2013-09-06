***************
Installing GWpy
***************

===================
Installing from git
===================

The source code for GWpy is under ``git`` version control, hosted by http://github.com.

You can install the package by first cloning the repository

.. code-block:: bash

    git clone https://github.com/gwpy/gwpy.git

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

============
Dependencies
============

**Build dependencies**

The GWpy package has the following build-time dependencies (i.e. required for installation):

* `astropy <http://astropy.org>`_
* `NumPy <http://www.numpy.org>`_ >= 1.7.1 (inherited dependency from astropy)

GWpy is dependent on the `astropy <http://astropy.org>`_ package for installation

**Runtime dependencies**

Additionally, in order for much of the code to import and run properly, users are required to have the following packages:

* `matplotlib <http://matplotlib.org>`_
* `glue <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_
* `lal <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ and `lalframe <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ (same URL)

=======================
Available installations
=======================

If you are a member of the LIGO Scientific Collaboration, both the GWpy and astropy packages are installed on all shared computing centres.

If you use the ``bash`` shell, you can source the following script to set up the environment for the GWpy package

.. code-block:: bash

    source /home/duncan.macleod/etc/gwpy-env.sh

If anyone wants to write an equivalent shell script for the ``csh`` shell, please e-mail it to `Duncan <duncan.macleod@ligo.org>`_.
