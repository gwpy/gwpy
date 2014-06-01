***************
Installing GWpy
***************

======================
Installing from GitHub
======================

At this time GWpy isn't mature enough to have a released version, but that will come soon, which will make installing it a lot easier.

In the mean time, it is easiest to install GWpy using the `pip <https://pip.pypa.io/en/latest/index.html>`_ installer:

.. code-block:: bash

   pip install --user git+https://github.com/gwpy/gwpy

The ``--user`` option tells the installer to copy codes into the standard user library paths, on linux machines this is

.. code-block:: bash

    ~/.local/lib

while on Mac OS this is

.. code-block:: bash

    ~/Library/Python/X.Y/lib

where ``X.Y`` is the python major and minor version numbers, e.g. ``2.7``. In either case, python will autmatically know about these directories, so you don't have to fiddle with any environment variables.

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

============
Dependencies
============

**Build dependencies**

The GWpy package has the following build-time dependencies (i.e. required for installation):

* `glue <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_
* `python-dateutil <https://pypi.python.org/pypi/python-dateutil/>`_
* `NumPy <http://www.numpy.org>`_ >= 1.5
* `matplotlib <http://matplotlib.org>`_ >= 1.3.0
* `astropy <http://astropy.org>`_ >= 0.3

.. note::

   The `GLUE <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_ package isn't available through PyPI, meaning you will have to install it manually from the link.

**Runtime dependencies**

Additionally, in order for much of the code to import and run properly, users are required to have the following packages:

* `lal <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ and `lalframe <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ (same URL)
* `NDS2 <https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html>`_ (including SWIG-wrappings for python)

=======================
Available installations
=======================

If you are a member of the LIGO Scientific Collaboration, both the GWpy and astropy packages are installed on all shared computing centres.

If you use the ``bash`` shell, you can source the following script to set up the environment for the GWpy package

.. code-block:: bash

    source /home/detchar/opt/gwpysoft/etc/gwpy-user-env.sh

If anyone wants to write an equivalent shell script for the ``csh`` shell, please e-mail it to `Duncan <duncan.macleod@ligo.org>`_.
