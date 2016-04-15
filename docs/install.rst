.. _install:

***************
Installing GWpy
***************

GWpy can be installed on the following platforms

- :ref:`mac-os-x`
- :ref:`linux`


.. _mac-os-x:

========
Mac OS X
========

The recommended way to install GWpy on Mac OS X is to install the dependencies via `MacPorts <https://www.macports.org>`_.

`Follow these instructions <https://www.macports.org/install.php>`_ to install the MacPorts package on your Mac, then execute the following command to install the required dependencies for GWpy:

.. code-block:: bash

   sudo port install python27 python_select py27-numpy py27-scipy py27-matplotlib +latex +dvipng texlive-latex-extra py27-astropy glue
   sudo port select --set python python27

You can also run the following to install a number of optional dependencies - packages that, when present, enhance the functionality of GWpy:

.. code-block:: bash

   sudo port install py27-ipython nds2-client kerberos5 py27-pykerberos py27-h5py lal lalframe

Once you have at least the required dependencies installed, you can :ref:`install GWpy using pip <install-pip>`.

.. _linux:

=====
Linux
=====

The GWpy package has the following build-time dependencies (i.e. required for installation):

* `glue <https://www.lsc-group.phys.uwm.edu/daswg/projects/glue.html>`_ >= 1.48
* `python-dateutil <https://pypi.python.org/pypi/python-dateutil/>`_
* `NumPy <http://www.numpy.org>`_ >= 1.5
* `SciPy <http://www.scipy.org>`_ >= 0.16
* `Matplotlib <http://matplotlib.org>`_ >= 1.3.0
* `Astropy <http://astropy.org>`_ >= 1.0

You should install these packages on your system using the instructions provided by their authors.

You can also install the following optional dependencies - packages that, when present, enhance the functionality of GWpy:

* `lal <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ and `lalframe <https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_ (same URL)
* `NDS2 <https://www.lsc-group.phys.uwm.edu/daswg/projects/nds-client.html>`_ (including SWIG-wrappings for python)

Once you have at least the required dependencies installed, you can :ref:`install GWpy using pip <install-pip>`.

.. _install-pip:

===
Pip
===

The easiest way to install GWpy is to use `pip <https://pip.pypa.io/en/latest/index.html>`_. You can install pip very easily with MacPorts and most linux-based package managers.

Once you have pip installed:

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

======
GitHub
======

Any user can install the latest development version of GWpy by directing ``pip`` to the GWpy GitHub repository:

.. code-block:: bash

   pip install --user git+https://github.com/gwpy/gwpy

.. warning::

   The latest developments are not guaranteed to be free of bugs, and so you should only install from GitHub if you really need to.

======
Source
======

The source code for GWpy is under ``git`` version control, hosted by http://github.com.
You can clone the repository from the Terminal as follows:

.. code-block:: bash

    git clone https://github.com/gwpy/gwpy.git

You can then, if you wish, install the package by running the ``setup.py`` script as follows:

.. code-block:: bash

    cd gwpy
    pip install -r requirements.txt
    pip install .

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

***********************
Available installations
***********************

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

