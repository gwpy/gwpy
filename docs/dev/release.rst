.. _gwpy-dev-release:

####################
Publishing a release
####################

This page describes the steps required to author a release of GWpy.

Notes:

* gwpy uses the
  `stable mainline <https://www.bitsnbites.eu/a-stable-mainline-branching-model-for-git/>`_
  branching model for releases
* all release numbers must follow `Semantic Versioning 2 <segmver.org>`_ and
  include major, minor, and patch numbers, e.g. ``X.Y.Z`` rather than
  ``1.0`` or just ``1``

============
Step-by-step
============

#. **If this is a bug-fix release, just check out that branch**:

    .. code-block:: bash

        git checkout release/X.Y.x

#. **Update the copyright**:

    .. code-block:: bash

        python -c "from setup_utils import update_all_copyright; update_all_copyright()"
        git commit -S -m "Updated copyright for release" .

#. **Publish the release**, allowing CI to run, and others to see it:

    .. code-block:: bash

        git push -u origin main

    for major/minor releases, or

    .. code-block:: bash

        git push -u origin release/X.Y.x

    for bug-fix releases

#. **Wait patiently for the continuous integration to finish**

#. **Announce the release** and ask for final contributions

#. **Tag the release**:

    .. code-block:: bash

        git tag --sign vX.Y.Z

#. **Create a maintenance branch** (major/minor releases only):

    .. code-block:: bash

        git branch release/X.Y.x

#. **Publish everything**:

    .. code-block:: bash

        # push maintenance branch
        git push --signed=if-asked origin release/X.Y.x
        # push main branch
        git push --signed=if-asked origin main
        # push new tag
        git push --signed=if-asked origin vX.Y.Z

#. **Draft a release on GitHub**

    * Go to https://github.com/gwpy/gwpy/releases/new
    * Use ``vX.Y.Z`` as the *Tag version*
    * Use X.Y.Z as the *Release title*
    * Copy the tag message into the text box to serve as release notes

#. **Publish the release documentation**

    This is done from the LDAS computing centre at Caltech:

    .. code-block:: bash

        cd /home/duncan.macleod/gwpy-nightly-build/
        bash release-build.sh X.Y.Z

    Once that is complete (~20 minutes), a few manual updates must be made:

    .. code-block:: bash

        cd /home/duncan.macleod/gwpy-nightly-build/gwpy.github.io/docs
        unlink stable && ln -s X.Y.Z stable
        sed -i 's/0.9.9/X.Y.Z/g' index.html

    The final command should be modified to replace the previous release ID
    with the current one.

    Then:

    .. code-block:: bash

        git commit --gpg-sign --message="X.Y.Z: release docs"
        git push --signed=if-asked  # <- this step needs an SSH key

    It should take ~5 minutes for the release documentation to actually
    appear on https://gwpy.github.io/docs/

====================================
Distributing the new release package
====================================

Package distributions for PyPI, Conda, Debian, and RHEL are done manually:

PyPI
----

To create a new release on PyPI:

.. code-block:: bash

    rm -rf dist/
    git checkout vX.Y.Z
    python -m build
    python -m twine upload --sign dist/gwpy-*

Conda
-----

Once the PyPI upload has completed, the conda-forge bot will automatically
open a pull request to `conda-forge/gwpy-feedstock
<https://github.com/conda-forge/gwpy-feedstock.git>`_.
Just double-check that the dependencies and tests are up-to-date, then
merge.

Debian/RHEL
-----------

* Upload the source tarball to software.ligo.org
* Open a new request to `sccb/requests <https://git.ligo.org/sccb/requests/>`_
  to announce the new release and request package build and deployment.
