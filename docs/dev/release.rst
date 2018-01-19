.. _gwpy-dev-release:

####################
Publishing a release
####################

This page describes the steps required to author a release of GWpy.

Notes:

* gwpy uses the `git flow <https://github.com/nvie/gitflow>`__ branching model
  for releases
* all release numbers must follow `Semantic Versioning 2 <segmver.org>`__ and
  include major, minor, and patch numbers, e.g. ``1.0.0`` rather than
  ``1.0`` or just ``1``

============
Step-by-step
============

#. **Create a release branch using git flow**

   .. code-block:: bash

      $ git flow release start 1.0.0

   and then ``publish`` it, allowing CI to run, and others to contribute:

   .. code-block:: bash

      $ git flow release publish 1.0.0

#. **Wait patiently for the continuous integration to finish**

#. **Announce the release** and ask for final contributions

#. **Finalise the release and push**

   .. code-block:: bash

      $ git flow release finish 1.0.0
      $ git push origin master
      $ git push origin --tags

   .. note::

      The ``git flow release finish`` command will open two prompts, one
      to merge the release branch into `master`, just leave that as is. The
      second prompt is the tag message, please complete this to include the
      release notes for this release.

#. **Draft a release on GitHub**

   * Go to https://github.com/gwpy/gwpy/releases/new
   * Use ``v1.0.0`` as the *Tag version*
   * Use 1.0.0 as the *Release title*
   * Copy the tag message into the text box to serve as release notes

#. **Publish the release documentation**

   This is done from the LDAS computing centre at Caltech:

   .. code-block:: bash

      $ cd /home/duncan.macleod/gwpy-nightly-build/
      $ bash release-build.sh 1.0.0

   Once that is complete (~20 minutes), a few manual updates must be made

   .. code-block:: bash

      $ cd /home/duncan.macleod/gwpy-nightly-build/gwpy.github.io/docs
      $ unlink stable && ln -s 1.0.0 stable
      $ sed -i 's/0.9.9/1.0.0/g' index.html

   The final command should be modified to replace the previous release ID
   with the current one.

   Then:

   .. code-block:: bash

      $ git commit --gpg-sign --message="1.0.0: release docs"
      $ git push  # <- this step needs an SSH key

   It should take ~5 minutes for the release documentation to actually
   appear on https://gwpy.github.io/docs/

====================================
Distributing the new release package
====================================

Package distributions for Debian, RHEL, and Macports, is still done manually:

Debian/RHEL
-----------

* Upload the source tarball to software.ligo.org
* Email daswg+announce@ligo.org to announce the new release and the presence of the tarball on http://software.ligo.org.

Macports
--------

* Generate an updated ``Portfile``

  .. code-block:: bash

     $ python setup.py port

* Open a new Pull Request on https://github.com/macports/macports-ports that
  replaces the old ``/python/py-gwpy/Portfile`` with this new version.

==============
Linked updates
==============

PyPI
----

Finishing the release and pushing the tags to ``origin`` will trigger a new
CI run on https://travis-ci.org, which will automatically deploy the new
release tarball to https://pypi.python.org and publish the release there.

Zenodo
------

Creating a new release on GitHub will automatically trigger a new DOI on
https://zenodo.org.
