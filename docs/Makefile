# vim: ts=4:noet
# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -m sphinx
SPHINXPROJ    = gwpy
SOURCEDIR     = .
BUILDDIR      = ../build/sphinx

.PHONY: help html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-rm -rf examples/*/*.rst
	-rm -rf $(CLIDIR)/examples/*.rst
	-rm -rf $(CLIDIR)/examples/*.png
	-rm -rf $(BUILDDIR)
	-rm -rf _generated
	-rm -rf api/
	-rm -f lal*.tag
