# vim: ts=4:noet
# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = ../build/sphinx

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

EXAMPLES := $(shell cd ../examples/ && ls */*py)
EXAMPLEDIRS := $(shell cd ../examples/ && ls -d */)
.PRECIOUS: examples/%/index.rst

CLIDIR = ./cli

.PHONY: help clean apidoc html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest

#This is needed with git because git doesn't create a dir if it's empty
$(shell [ -d "_static" ] || mkdir -p _static)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf examples/*/*.rst
	-rm -rf $(CLIDIR)/examples/*.rst
	-rm -rf $(CLIDIR)/examples/*.png
	-rm -rf $(BUILDDIR)
	-rm -rf _generated
	-rm -rf api/
	-rm -f lal*.tag

apidoc:
	sphinx-apidoc -o _generated ../gwpy

html: examples lal.tag lalframe.tag
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# -- API examples -------------------------------------------------------------

.PHONY: examples
examples: $(addprefix examples/,$(EXAMPLES:py=rst))

examples/%/index.rst: ../examples/%/index.rst
	@mkdir -p examples/$*
	@cp ../examples/$*/index.rst $@

.SECONDEXPANSION:
examples/%.rst: $$(@D)/index.rst
examples/%.rst: ex2rst.py examples/index.rst ../examples/%.py $$(@D)/index.rst
	python ex2rst.py ../examples/$*.py $@

# -- extras -------------------------------------------------------------------

%.tag:
	-wget --no-check-certificate http://software.ligo.org/docs/lalsuite/$*/$*.tag -O $@ --quiet
