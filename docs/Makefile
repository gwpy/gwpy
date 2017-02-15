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

CLIPROG       = ../bin/gwpy-plot
CLIEXAMPLEDIR = $(BUILDDIR)/cli_examples
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
	-rm -rf examples/*/
	-rm -rf $(BUILDDIR)
	-rm -rf _generated
	-rm -f lal*.tag

apidoc:
	sphinx-apidoc -o _generated ../gwpy

html: cli-examples examples lal.tag lalframe.tag
	@cp -r $(BUILDDIR)/cli_examples $(BUILDDIR)/html
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	@cp -r $(BUILDDIR)/cli_examples $(BUILDDIR)/dirhtml
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/Astropy.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/Astropy.qhc"

devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/Astropy"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/Astropy"
	@echo "# devhelp"

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

.PHONY: examples
examples: $(addprefix examples/,$(EXAMPLES:py=rst))

examples/%/index.rst: ../examples/%/index.rst
	@mkdir -p examples/$*
	@cp ../examples/$*/index.rst $@

examples/%.rst: export INDEX=$(addprefix examples/,$(dir $*))index.rst
examples/%.rst: ex2rst.py ../examples/%.py examples/index.rst
	@$(MAKE) $(INDEX)
	python ex2rst.py ../examples/$*.py $@

.PHONY: cli-examples
cli-examples: $(CLIEXAMPLEDIR)/cli-01.png \
              $(CLIEXAMPLEDIR)/cli-02.png \
              $(CLIEXAMPLEDIR)/cli-ts-01.png \
              $(CLIEXAMPLEDIR)/cli-ts-02.png \
              $(CLIEXAMPLEDIR)/cli-sp-01.png \
              $(CLIEXAMPLEDIR)/cli-sp-02.png \
              $(CLIEXAMPLEDIR)/cli-cohg-01.png \
              $(CLIEXAMPLEDIR)/cli-spg-01.png \
              $(CLIEXAMPLEDIR)/cli-coh-01.png

$(CLIEXAMPLEDIR):
	@echo "building cli examples"
	@mkdir -p $(CLIEXAMPLEDIR)/

$(CLIEXAMPLEDIR)/cli-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ];  then \
		echo "cli-01.png"; \
		$(CLIPROG) timeseries --chan L1:OAF-CAL_DARM_DQ \
			--start 1102766651 --out $(CLIEXAMPLEDIR)/cli-01.png; \
	fi

$(CLIEXAMPLEDIR)/cli-02.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-02.png"; \
		$(CLIPROG) spectrum --chan L1:OAF-CAL_DARM_DQ --start 1102766451 \
			--start 1102766651  --out $(CLIEXAMPLEDIR)/cli-02.png; \
	fi

$(CLIEXAMPLEDIR)/cli-ts-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-ts-01"; \
		$(CLIPROG) timeseries --chan L1:OAF-CAL_DARM_DQ --start 1107936100 \
			--xmin 1107936100.2 --xmax 1107936100.25 --epoch 0.225 \
			--duration 1 --out $(CLIEXAMPLEDIR)/cli-ts-01.png; \
	fi

$(CLIEXAMPLEDIR)/cli-ts-02.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-ts-02"; \
		$(CLIPROG) timeseries --chan LVE-EX:X3_810BTORR.mean,m-trend \
			--start 1072539420 --dur 36000 --logy \
			--out $(CLIEXAMPLEDIR)/cli-ts-02.png; \
	fi

$(CLIEXAMPLEDIR)/cli-sp-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-sp-01"; \
		$(CLIPROG) spectrum --chan L1:OAF-CAL_DARM_DQ --start 1107936067 \
			--duration 120  --secpfft 10 --out $(CLIEXAMPLEDIR)/cli-sp-01.png; \
	fi

$(CLIEXAMPLEDIR)/cli-sp-02.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-sp-02"; \
		$(CLIPROG) spectrum --chan L1:OAF-CAL_DARM_DQ --start 1107936067 \
			--duration 120 --out $(CLIEXAMPLEDIR)/cli-sp-02.png --secpfft 10.000 \
			--overlap 0.90 --nologf --fmin 500 --fmax 700; \
	fi

$(CLIEXAMPLEDIR)/cli-coh-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-coh-01"; \
		$(CLIPROG) coherence --chan L1:LSC-SRCL_IN1_DQ L1:LSC-CARM_IN1_DQ \
			--start 1107820816  --duration 900  \
			--out $(CLIEXAMPLEDIR)/cli-coh-01.png \
			--ref L1:LSC-SRCL_IN1_DQ --secpfft 8.0 --overlap 0.50 ;\
	fi

$(CLIEXAMPLEDIR)/cli-cohg-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-cohg-01"; \
		$(CLIPROG) coherencegram --chan L1:LSC-SRCL_IN1_DQ L1:LSC-CARM_IN1_DQ \
			--start 1107820816 --duration 900 \
			--out $(CLIEXAMPLEDIR)/cli-cohg-01.png \
			--ref L1:LSC-SRCL_IN1_DQ --secpfft 0.500 --overlap 0.90 ;\
	fi

$(CLIEXAMPLEDIR)/cli-spg-01.png: $(CLIEXAMPLEDIR)
	@if [ ! -e $@ ]; then \
		echo "cli-spg-01"; \
		$(CLIPROG) spectrogram --chan L1:LSC-SRCL_IN1_DQ --start 1107937307 --dur 240 \
		    --out $(CLIEXAMPLEDIR)/cli-spg-01.png; \
	fi

%.tag:
	-wget --no-check-certificate http://software.ligo.org/docs/lalsuite/$*/$*.tag -O $@ --quiet
