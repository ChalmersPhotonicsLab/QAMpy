BUILDDIR = build
HTMLDIR = $(BUILDDIR)/docs/html
PDFDIR = $(BUILDDIR)/docs/pdf
SOURCEDIR = docs/

sdist:
	python -m build --sdist

wheel:
	python -m build --wheel

docs-html:
	sphinx-build -b html $(SOURCEDIR) $(HTMLDIR)

docs: docs-html

build:
	python -m build 

install: 
	pip install .

develop:
	pip install -e .

uninstall:
	pip uninstall qampy

clean:
	rm -rf build/
	rm -f qampy/core/*so
	rm -f qampy/core/equalisation/*so

