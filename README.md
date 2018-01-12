Orange3 Chemoinformatics
=======================
[![Build Status](https://travis-ci.org/biolab/orange3-imageanalytics.svg?branch=master)](https://travis-ci.org/biolab/orange3-imageanalytics)
[![codecov](https://codecov.io/gh/biolab/orange3-imageanalytics/branch/master/graph/badge.svg)](https://codecov.io/gh/biolab/orange3-imageanalytics)

Orange3 Chemoinformatics is an add-on for the [Orange3](http://orange.biolab.si) data mining suite. It provides extensions for embedding molecule dataset annotated with SMILES through a pre-trained deep neural networks.

Installation
------------
RDKit is a requirement for this widget. To install it run

    sudo apt-get install python-rdkit librdkit1 rdkit-data

To install the add-on from source run

    python setup.py install

To register this add-on with Orange but keep the code in the development directory (do not copy it to
Python's site-packages directory) run

    python setup.py develop

You can also run

    pip install -e .

which is sometimes preferable as you can *pip uninstall* packages later.

Usage
-----

After the installation the widgets from this add-on are registered with Orange. To run Orange from the terminal
use

    orange-canvas

or

    python3 -m Orange.canvas

New widgets are in the toolbox bar under the Image Analytics section.
