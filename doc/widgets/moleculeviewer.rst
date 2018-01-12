Molecule Viewer
============

.. figure:: icons/MoleculeViewer.png

Displays images that come with a data set.

Signals
-------

**Inputs**:

-  **Data**

   A data set with molecules annotated with SMILES.

**Outputs**:

-  **Data**

   Images that come with the data.

Description
-----------

The **Molecule Viewer** widget can display the 2D structure from a data set of molecules annotated with SMILES.


1. Information on the data set
2. Select the column with image data (links).
3. Select the column with image titles.
4. Zoom in or out.
5. Saves the visualization in a file.
6. Tick the box on the left to commit changes automatically.
   Alternatively, click *Send*.


Examples
--------

A very simple way to use this widget is to connect the :doc:`File <../data/file>` widget with
**Molecule Viewer** and see all the molecules that come with your data set.

Alternatively, you can visualize only selected instances, as shown in the
example below.
