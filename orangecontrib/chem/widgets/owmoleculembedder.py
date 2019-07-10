from collections import OrderedDict
import numpy as np
import json

import os.path
import pathlib

from AnyQt.QtWidgets import QFormLayout
from AnyQt.QtCore import Qt

from Orange.data import ContinuousVariable, Domain
from Orange.data.table import Table
from Orange.data.io import Compression, FileFormat, TabReader, CSVReader, PickleReader
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting, DomainContextHandler
from Orange.widgets.utils import filedialogs
from Orange.widgets.utils.itemmodels import VariableListModel, DomainModel
from Orange.widgets.widget import Input, Output
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit

from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

from orangecontrib.chem.preprocess import encoder, moleculeembedder

AUTOENCODER = "Autoencoder"
CIRCULAR = "Circular"

EMBEDDERS = OrderedDict({
    "Topological": Chem.RDKFingerprint,
    CIRCULAR: Chem.AllChem.GetMorganFingerprintAsBitVect,
    "MACCS": Chem.MACCSkeys.FingerprintMol,
    AUTOENCODER: encoder.encoder, }
)

CIRCLE_RAD = 5
MAXLEN = 120
CHARSET = OrderedDict(
    [(" ", 0), ("#", 1), ("(", 2), (")", 3), ("+", 4), ("-", 5), ("/", 6), ("1", 7), ("2", 8), ("3", 9), ("4", 10),
     ("5", 11), ("6", 12), ("7", 13), ("8", 14), ("=", 15), ("@", 16), ("B", 17), ("C", 18), ("F", 19), ("H", 20),
     ("I", 21), ("N", 22), ("O", 23), ("P", 24), ("S", 25), ("[", 26), ("\\", 27), ("]", 28), ("c", 29), ("l", 30),
     ("n", 31), ("o", 32), ("r", 33), ("s", 34), ])


class OWMoleculeEmbedder(widget.OWWidget):
    name = "Molecule Embedder"
    description = "Embedding of molecule in SMILES notation."
    icon = "../widgets/icons/category.svg"
    priority = 150

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        embedded_smiles = Output("Embedded molecules", Table, default=True)
        skipped_smiles = Output("Skipped molecules", Table)

    class Error(widget.OWWidget.Error):
        no_instances = widget.Msg("At least 1 data instances are required")
        no_string_att = widget.Msg("At least 1 string atttribute is required")

    want_main_area = False
    resizing_enabled = False

    settingsHandler = DomainContextHandler()
    attribute_id = Setting(default=0)
    embedder_id = Setting(default=0)

    auto_commit = Setting(default=True)

    def __init__(self):
        super().__init__()
        self.data = None
        self.embedder = ""

        self.basename = ""
        self.type_ext = ""
        self.compress_ext = ""
        self.writer = None

        self.id_smiles_attr = 0
        self._embedder = ''

        self._embedders = list(EMBEDDERS.keys())
        self._smiles_attr = ''

        form = QFormLayout()
        box = gui.vBox(self.controlArea, "Options")

        gui.comboBox(
            box, self, "_smiles_attr", sendSelectedValue=True,
            callback=self._update_options,
        )
        form.addRow(
            "SMILES attribute: ",
            self.controls._smiles_attr
        )
        gui.comboBox(
            box, self, "_embedder", sendSelectedValue=True,
            callback=self._update_options,
            items=self._embedders,
        )
        form.addRow(
            "Embedder: ",
            self.controls._embedder
        )
        box.layout().addLayout(form)

        gui.auto_commit(self.controlArea, self, "auto_commit", "Apply",
                        callback=self.commit,
                        checkbox_label="Apply automatically")

        self.adjustSize()

    @Inputs.data
    def dataset(self, data):
        self.clear()
        self.data = data
        if self.data:
            # error check
            if len(self.data) == 0:
                self.Error.no_instances()
                return

            self._smiles_attrs = moleculeembedder.filter_string_attributes(data)
            if not self._smiles_attrs:
                self.Error.no_string_att()
                return

            # update selection
            self.controls._smiles_attr.setModel(VariableListModel(self._smiles_attrs))
            if self._smiles_attr == '' or self._smiles_attr is None:
                self._smiles_attr = self._smiles_attrs[0]

            if self._embedder == '' or self._embedder is None:
                self._embedder = self._embedders[0]

            self.commit()

    def clear(self):
        self.Outputs.embedded_smiles.send(None)
        self.Outputs.skipped_smiles.send(None)

        self.Error.no_instances.clear()
        self.Error.no_string_att.clear()

    def commit(self):
        if self._embedder != '' and self._smiles_attr != '':
            smiles = self.data[:, self._smiles_attr].metas.flatten()
            embedded, valid = self.to_fingerprints(smiles, self._embedder)
            invalid = list(set(range(len(smiles))) - set(valid))

            if not valid == []:
                embedded_table = Table.from_numpy(
                    Domain(
                        [ContinuousVariable.make("C_{}".format(x)) for x in
                         range(embedded.shape[1])],
                        self.data.domain.class_vars,
                        self.data.domain.metas
                    ),
                    embedded,
                    self.data.Y[valid],
                    self.data.metas[valid],
                    self.data.W[valid]
                )
                self.Outputs.embedded_smiles.send(embedded_table)
            else:
                self.Outputs.embedded_smiles.send(None)

            if not invalid == []:
                invalid_table = Table.from_numpy(
                    self.data.domain,
                    self.data.X[invalid],
                    self.data.Y[invalid],
                    self.data.metas[invalid],
                    self.data.W[invalid]
                )
                self.Outputs.skipped_smiles.send(invalid_table)
            else:
                self.Outputs.skipped_smiles.send(None)

    def _update_options(self):
        self.commit()
        pass

    @staticmethod
    def to_fingerprints(X, method="Topological"):
        if method not in EMBEDDERS:
            return -1

        if method.upper() == AUTOENCODER.upper():
            onehot = moleculeembedder.onehot_smiles(X, MAXLEN, CHARSET)
            out = EMBEDDERS[method]().predict([onehot])[0]
            valid = list(range(X.shape[0]))
        else:
            mols = np.array([Chem.MolFromSmiles(smile) for smile in X])
            valid = [i for i, x in enumerate(mols) if x is not None]
            mols = mols[valid]
            if method.upper() == CIRCULAR.upper():
                out = [EMBEDDERS[method](x, CIRCLE_RAD) for x in mols]
            else:
                out = [EMBEDDERS[method](x) for x in mols]
            out = np.asarray(out)

        return out, valid


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    table = Table("./BBBP2.csv")

    ow = OWMoleculeEmbedder()
    ow.show()
    ow.dataset(table)
    a.exec()
    ow.saveSettings()
