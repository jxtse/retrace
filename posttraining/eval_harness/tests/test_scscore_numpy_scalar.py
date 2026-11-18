import numpy as np

from tools.verifier import scscore
from tools.verifier.scscore_coley.standalone_model_numpy import sigmoid


class _ArrayScoreColey:
    def get_score_from_smi(self, smiles):
        return smiles, np.array([2.75])


def test_sc_score_accepts_one_element_numpy_array(monkeypatch):
    monkeypatch.setattr(scscore, "_try_load_coley", lambda: _ArrayScoreColey())
    monkeypatch.setattr(scscore, "_try_import_sascorer", lambda: None)

    assert scscore.sc_score("CCO") == 2.75


def test_coley_sigmoid_accepts_numpy_array():
    out = sigmoid(np.array([0.0]))

    assert out.shape == (1,)
    assert float(out[0]) == 0.5
