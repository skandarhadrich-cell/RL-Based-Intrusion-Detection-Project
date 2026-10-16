"""Tests for the holdout per-class metrics and observation helpers.

Synthetic and tiny - no dataset or checkpoint is loaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_ids import config as C
from rl_ids.holdout_eval import action_bucket, attack_labels_of, corrupt, evaluate
from rl_ids.metrics import (
    accuracy, binary_metrics, confusion, macro_average, per_class_metrics,
    weighted_average,
)

LABELS = ["Normal", "Suspicious", "Compromised"]


# --------------------------------------------------------------------------- #
# confusion / per-class metrics
# --------------------------------------------------------------------------- #
def test_confusion_is_rows_true_cols_pred():
    y_true = ["Normal", "Normal", "Compromised", "Compromised", "Suspicious"]
    y_pred = ["Normal", "Compromised", "Compromised", "Normal", "Suspicious"]
    m = confusion(y_true, y_pred, LABELS)
    assert m.shape == (3, 3)
    assert m[0].tolist() == [1, 0, 1]     # true Normal
    assert m[1].tolist() == [0, 1, 0]     # true Suspicious
    assert m[2].tolist() == [1, 0, 1]     # true Compromised
    assert m.sum() == 5


def test_per_class_metrics_hand_computed():
    # clean 2-class symmetric matrix [[1,1],[1,1]] -> every class 0.5/0.5
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "A"]
    rep = per_class_metrics(y_true, y_pred, ["A", "B"])
    for lab in ("A", "B"):
        v = rep[lab]
        assert v["tp"] == 1 and v["fp"] == 1 and v["fn"] == 1 and v["tn"] == 1
        assert v["precision"] == pytest.approx(0.5)
        assert v["recall"] == pytest.approx(0.5)
        assert v["f1"] == pytest.approx(0.5)
        assert v["fpr"] == pytest.approx(0.5)
        assert v["support"] == 2


def test_per_class_metrics_three_class():
    # Suspicious: 4 TP, 1 FN, 0 FP (its FN becomes a Normal FP)
    y_true = ["Suspicious"] * 5 + ["Normal"]
    y_pred = ["Suspicious"] * 4 + ["Normal", "Normal"]
    rep = per_class_metrics(y_true, y_pred, LABELS)
    s = rep["Suspicious"]
    assert s["tp"] == 4 and s["fn"] == 1 and s["fp"] == 0
    assert s["precision"] == pytest.approx(1.0)
    assert s["recall"] == pytest.approx(4 / 5)
    n = rep["Normal"]
    assert n["tp"] == 1 and n["fn"] == 0 and n["fp"] == 1


def test_fpr_one_vs_rest():
    # Compromised: 1 TP, 0 FN; everything else predicted compromised
    y_true = ["Compromised", "Normal", "Normal", "Normal", "Normal"]
    y_pred = ["Compromised"] * 5
    rep = per_class_metrics(y_true, y_pred, LABELS)
    c = rep["Compromised"]
    assert c["tp"] == 1 and c["fp"] == 4
    assert c["precision"] == pytest.approx(1 / 5)
    assert c["recall"] == pytest.approx(1.0)
    assert c["fpr"] == pytest.approx(1.0)   # 4 FP / (4 FP + 0 TN)


def test_macro_and_weighted_average():
    y_true = ["Normal", "Normal", "Compromised", "Suspicious"]
    y_pred = ["Normal", "Normal", "Suspicious", "Suspicious"]
    rep = per_class_metrics(y_true, y_pred, LABELS)
    assert 0.0 <= macro_average(rep, "f1") <= 1.0
    assert 0.0 <= weighted_average(rep, "f1") <= 1.0
    # support-weighted should equal the overall accuracy for recall here
    assert weighted_average(rep, "recall") == pytest.approx(accuracy(y_true, y_pred))


def test_accuracy_edge_cases():
    assert accuracy([], []) == 0.0
    assert accuracy(["a", "a"], ["a", "a"]) == 1.0
    assert accuracy(["a", "b"], ["b", "a"]) == 0.0


# --------------------------------------------------------------------------- #
# binary metrics (label-vs-benign)
# --------------------------------------------------------------------------- #
def test_binary_metrics_hand_computed():
    y_true = [True, True, True, False, False]     # 3 positives
    y_pred = [True, True, False, True, False]     # 2 TP, 1 FN, 1 FP, 1 TN
    m = binary_metrics(y_true, y_pred)
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 2)
    assert m["support"] == 3


def test_binary_metrics_no_positives_is_zero_fpr():
    m = binary_metrics([False, False], [False, False])
    assert m["fpr"] == 0.0 and m["recall"] == 0.0


# --------------------------------------------------------------------------- #
# observation helpers / action mapping
# --------------------------------------------------------------------------- #
def test_corrupt_preserves_shape_and_can_zero_features():
    rng = np.random.default_rng(0)
    X = np.full((50, len(C.OBS_FEATURES)), 5.0, dtype=np.float32)
    out = corrupt(X, rng)
    assert out.shape == X.shape and out.dtype == np.float32
    # with masking probability p, lots of entries should be exactly 0
    assert np.mean(out == 0.0) > 0.0
    assert not np.allclose(out, X)


def test_action_bucket_mapping():
    idx = C.ACTION_INDEX
    actions = [idx["Monitor"], idx["DPI"], idx["Throttle"], idx["Honeypot"],
               idx["Block"], idx["Isolate"]]
    assert action_bucket(actions).tolist() == [
        "Normal", "Suspicious", "Suspicious", "Suspicious", "Compromised", "Compromised"]


# --------------------------------------------------------------------------- #
# end-to-end evaluate() on synthetic, well-separated data
# --------------------------------------------------------------------------- #
class _EchoProbe:
    """Trivial 'probe' that predicts a column-preset label - avoids sklearn here."""

    def __init__(self, preds):
        self._preds = preds

    def predict(self, X):  # noqa: N803
        return np.asarray(self._preds)


def test_evaluate_shapes_and_label_metrics():
    import pandas as pd

    y_true = ["Normal", "Normal", "Compromised", "Compromised"]
    df = pd.DataFrame({
        "mdp_state": y_true,
        "attack_label": ["Benign", "Benign", "Infilteration", "Infilteration"],
    })
    pred = np.array(["Normal", "Normal", "Normal", "Compromised"], dtype=object)
    rep = evaluate(_EchoProbe(pred), np.zeros((4, 3)), df)
    assert rep["n"] == 4
    assert rep["acc"] == pytest.approx(0.75)
    assert rep["benign_fpr"] == pytest.approx(0.0)
    assert "Infilteration" in rep["labels"]
    # label-vs-benign: 1 TP, 1 FN, 0 FP -> recall 0.5, precision 1.0
    lab = rep["labels"]["Infilteration"]
    assert lab["tp"] == 1 and lab["fn"] == 1 and lab["fp"] == 0
    assert lab["recall"] == pytest.approx(0.5)
    assert lab["fpr"] == pytest.approx(0.0)


def test_attack_labels_of_excludes_benign():
    import pandas as pd

    df = pd.DataFrame({
        "mdp_state": ["Normal", "Suspicious", "Compromised"],
        "attack_label": ["BENIGN", "PortScan", "DDoS"],
    })
    assert attack_labels_of(df) == ["DDoS", "PortScan"]
