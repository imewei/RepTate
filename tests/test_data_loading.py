import json
import os
import numpy as np
import pytest

from RepTate.core.ctypes_loader import load_ctypes_library
from RepTate.theories.linlin_io import load_linlin_data
from RepTate.tools import materials_db_io


def test_load_materials_json(tmp_path):
    payload = {
        "TEST": {
            "name": "TEST",
            "long": "Test Polymer",
            "tau_e": 1.23,
            "extra_key": 4.56,
        }
    }
    json_path = tmp_path / "materials_database.json"
    json_path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True), encoding="utf-8")

    db = materials_db_io.load_materials_json(str(json_path))
    assert "TEST" in db
    assert db["TEST"].data["name"] == "TEST"
    assert db["TEST"].data["extra_key"] == 4.56


def test_load_linlin_v2(tmp_path):
    Z = np.array([1, 2], dtype=np.int32)
    cnu = np.array([0.1], dtype=np.float64)
    table0 = np.ones((2, 3))
    table1 = np.zeros((2, 3))
    np.savez_compressed(
        tmp_path / "linlin_v2.npz",
        Z=Z,
        cnu=cnu,
        data_0000=table0,
        data_0001=table1,
    )

    Z_out, cnu_out, data_out = load_linlin_data(str(tmp_path))
    assert np.array_equal(Z_out, Z)
    assert np.array_equal(cnu_out, cnu)
    assert len(data_out) == 2
    assert np.array_equal(data_out[0], table0)
    assert np.array_equal(data_out[1], table1)


def test_load_linlin_legacy_migrates(tmp_path):
    Z = np.array([1, 2], dtype=np.int32)
    cnu = np.array([0.1], dtype=np.float64)
    data_obj = np.empty((2,), dtype=object)
    data_obj[0] = np.ones((2, 3))
    data_obj[1] = np.zeros((2, 3))
    np.savez(tmp_path / "linlin.npz", Z=Z, cnu=cnu, data=data_obj)

    Z_out, cnu_out, data_out = load_linlin_data(str(tmp_path))
    assert np.array_equal(Z_out, Z)
    assert np.array_equal(cnu_out, cnu)
    assert len(data_out) == 2
    assert (tmp_path / "linlin_v2.npz").exists()


def test_missing_ctypes_library_raises():
    with pytest.raises(ImportError):
        load_ctypes_library(os.path.join("missing", "libmissing.so"), "missing lib")
