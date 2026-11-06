import numpy as np

from charm.utils import csv_reader


class TestCSV:
    def test_read_csv_fiducial_electrode(self, tmp_path):
        fn = tmp_path / "positions.csv"
        fn.write_bytes(
            b"Fiducial,1,2,3,fiducial,a2\nElectrode,1.2,2.4,3.6,9,8,7,electrode,b2,b3\nReferenceElectrode,1.2,2.4,7.1\n"
        )
        type_, coordinates, extra, name, extra_cols, header = csv_reader.read_csv_positions(fn)
        assert type_ == ["Fiducial", "Electrode", "ReferenceElectrode"]
        assert np.allclose(coordinates, [[1, 2, 3], [1.2, 2.4, 3.6], [1.2, 2.4, 7.1]])
        assert header == []
        assert name == ["fiducial", "electrode", None]
        assert extra_cols == [["a2"], ["b2", "b3"], None]
        assert extra[0] is None
        assert np.allclose(extra[1], [9, 8, 7])
        assert extra[2] is None

    def test_write_csv_generic(self, tmp_path):
        type_ = ["Generic", "Generic"]
        coordinates = np.array([[1, 2, 3], [1.2, 2.4, 3.6]])
        header = ["type", "x", "y", "z", "extra1", "extra2"]
        name = ["a1", "b1"]
        extra_cols = [["a2"], ["b2"]]
        extra = [None, None]
        fn = tmp_path / "positions.csv"
        csv_reader.write_csv_positions(fn, type_, coordinates, name, extra, extra_cols, header)
        with open(fn) as f:
            assert f.readline().strip() == "type,x,y,z,extra1,extra2"
            assert f.readline().strip() == "Generic,1.0,2.0,3.0,a1,a2"
            assert f.readline().strip() == "Generic,1.2,2.4,3.6,b1,b2"

    def test_write_csv_electrode(self, tmp_path):
        type_ = ["Fiducial", "Electrode", "ReferenceElectrode"]
        coordinates = np.array([[1, 2, 3], [1.2, 2.4, 3.6], [1.2, 2.4, 7.1]])
        header = []
        name = ["fiducial", "electrode", None]
        extra_cols = [["a2"], ["b2", "b3"], None]
        extra = [None, np.array([9, 8, 7], dtype=float), None]

        fn = tmp_path / "positions.csv"
        csv_reader.write_csv_positions(fn, type_, coordinates, name, extra, extra_cols, header)
        with open(fn) as f:
            assert f.readline().strip() == "Fiducial,1.0,2.0,3.0,fiducial,a2"
            assert f.readline().strip() == "Electrode,1.2,2.4,3.6,9.0,8.0,7.0,electrode,b2,b3"
            assert f.readline().strip() == "ReferenceElectrode,1.2,2.4,7.1"

    def test_write_csv_electrode_simple(self, tmp_path):
        "Test without extra, extra_cols, and header."
        type_ = ["Fiducial", "Electrode", "ReferenceElectrode"]
        coordinates = np.array([[1, 2, 3], [1.2, 2.4, 3.6], [1.2, 2.4, 7.1]])
        name = ["fiducial", "electrode", None]
        fn = tmp_path / "positions.csv"
        csv_reader.write_csv_positions(fn, type_, coordinates, name)
        with open(fn) as f:
            assert f.readline().strip() == "Fiducial,1.0,2.0,3.0,fiducial"
            assert f.readline().strip() == "Electrode,1.2,2.4,3.6,electrode"
            assert f.readline().strip() == "ReferenceElectrode,1.2,2.4,7.1"
