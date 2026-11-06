import csv
import os
import warnings

import numpy as np


def read_csv_positions(fn):
    """Read positions from a .csv file.

    Parameters
    ------------
    fn: str
        Name of csv file

    Returns
    --------
    type: list
        Type of position in each row ('Generic', 'Fiducial', 'Electrode', 'ReferenceElectrode' or
        'Coil')

    coordinates: list
        Coordinates in each row

    extra: list
        extra coordinates in each row (eg: electrode axes)

    name: list
        Name of position

    extra_cols: list
        Any extra information stored in the columns

    header: str
        Any information in the header

    """
    with open(os.path.expanduser(fn), "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    if len(rows[-1]) < 3:
        raise OSError("CSV file must have at least 4 rows")

    coordinates = []
    try:
        float(rows[0][1])
        header = []
        start = 0
    except ValueError:
        header = rows[0]
        start = 1

    rows = rows[start:]
    type_ = [r[0] for r in rows]

    extra = []
    name = []
    extra_cols = []
    type_filtered = []
    rows_filtered = []

    for t, r, i in zip(type_, rows, range(len(type_))):
        if t == "Fiducial" or len(r) == 4:
            name += [r[4] if len(r) >= 5 else None]
            extra_cols += [r[5:] if len(r) > 5 else None]
            extra += [None]
            type_filtered.append(t)
            rows_filtered.append(r)
        elif t in ["Electrode", "ReferenceElectrode"]:
            try:
                extra_ = np.array(r[4:7], float)
                assert len(extra_) == 3
                extra += [extra_]
                name += [r[7] if len(r) >= 8 else None]
                extra_cols += [r[8:] if len(r) > 8 else None]
            except (ValueError, AssertionError):
                extra += [None]
                name += [r[4] if len(r) >= 5 else None]
                extra_cols += [r[5:] if len(r) > 5 else None]
            type_filtered.append(t)
            rows_filtered.append(r)
        else:
            warnings.warn(f"Unrecognized column type: {t}")
    type_ = type_filtered
    rows = rows_filtered
    try:
        coordinates = np.array([[float(d) for d in r[1:4]] for r in rows], dtype=float)
    except ValueError as e:
        raise OSError("Could not read coordinates from CSV file") from e

    return type_, coordinates, extra, name, extra_cols, header


def write_csv_positions(filename, types, coordinates, name, extra=None, extra_cols=None, header=None):
    """Write positions to a .csv file.

    Parameters
    ------------
    filename: str
        Name of csv file
    types: list
        Type of position in each row ('Generic', 'Fiducial', 'Electrode', 'ReferenceElectrode' or
        'Coil')
    coordinates: numpy array
        Coordinates in each row
    name: list
        Name of position
    extra: list
        extra coordinates in each row (eg: electrode axes)
    extra_cols: list
        Any extra information stored in the columns
    header: str
        Any information in the header

    """
    n = len(types)

    coordinates = coordinates.tolist()

    name = [[n] if n else [] for n in name]

    if extra is None:
        extra = [None] * n
    extra = [[] if e is None else e.tolist() for e in extra]

    if extra_cols is None or len(extra_cols) == 0:
        extra_cols = [None] * n
    extra_cols = [e_c or [] for e_c in extra_cols]

    if header is None:
        header = []

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        if header != []:
            writer.writerow(header)
        for t, c, e, n, e_c in zip(types, coordinates, extra, name, extra_cols):
            writer.writerow([t] + c + e + n + e_c)
