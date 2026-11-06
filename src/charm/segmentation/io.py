import functools
import os
from typing import NamedTuple

import h5py
import numpy as np

from charm.segmentation import gems


class GMMParameter(NamedTuple):
    merged_name: str
    number_of_components: int
    search_strings: list[str]


def read_shared_gmm_parameters(file_name):
    """Read a text file where each line is "mergedName numberOfComponents searchString(s)"."""
    parameters = []
    with open(file_name) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] == "#":
                continue
            if len(fields) < 2:
                raise ValueError(f"Can't parse line: {line}")
            parameters.append(GMMParameter(fields[0], int(fields[1]), fields[2:]))
    return parameters


def read_compression_lookup_table(file_name):
    """Read "FreeSurferLabel compressedLabel name ..." lines, sorted by compressedLabel."""
    with open(file_name) as f:
        rows = sorted((line.split() for line in f), key=lambda row: int(row[1]))
    return [int(row[0]) for row in rows], [row[2] for row in rows]


@functools.lru_cache(maxsize=4)
def _read_mesh_arrays(file_name, modification_time, size):
    # The modification time and size are in the key so that a rewritten file isn't served from the cache
    with h5py.File(file_name, "r") as f:
        positions = f["positions"][()].astype(np.float64)
        indptr = f["alpha_indptr"][()]
        alphas = np.zeros((len(positions), f.attrs["number_of_labels"]), dtype=np.float32)
        rows = np.repeat(np.arange(len(positions)), np.diff(indptr))
        alphas[rows, f["alpha_indices"][()]] = f["alpha_values"][()]
        return positions, f["tetrahedra"][()], alphas, f["can_move"][()], float(f.attrs["k"])


def read_mesh_collection(file_name):
    file_name = os.fspath(file_name)
    status = os.stat(file_name)
    positions, tetrahedra, alphas, can_move, k = _read_mesh_arrays(file_name, status.st_mtime_ns, status.st_size)
    collection = gems.KvlMeshCollection()
    collection.assign(positions, tetrahedra, alphas, can_move, k)
    return collection


def write_mesh_arrays(file_name, positions, tetrahedra, alphas, can_move, k, compression_level=1):
    """Write mesh collection arrays to HDF5. Positions are single precision, as the original atlases were, and the
    alphas are stored sparse."""
    rows, labels = np.nonzero(alphas)
    options = {"compression": "gzip", "compression_opts": compression_level, "shuffle": True}
    with h5py.File(file_name, "w") as f:
        f.attrs["k"] = k
        f.attrs["number_of_labels"] = alphas.shape[1]
        f.create_dataset("positions", data=np.asarray(positions, dtype=np.float32), **options)
        f.create_dataset("tetrahedra", data=np.asarray(tetrahedra, dtype=np.uint32), **options)
        f.create_dataset(
            "alpha_indptr",
            data=np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(alphas))))).astype(np.uint32),
            **options,
        )
        f.create_dataset("alpha_indices", data=labels.astype(np.uint8), **options)
        f.create_dataset("alpha_values", data=alphas[rows, labels], **options)
        f.create_dataset("can_move", data=np.asarray(can_move, dtype=np.uint8), **options)


def write_mesh_collection(collection, file_name):
    write_mesh_arrays(file_name, *collection.contents())
