#!/usr/bin/env python3
"""Build the atlas, templates and EEG caps bundled with charm from a SimNIBS checkout.

    uv run python tools/build_package_data.py --simnibs ../simnibs

Only what charm reads is kept. The atlas mesh files lose the second copy of the node
positions and their vertices, lines and triangles, none of which anything uses. The
fsaverage surfaces share their faces and leave out the base64 encoding of GIfTI.
"""

import argparse
import gzip
import shutil
from pathlib import Path

import cortech
import h5py
import numpy as np

from charm.segmentation.io import write_mesh_arrays

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "src" / "charm" / "_data"
ATLAS = "charm_atlas_mni_v1-1"
MESHES = {
    "affine_no_neck.txt.gz": "affine_no_neck.h5",
    "atlas_level1.txt.gz": "atlas_level1.h5",
    "atlas_level2.txt.gz": "atlas_level2.h5",
}


def _count(f):
    return int(f.readline().split(":")[1])


def convert_mesh(source: Path, destination: Path):
    with gzip.open(source, "rt") as f:
        n_points, n_cells, n_labels, n_meshes = (_count(f) for _ in range(4))

        f.readline()
        reference = np.loadtxt(f, max_rows=n_points)
        # Node numbers in a file can have gaps
        new_number = np.full(int(np.max(reference[:, 0])) + 1, -1, dtype=np.int64)
        new_number[reference[:, 0].astype(int)] = np.arange(n_points)
        k = float(f.readline().split(":")[1])

        for _ in range(n_meshes):
            f.readline()
            np.loadtxt(f, max_rows=n_points)

        f.readline()
        tetrahedra = []
        for _ in range(n_cells):
            fields = f.readline().split()
            if fields[1] == "TETRAHEDRON":
                tetrahedra.append(fields[2:6])
        tetrahedra = new_number[np.array(tetrahedra, dtype=np.int64)]

        f.readline()
        alphas = np.zeros((n_points, n_labels), dtype=np.float32)
        can_move = np.zeros((n_points, 3), dtype=np.uint8)
        for _ in range(n_points):
            fields = f.readline().split()
            node = new_number[int(fields[0])]
            alphas[node] = np.array(fields[1 : 1 + n_labels], dtype=np.float64)
            can_move[node] = [flag == "true" for flag in fields[2 + n_labels : 5 + n_labels]]

    # Subnormal alphas are noise, and would only make the atlas less sparse
    alphas[np.abs(alphas) < np.finfo(np.float32).tiny] = 0
    # In order of their first node, the tetrahedra compress to half the size
    tetrahedra = tetrahedra[np.lexsort(tetrahedra.T[::-1])]
    write_mesh_arrays(destination, reference[:, 1:], tetrahedra, alphas, can_move, k, compression_level=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simnibs", type=Path, help="Path to a SimNIBS checkout (the repo root)")
    parser.add_argument(
        "--mesh", nargs=2, type=Path, metavar=("SOURCE", "DESTINATION"), help="Convert a single mesh file"
    )
    args = parser.parse_args()

    if args.mesh:
        convert_mesh(*args.mesh)
        return

    package = args.simnibs / "simnibs"
    build_atlas(package / "segmentation" / "atlases" / ATLAS, DATA / "atlases" / ATLAS)
    build_templates(package / "resources", DATA)


def build_atlas(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for text, h5 in MESHES.items():
        print(f"converting {text}")
        convert_mesh(source / text, out / h5)

    with (
        open(source / "template.nii", "rb") as f,
        open(out / "template.nii.gz", "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as compressed,
    ):
        shutil.copyfileobj(f, compressed)
    for name in ("compressionLookupTable.txt", "sharedGMMParameters.txt"):
        shutil.copy2(source / name, out / name)

    settings = (source / f"{ATLAS}.ini").read_text()
    for text, h5 in MESHES.items():
        settings = settings.replace(text, h5)
    settings = settings.replace('"template.nii"', '"template.nii.gz"')
    settings = "".join(
        line
        for line in settings.splitlines(keepends=True)
        if not line.startswith(("lookup_table_segmentation", "# affine_atlas"))
    )
    (out / f"{ATLAS}.ini").write_text(settings)


def build_templates(resources: Path, out: Path) -> None:
    templates = out / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resources / "templates" / "MNI152_T1_1mm.nii.gz", templates)

    options = {"compression": "gzip", "compression_opts": 9, "shuffle": True}
    with h5py.File(templates / "fsaverage.h5", "w") as f:
        for hemi in ("lh", "rh"):
            for name in ("white", "sphere"):
                surface = cortech.Surface.from_file(resources / "templates" / "fsaverage_surf" / f"{hemi}.{name}.gii")
                f.create_dataset(f"{hemi}/{name}", data=surface.vertices.astype(np.float32), **options)
        f.create_dataset("faces", data=surface.faces.astype(np.int32), **options)

    caps = out / "eeg_caps"
    caps.mkdir(parents=True, exist_ok=True)
    for cap in (resources / "ElectrodeCaps_MNI").glob("*.csv"):
        shutil.copy2(cap, caps)


if __name__ == "__main__":
    main()
