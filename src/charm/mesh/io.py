"""Reading and writing meshes and surfaces.

Derived from the SimNIBS package.
"""

import itertools
import os
import struct
import warnings

import cortech
import nibabel
import numpy as np

from charm.mesh.element_tags import ElementTags, ElementTypes
from charm.mesh.msh import Elements, Msh, Nodes, make_surface_mesh
from charm.subject import SURFACE_FILE_NAME_TO_ELEMENT_TAG, SubjectFiles

_NODES_PER_ELEMENT: dict[int, int] = {
    ElementTypes.POINT: 1,
    ElementTypes.LINE: 2,
    ElementTypes.TRIANGLE: 3,
    ElementTypes.TETRAHEDRON: 4,
}
_WRITE_ORDER = (
    ElementTypes.POINT,
    ElementTypes.LINE,
    ElementTypes.TRIANGLE,
    ElementTypes.TETRAHEDRON,
)
# Node counts of every gmsh element type, to skip the ones that cannot be read
_NODES_PER_UNREADABLE_ELEMENT = dict(
    enumerate((2, 3, 4, 4, 8, 6, 5, 3, 6, 9, 10, 27, 18, 14, 1, 8, 20, 15, 13), start=1)
)


def read_msh(fn, m=None):
    """Read a gmsh v2 '.msh' file.

    Parameters
    ------------
    fn: str
        File name
    m: charm.mesh.msh.Msh (optional)
        Mesh to fill in. A new one is created if unset.

    Returns
    --------
    msh: charm.mesh.msh.Msh
        Mesh structure
    """
    if m is None:
        m = Msh()

    fn = os.path.expanduser(fn)

    if not os.path.isfile(fn):
        raise OSError(fn + " not found")

    version_number = _find_mesh_version(fn)
    if version_number == 2:
        m = _read_msh_2(fn, m)

    elif version_number == 4:
        raise OSError(f"Can not read Mesh with file version : {version_number}")

    else:
        raise OSError(f"Unrecgnized Mesh file version : {version_number}")

    return m


def _find_mesh_version(fn):
    if not os.path.isfile(fn):
        raise OSError(fn + " not found")

    with open(fn, "rb") as f:
        first_line = f.readline().decode()
        if not first_line.startswith("$MeshFormat"):
            raise OSError(fn, "must start with $MeshFormat")

        version_number, file_type, data_size = f.readline().decode().split()
        version_number = int(version_number[0])
        file_type = int(file_type)
        data_size = int(data_size)
    return version_number


def _read_msh_2(fn, m):
    m.fn = fn

    with open(fn, "rb") as f:
        first_line = f.readline()
        if not first_line.startswith(b"$MeshFormat"):
            raise OSError(fn, "must start with $MeshFormat")

        version_number, file_type, data_size = f.readline().decode().strip().split()
        version_number = int(version_number[0])
        file_type = int(file_type)
        data_size = int(data_size)

        if version_number != 2:
            raise OSError("Can only handle v2 meshes")

        if file_type == 1:
            binary = True
        elif file_type == 0:
            binary = False
        else:
            raise OSError(f"File_type not recognized: {file_type}")

        if data_size != 8:
            raise OSError(f"data_size should be double (8), i'm reading: {data_size}")

        if binary:
            endianness = struct.unpack("i", f.readline()[:4])[0]
        else:
            endianness = 1

        if endianness != 1:
            raise OSError("endianness is not 1, is the endian order wrong?")

        if not f.readline().startswith(b"$EndMeshFormat"):
            raise OSError(fn + " expected $EndMeshFormat")

        while not f.readline().startswith(b"$Nodes"):
            continue

        try:
            node_nr = int(f.readline().decode().strip())
        except ValueError as e:
            raise OSError(fn + " something wrong with Line 5 - should be a number") from e

        if binary:
            dt = np.dtype([("id", np.int32), ("coord", np.float64, 3)])

            temp = np.fromfile(f, dtype=dt, count=node_nr)
            node_number = np.copy(temp["id"])
            node_coord = np.copy(temp["coord"])

            # Some writers put a line feed after the binary nodes and some don't
            LF_byte = f.read(1)
            if ord(LF_byte) != 10:
                f.seek(-1, 1)

        else:
            rows = np.fromstring(b"".join(itertools.islice(f, node_nr)).decode(), sep=" ").reshape(-1, 4)
            node_number = rows[:, 0].astype("int32")
            node_coord = rows[:, 1:]

        if not np.all(node_number == np.arange(1, node_nr + 1)):
            warnings.warn("Mesh file with discontinuos nodes, things can fail unexpectedly")
        m.nodes.node_coord = node_coord

        if not f.readline().startswith(b"$EndNodes"):
            raise OSError(fn + " expected $EndNodes after reading " + str(node_nr) + " nodes")

        if not f.readline().startswith(b"$Elements"):
            raise OSError(fn, "expected line with $Elements")

        try:
            elm_nr = int(f.readline().decode().strip())
        except ValueError as e:
            raise OSError(
                fn + " something wrong when reading number of elements (line after $Elements)- should be a number"
            ) from e

        elm_number = np.empty(elm_nr, dtype="int32")
        m.elm.elm_type = np.empty(elm_nr, dtype="int32")
        m.elm.tag1 = np.empty(elm_nr, dtype="int32")
        m.elm.tag2 = np.empty(elm_nr, dtype="int32")
        m.elm.node_number_list = -np.ones((elm_nr, 4), dtype="int32")
        read = np.ones(elm_nr, dtype=bool)

        if binary:
            current_element = 0
            while current_element < elm_nr:
                elm_type, nr, _ = np.fromfile(f, "int32", 3)
                block = slice(current_element, current_element + nr)
                if elm_type in _NODES_PER_ELEMENT:
                    n_nodes = _NODES_PER_ELEMENT[elm_type]
                    tmp = np.fromfile(f, "int32", nr * (3 + n_nodes)).reshape(-1, 3 + n_nodes)
                    m.elm.elm_type[block] = elm_type
                    elm_number[block] = tmp[:, 0]
                    m.elm.tag1[block] = tmp[:, 1]
                    m.elm.tag2[block] = tmp[:, 2]
                    m.elm.node_number_list[block, :n_nodes] = tmp[:, 3:]
                else:
                    warnings.warn(f"element of type {elm_type} cannot be read, ignoring it")
                    np.fromfile(f, "int32", nr * (3 + _NODES_PER_UNREADABLE_ELEMENT[elm_type]))
                    read[block] = False
                current_element += nr

        else:
            for ii in range(elm_nr):
                line = f.readline().split()
                elm_type = int(line[1])
                if elm_type in _NODES_PER_ELEMENT:
                    elm_number[ii] = int(line[0])
                    m.elm.elm_type[ii] = elm_type
                    m.elm.tag1[ii] = int(line[3])
                    m.elm.tag2[ii] = int(line[4])
                    m.elm.node_number_list[ii, : _NODES_PER_ELEMENT[elm_type]] = [int(i) for i in line[5:]]
                else:
                    read[ii] = False
                    warnings.warn(f"element of type {elm_type} cannot be read, ignoring it")

        elm_number = elm_number[read]
        m.elm.elm_type = m.elm.elm_type[read]
        m.elm.tag1 = m.elm.tag1[read]
        m.elm.tag2 = m.elm.tag2[read]
        m.elm.node_number_list = m.elm.node_number_list[read]

        if not np.all(elm_number == np.arange(1, m.elm.nr + 1)):
            warnings.warn("Changing element numbering")
            elm_number = np.arange(1, m.elm.nr + 1)

        line = f.readline()
        if b"$EndElements" not in line:
            line = f.readline()
            if b"$EndElements" not in line:
                raise OSError(fn + " expected $EndElements after reading " + str(m.elm.nr) + " elements. Read " + line)

    m.compact_ordering(node_number)
    return m


def write_msh(msh, file_name=None, mode="binary"):
    """Write a gmsh v2 '.msh' file.

    Parameters
    ------------
    msh: charm.mesh.msh.Msh
        Mesh structure
    file_name: str (optional)
        File name. Default: msh.fn
    mode: 'binary' or 'ascii'
        The mode in which the file is written
    """
    if file_name is not None:
        msh.fn = file_name

    fn = msh.fn

    if fn[0] == "~":
        fn = os.path.expanduser(fn)

    if mode not in ["ascii", "binary"]:
        raise ValueError("Only 'ascii' and 'binary' are allowed")

    with open(fn, "wb") as f:
        if mode == "ascii":
            f.write(b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        elif mode == "binary":
            f.write(b"$MeshFormat\n2.2 1 8\n")
            f.write(struct.pack("i", 1))
            f.write(b"\n$EndMeshFormat\n")

        f.write(b"$Nodes\n")
        f.write(f"{msh.nodes.nr}\n".encode("ascii"))
        if mode == "ascii":
            np.savetxt(
                f,
                np.column_stack((msh.nodes.node_number, msh.nodes.node_coord)),
                fmt=("%d", "%.17g", "%.17g", "%.17g"),
            )
        else:
            nodes = np.empty(msh.nodes.nr, dtype=[("id", np.int32), ("coord", np.float64, 3)])
            nodes["id"] = msh.nodes.node_number
            nodes["coord"] = msh.nodes.node_coord
            f.write(nodes.tobytes())
        f.write(b"$EndNodes\n")

        f.write(b"$Elements\n")
        f.write(f"{msh.elm.nr}\n".encode("ascii"))
        is_written = np.isin(msh.elm.elm_type, list(_NODES_PER_ELEMENT))
        if not np.all(is_written):
            raise OSError(
                "ERROR: cant write meshes with elements of type",
                np.unique(msh.elm.elm_type[~is_written]),
            )
        elm_number = msh.elm.elm_number
        for elm_type in _WRITE_ORDER:
            idx = np.flatnonzero(msh.elm.elm_type == elm_type)
            if len(idx) == 0:
                continue
            n_nodes = _NODES_PER_ELEMENT[elm_type]
            if mode == "ascii":
                rows = np.column_stack(
                    (
                        elm_number[idx],
                        np.full(len(idx), elm_type),
                        np.full(len(idx), 2),
                        msh.elm.tag1[idx],
                        msh.elm.tag2[idx],
                        msh.elm.node_number_list[idx, :n_nodes],
                    )
                )
                np.savetxt(f, rows, fmt="%d")
            else:
                f.write(np.array((elm_type, len(idx), 2), "int32").tobytes())
                rows = np.column_stack(
                    (
                        elm_number[idx],
                        msh.elm.tag1[idx],
                        msh.elm.tag2[idx],
                        msh.elm.node_number_list[idx, :n_nodes],
                    )
                )
                f.write(rows.astype("int32").tobytes())

        f.write(b"$EndElements\n")


def read_gifti_surface(fn, element_tag: ElementTags | None = None):
    """Read a gifti surface.

    Parameters
    -----------
    fn: str
        File name
    element_tag: ElementTags (optional)
        Tag of the mesh elements. If unset, the tag comes from the "ElementTag" metadata, then from the file name
        if it is a known surface, and is ElementTags.UNKNOWN_SURFACE otherwise.

    Returns
    ---------
    msh: Msh
        Surface mesh
    """
    gii = nibabel.load(fn)
    if not isinstance(gii, nibabel.gifti.GiftiImage):
        raise TypeError(f"{fn} is not a GIFTI surface")
    surface = cortech.Surface.from_gifti(gii)
    surface.to_scanner_ras()

    if element_tag is not None:
        mesh_element_tag = element_tag
    elif "ElementTag" in gii.meta:
        mesh_element_tag = ElementTags(int(gii.meta["ElementTag"]))
    elif os.path.basename(fn) in SURFACE_FILE_NAME_TO_ELEMENT_TAG:
        mesh_element_tag = SURFACE_FILE_NAME_TO_ELEMENT_TAG[os.path.basename(fn)]
    else:
        mesh_element_tag = ElementTags.UNKNOWN_SURFACE

    mesh = make_surface_mesh(surface.vertices, surface.faces + 1, mesh_element_tag)
    return mesh


def write_gifti_surface(
    surface: cortech.Surface | Msh,
    fn,
    element_tag: ElementTags | None = None,
    geometry=None,
):
    """Write a surface as a gifti file.

    Parameters
    -----------
    surface: cortech.Surface or Msh
        Surface to write. Only the triangles of a Msh are written.
    fn: str
        File name
    element_tag: ElementTags (optional)
        Stored as "ElementTag" metadata
    geometry: optional
        Geometry passed to cortech.Surface when surface is a Msh
    """
    match surface:
        case Msh():
            surface = surface.crop_mesh(elm_type=ElementTypes.TRIANGLE)
            vertices = surface.nodes.node_coord
            faces = surface.elm.node_number_list[:, :3] - 1
            surface = cortech.Surface(vertices, faces, space="scanner", geometry=geometry)
        case cortech.Surface():
            pass
        case _:
            raise ValueError(f"surface must be an instance of {cortech.Surface} or {Msh} (got {type(surface)})")
    surface.to_surface_ras()
    gii = surface.as_gifti()

    if element_tag is not None:
        gii.meta["ElementTag"] = int(element_tag)

    gii.to_filename(fn)


def read_medit(fn):
    """Read a MEDIT ".mesh" file.

    Parameters
    -----------
    fn: str
        File name

    Returns
    --------
    mesh: Msh
    """

    def next_line(f):
        # CGAL >= 6 writes "# CGAL::..." comment lines into the file.
        while line := f.readline():
            if line.strip() and not line.startswith("#"):
                return line.strip()
        return ""

    with open(fn, "r") as f:
        if not next_line(f).startswith("MeshVersionFormatted"):
            raise OSError("invalid mesh format")
        if next_line(f) != "Dimension 3":
            raise OSError("Can only read 3D meshes")
        if next_line(f) != "Vertices":
            raise OSError("invalid mesh format")
        n_vertices = int(next_line(f))
        assert n_vertices > 0
        vertices = np.loadtxt(f, dtype=float, max_rows=n_vertices)[:, :-1]
        nodes = Nodes(vertices)
        elm = Elements()
        while True:
            element_type = next_line(f)
            if element_type == "Triangles":
                elm_type = 2
            elif element_type == "Tetrahedra":
                elm_type = 4
            elif element_type == "End":
                break
            else:
                raise OSError(f"Cant read element type: {element_type}")
            n_elements = int(next_line(f))
            elements_tag = np.loadtxt(f, dtype=int, max_rows=n_elements, ndmin=2)
            elements = elements_tag[:, :-1]
            tag = elements_tag[:, -1]
            if elements.shape[1] == 3:
                elements = np.hstack((elements, -1 * np.ones((n_elements, 1), int)))
            elm.node_number_list = np.vstack((elm.node_number_list, elements))
            elm.elm_type = np.hstack((elm.elm_type, elm_type * np.ones(n_elements, int)))
            elm.tag1 = np.hstack((elm.tag1, tag))

        elm.tag2 = elm.tag1.copy()
        return Msh(nodes, elm)


def load_subject_surfaces(sub_files: SubjectFiles, surface: str, subsampling: int | None = None) -> dict[str, Msh]:
    """Load the surfaces of a subject, keyed by hemisphere.

    Parameters
    -----------
    sub_files : charm.subject.SubjectFiles
    surface : str
        The surface to load (e.g., central).
    subsampling : int | None
        The subsampled surface to load. If None, load the full resolution surface.
    """
    # Leadfield simulations append the hemispheres in the order of SubjectFiles.hemispheres, so keep that order
    return {h: read_gifti_surface(sub_files.get_surface(h, surface, subsampling)) for h in sub_files.hemispheres}
