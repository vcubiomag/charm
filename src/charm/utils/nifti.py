"""Reading volume images with their types known, and writing NIfTI images compressed on all cores.

The deflate stream of a .nii.gz is cut into chunks that end on a byte boundary, as
pigz does, so that they compress independently and still make up a single gzip
member that any reader can open.
"""

import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
from nibabel.spatialimages import SpatialImage

_CHUNK_SIZE = 1 << 23
# nibabel's own default, which trades a few percent of size for speed
_COMPRESSION_LEVEL = 1


def _deflate(chunk, is_last):
    compressor = zlib.compressobj(_COMPRESSION_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(chunk) + compressor.flush(zlib.Z_FINISH if is_last else zlib.Z_SYNC_FLUSH)


def load(filename) -> SpatialImage:
    """Like nibabel.load, for the NIfTI and MGH volumes charm reads."""
    image = nib.load(filename)
    if not isinstance(image, SpatialImage):
        raise TypeError(f"{filename} is not a volume image")
    return image


def load_nifti(filename) -> nib.Nifti1Image:
    """Like nibabel.load, for a NIfTI-1 or NIfTI-2 file."""
    image = nib.load(filename)
    if not isinstance(image, nib.Nifti1Image):
        raise TypeError(f"{filename} is not a NIfTI image")
    return image


def save(image, filename):
    """Like nibabel.save, with .gz files compressed in parallel."""
    filename = os.fspath(filename)
    if not filename.endswith(".gz") or not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        nib.save(image, filename)
        return

    data = memoryview(image.to_bytes())
    chunks = [data[start : start + _CHUNK_SIZE] for start in range(0, len(data), _CHUNK_SIZE)] or [data]
    with ThreadPoolExecutor() as executor:
        deflated = list(executor.map(_deflate, chunks, [i == len(chunks) - 1 for i in range(len(chunks))]))

    with open(filename, "wb") as f:
        # gzip header with no name and no time stamp, as nibabel writes it
        f.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff")
        f.writelines(deflated)
        f.write(struct.pack("<II", zlib.crc32(data) & 0xFFFFFFFF, len(data) & 0xFFFFFFFF))
