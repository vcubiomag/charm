import time

import cat_surf
import cortech
import h5py
import numpy as np

from charm.logging import logger
from charm.mesh.element_tags import ElementTags
from charm.mesh.io import write_gifti_surface
from charm.resources import templates
from charm.subject import SubjectFiles


def spherical_registration(m2m: SubjectFiles, hemi: str):
    """Map the white surface of `hemi` to a sphere and register it to fsaverage."""
    white = cortech.Surface.from_file(m2m.get_surface(hemi, "white"))
    with h5py.File(templates.fsaverage, "r") as f:
        fsavg_faces = f["faces"][()].astype(np.int64)
        fsavg_white = f[hemi]["white"][()].astype(float)
        fsavg_sphere = f[hemi]["sphere"][()].astype(float)

    logger.info(f"Generating sphere ({hemi})")
    start = time.perf_counter()
    sphere_vertices, sphere_faces = cat_surf.surf_to_sphere(white.vertices.astype(float), white.faces, stop_at=10)
    write_gifti_surface(
        cortech.Sphere(sphere_vertices, sphere_faces, normalize=False),
        m2m.get_surface(hemi, "sphere"),
        element_tag=ElementTags.from_surface_file_name("sphere", hemi),
    )
    logger.info(f"Generating sphere ({hemi}) : {_elapsed(start)}")

    logger.info(f"Registering sphere ({hemi})")
    start = time.perf_counter()
    registered_vertices, registered_faces = cat_surf.spherical_demon(
        (white.vertices.astype(float), white.faces),
        (sphere_vertices, sphere_faces),
        (fsavg_white, fsavg_faces),
        (fsavg_sphere, fsavg_faces),
    )
    write_gifti_surface(
        cortech.Sphere(registered_vertices, registered_faces),
        m2m.get_surface(hemi, "sphere.reg"),
        element_tag=ElementTags.from_surface_file_name("sphere.reg", hemi),
    )
    logger.info(f"Registering sphere ({hemi}) : {_elapsed(start)}")


def _elapsed(start: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))
