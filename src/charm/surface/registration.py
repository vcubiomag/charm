import time
from dataclasses import dataclass

import brainsynth.resources
import cortech
import h5py
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
from scipy.spatial import KDTree

from charm.logging import logger
from charm.mesh.element_tags import ElementTags
from charm.mesh.io import write_gifti_surface
from charm.resources import templates
from charm.subject import SubjectFiles


@dataclass(frozen=True)
class _Level:
    features: tuple[str, ...]
    sigma: float  # smoothing of the features along the white surface, mm
    n_control: int
    iters: int
    lr: float


# Validated against FreeSurfer's own registration of a subject. Finer levels made it worse: below 4 mm the folding of
# fsaverage, an average surface, no longer matches an individual brain.
_LEVELS = (
    _Level(("sulc",), 16.0, 162, 150, 0.005),
    _Level(("sulc",), 8.0, 642, 150, 0.003),
    _Level(("sulc", "sulc_fine"), 4.0, 2562, 150, 0.002),
)
_SULC_SIGMA = {"sulc": 15.0, "sulc_fine": 5.0}
_AREA_WEIGHT = 0.02
_FOLD_WEIGHT = 0.01


def spherical_registration(m2m: SubjectFiles, hemi: str):
    """Map the white surface of `hemi` to a sphere and register it to fsaverage."""
    white = cortech.Surface.from_file(m2m.get_surface(hemi, "white"))
    with h5py.File(templates.fsaverage, "r") as f:
        fsavg_faces = f["faces"][()].astype(np.int64)
        fsavg_white = f[hemi]["white"][()].astype(float)
        fsavg_sphere = f[hemi]["sphere"][()].astype(float)

    logger.info(f"Generating sphere ({hemi})")
    start = time.perf_counter()
    sphere = template_sphere(white, hemi)
    write_gifti_surface(
        cortech.Sphere(sphere, white.faces),
        m2m.get_surface(hemi, "sphere"),
        element_tag=ElementTags.from_surface_file_name("sphere", hemi),
    )
    logger.info(f"Generating sphere ({hemi}) : {_elapsed(start)}")

    logger.info(f"Registering sphere ({hemi})")
    start = time.perf_counter()
    registered = register_sphere(white, sphere, cortech.Surface(fsavg_white, fsavg_faces), fsavg_sphere)
    write_gifti_surface(
        cortech.Sphere(registered, white.faces),
        m2m.get_surface(hemi, "sphere.reg"),
        element_tag=ElementTags.from_surface_file_name("sphere.reg", hemi),
    )
    logger.info(f"Registering sphere ({hemi}) : {_elapsed(start)}")


def template_sphere(white: cortech.Surface, hemi: str) -> np.ndarray:
    """The sphere of a TopoFit surface.

    TopoFit deforms a template mesh without changing its connectivity, so the
    template's own registered sphere is a fold-free sphere for every surface it
    produces, and roughly aligned with fsaverage.
    """
    template = brainsynth.resources.Template().load_surface(hemi, "sphere.reg")
    if not np.array_equal(white.faces, template["faces"].numpy()):
        raise ValueError(f"The {hemi} white surface is not on the TopoFit template mesh")
    return _unit(template["vertices"].numpy().astype(float))


def register_sphere(
    surface: cortech.Surface, sphere: np.ndarray, target: cortech.Surface, target_sphere: np.ndarray
) -> np.ndarray:
    """Warp `sphere` so that the folding of `surface` lines up with that of `target` on `target_sphere`.

    The warp is fitted coarse to fine on sulcal depth, as a smooth field of
    tangent displacements, with a penalty on changing the area of the triangles
    and a stronger one on folding them. Returns unit vectors.
    """
    surface_diffusion = _HeatDiffusion(surface)
    target_diffusion = _HeatDiffusion(target)
    names = {name for level in _LEVELS for name in level.features}
    features = {name: _sulcal_depth(surface, surface_diffusion, _SULC_SIGMA[name]) for name in names}
    target_features = {name: _sulcal_depth(target, target_diffusion, _SULC_SIGMA[name]) for name in names}

    x = torch.tensor(_unit(sphere), dtype=torch.float32)
    faces = torch.tensor(surface.faces, dtype=torch.int64)
    # Dividing by the area on the starting sphere also cancels the face orientation, which differs between hemispheres
    reference_area = _signed_areas(x, faces)
    target_sphere = _unit(target_sphere)
    for level in _LEVELS:
        moving = _zscore(np.stack([surface_diffusion(features[n], level.sigma) for n in level.features], 1))
        fixed = _zscore(np.stack([target_diffusion(target_features[n], level.sigma) for n in level.features], 1))
        control = torch.tensor(_fibonacci_sphere(level.n_control), dtype=torch.float32)
        x = _fit_warp(
            x,
            torch.tensor(moving, dtype=torch.float32),
            _SphereField(target_sphere, fixed),
            faces,
            reference_area,
            control,
            level,
        )
    return x.numpy().astype(float)


def _fit_warp(x, moving, field, faces, reference_area, control, level):
    idx, weights = _control_weights(x, control)
    displacement = torch.zeros(len(control), 3, requires_grad=True)
    optimizer = torch.optim.Adam([displacement], lr=level.lr)
    schedule = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.1, level.iters)
    for _ in range(level.iters):
        optimizer.zero_grad()
        warped = _warp(x, displacement, idx, weights)
        area = _signed_areas(warped, faces) / reference_area
        loss = (
            ((field(warped) - moving) ** 2).mean()
            + _AREA_WEIGHT * (torch.log(torch.clamp(area, min=1e-3)) ** 2).mean()
            # Summed rather than averaged, so that a few folded triangles cannot hide among half a million
            + _FOLD_WEIGHT * (torch.relu(0.1 - area) ** 2).sum()
        )
        loss.backward()
        optimizer.step()
        schedule.step()
    with torch.no_grad():
        return _warp(x, displacement, idx, weights)


def _warp(x, displacement, idx, weights):
    u = (weights[:, :, None] * displacement[idx]).sum(1)
    u = u - (u * x).sum(1, keepdim=True) * x
    x = x + u
    return x / torch.linalg.norm(x, dim=1, keepdim=True)


def _control_weights(x, control, k=8):
    # Gaussian rather than barycentric interpolation keeps the warp smooth across the control triangles, which the
    # optimizer needs.
    control = control.numpy()
    spacing = np.median(KDTree(control).query(control, k=2)[0][:, 1])
    dist, idx = KDTree(control).query(x.numpy(), k=k)
    weights = np.exp(-0.5 * (dist / spacing) ** 2)
    weights /= weights.sum(1, keepdims=True)
    return torch.tensor(idx), torch.tensor(weights, dtype=torch.float32)


def _signed_areas(x, faces):
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    return 0.5 * (torch.linalg.cross(b - a, c - a) * (a + b + c)).sum(1) / torch.linalg.norm(a + b + c, dim=1)


class _HeatDiffusion:
    """Gaussian-like smoothing along a surface, with a width in mm that does not depend on the mesh resolution."""

    def __init__(self, surface: cortech.Surface):
        self.mass = surface.voronoi_area()
        # cortech's stiffness matrix holds only the (positive) cotangent weights of the edges
        weights = surface.stiffness_matrix()
        self.stiffness = scipy.sparse.diags_array(np.asarray(weights.sum(1)).ravel()) - weights

    def __call__(self, values: np.ndarray, sigma: float) -> np.ndarray:
        # Each implicit step adds 2t to the variance, so two steps of t = sigma^2 / 4 give a width of sigma
        a = (scipy.sparse.diags_array(self.mass) + sigma**2 / 4 * self.stiffness).tocsr()
        preconditioner = scipy.sparse.diags_array(1 / a.diagonal())
        out = np.array(values, dtype=float).reshape(len(self.mass), -1)
        for j in range(out.shape[1]):
            for _ in range(2):
                out[:, j], _ = scipy.sparse.linalg.cg(
                    a, self.mass * out[:, j], x0=out[:, j], rtol=1e-5, M=preconditioner
                )
        return out.reshape(np.shape(values))


def _sulcal_depth(surface: cortech.Surface, diffusion: _HeatDiffusion, sigma: float) -> np.ndarray:
    """Height of the surface over a smoothed copy of itself, negative in the sulci."""
    smooth = cortech.Surface(diffusion(surface.vertices, sigma), surface.faces)
    return np.sum((surface.vertices - smooth.vertices) * smooth.vertex_normals(), 1)


class _SphereField:
    """Values on the unit sphere that can be sampled differentiably at any point.

    They are stored on two latitude-longitude grids whose poles are
    orthogonal, and each grid is faded out towards its own poles.
    """

    def __init__(self, sphere: np.ndarray, values: np.ndarray, height: int = 512):
        width = 2 * height
        dphi = 2 * np.pi / width
        # One extra column on each side wraps around the date line
        theta, phi = np.meshgrid(
            np.linspace(0, np.pi, height), np.linspace(-np.pi - dphi, np.pi + dphi, width + 3), indexing="ij"
        )
        points = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
        tree = KDTree(sphere)
        self.grids = []
        for axes in ((0, 1, 2), (2, 0, 1)):
            dist, idx = tree.query(points[:, axes], k=4)
            weights = 1 / np.maximum(dist, 1e-9)
            weights /= weights.sum(1, keepdims=True)
            grid = np.einsum("pk,pkc->pc", weights, values[idx]).reshape(height, width + 3, -1)
            self.grids.append(torch.tensor(grid.transpose(2, 0, 1)[None], dtype=torch.float32))
        self.phi_max = np.pi + dphi

    def _sample(self, grid, x):
        theta = torch.acos(torch.clamp(x[:, 2], -1, 1))
        phi = torch.atan2(x[:, 1], x[:, 0])
        coords = torch.stack([phi / self.phi_max, 2 * theta / np.pi - 1], -1)[None, None]
        return torch.nn.functional.grid_sample(grid, coords, align_corners=True)[0, :, 0].T

    def __call__(self, x):
        # The second grid's pole lies on the x axis, so where one weight vanishes the other is 1
        wa = 1 - x[:, 2] ** 2
        wb = 1 - x[:, 0] ** 2
        a = self._sample(self.grids[0], x)
        b = self._sample(self.grids[1], x[:, [1, 2, 0]])
        return (wa[:, None] * a + wb[:, None] * b) / (wa + wb)[:, None]


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / x.std(0)


def _elapsed(start: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))
