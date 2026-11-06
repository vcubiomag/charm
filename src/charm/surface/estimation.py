"""Cortical surface estimation with the brainnet models.

Importing this module loads torch, so it is kept apart from
charm.surface.brain_surface, which the meshing path needs.
"""

import brainnet
import brainnet.helpers
import cortech
import numpy as np
import numpy.typing as npt
import scipy.io
import torch

from charm.subject import SubjectFiles

MNI305_to_MNI152 = brainnet.datasets.MNI305_to_MNI152.numpy()


class SubjectImageDataset(brainnet.datasets.ImageDataset):
    def __init__(
        self,
        m2m_dirs: list[SubjectFiles] | tuple,
        image: str = "reference_volume",
        **kwargs,
    ):
        images = [getattr(m2m, image) for m2m in m2m_dirs]
        super().__init__(images, **kwargs)


class SubjectTopoFitDataset(brainnet.datasets.TopoFitDataset):
    def __init__(
        self,
        m2m_dirs: list[SubjectFiles] | tuple,
        image: str = "reference_volume",
        **kwargs,
    ):
        images = [getattr(m2m, image) for m2m in m2m_dirs]
        mni_transforms = [m2m.coregistration_matrices for m2m in m2m_dirs]
        kwargs |= {"mni_transforms": mni_transforms, "mni_direction": "mni2sub", "mni_space": "mni152"}
        super().__init__(images, **kwargs)

        self.m2m_dirs = m2m_dirs


def _load_mni_transform(filename):
    # worldToWorldTransformMatrix is mni to subject space
    return scipy.io.loadmat(filename)["worldToWorldTransformMatrix"]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve "auto" to a GPU when there is one, since it runs these models an order of magnitude faster."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def estimate_mni152_to_ras_affine(
    m2m_dirs: list[SubjectFiles] | tuple,
    transform: str = "brain",
    device: str | torch.device = "auto",
):
    def _get_mni152_to_ras(v):
        return MNI305_to_MNI152 @ v.squeeze(0).cpu().numpy()

    dataset = SubjectImageDataset(m2m_dirs, conform=True)
    step = brainnet.helpers.trega.PredictionStep.from_pretrained(device=resolve_device(device))
    return [_get_mni152_to_ras(step(None, batch)[transform]) for batch in dataset]


def _to_Surface(s):
    s.to("cpu")
    return cortech.Surface(s.vertices.squeeze(0).numpy(), s.topology.faces.numpy())


def _to_Hemisphere(hemi, surfaces):
    return cortech.Hemisphere(hemi, **{k: _to_Surface(v) for k, v in surfaces.items()})


def _extract_vertex_data(y_pred: dict[str, dict[str, brainnet.Surface]]):
    def _convert_vertex_data(v: brainnet.Surface):
        def _norm_if_2d(v: torch.Tensor):
            v = v.squeeze(0).cpu().numpy()
            assert v.ndim <= 2
            return np.linalg.norm(v, axis=1) if v.ndim == 2 else v

        return {name: _norm_if_2d(value) for name, value in v.vertex_data.items()}

    return {h: {s: _convert_vertex_data(vv) for s, vv in v.items()} for h, v in y_pred.items()}


def cortical_surface_estimation(
    m2m_dirs: list[SubjectFiles] | tuple,
    contrast: str = "T1w",
    resolution: str = "1mm",
    device: str | torch.device = "auto",
) -> tuple[list[cortech.Cortex], list[dict[str, dict[str, npt.NDArray]]]]:
    """Estimate the white and pial surfaces of each subject with TopoFit.

    Parameters
    ----------
    m2m_dirs: list[SubjectFiles]
    contrast: str
        The contrast the model was trained on (synth or t1w).
    resolution: str
        The resolution the model was trained on (1mm or random).
    device: str or torch.device
        See resolve_device.

    Returns
    -------
    One cortech.Cortex per subject, and per subject the vertex data as
    {hemi: {surface: {name: array}}}.
    """
    contrast = contrast.lower()
    resolution = resolution.lower()
    device = resolve_device(device)

    dataset = SubjectTopoFitDataset(m2m_dirs, conform=True, mni_transform_loader=_load_mni_transform)
    step = brainnet.helpers.topofit.PredictionStep.from_pretrained(contrast, resolution, device)

    predictions = []
    vertex_data = []
    for batch in dataset:
        y_pred = step(None, batch)
        y_pred = step.model.swap_output_levels(y_pred)  # {hemi: surf: ...}
        cortex = cortech.Cortex(**{h: _to_Hemisphere(h, surfs) for h, surfs in y_pred.items()})
        predictions.append(cortex)
        vertex_data.append(_extract_vertex_data(y_pred))
    return predictions, vertex_data
