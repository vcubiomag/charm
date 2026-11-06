"""Per-subject output layout.

``SubjectFiles`` is pure string composition over an ``m2m_<subid>`` directory;
it does not check that any of the files exist.
"""

import os
import re
from pathlib import Path

from charm.mesh.element_tags import ElementTags

# The order of this tuple is the hemisphere order used throughout charm.
HEMISPHERES = ("lh", "rh")

SURFACE_FILE_NAME_TO_ELEMENT_TAG = {
    "lh.sphere.reg.gii": ElementTags.LH_SPHERE_REG,
    "lh.sphere.gii": ElementTags.LH_SPHERE,
    "lh.pial.gii": ElementTags.LH_PIAL_SURFACE,
    "lh.white.gii": ElementTags.LH_WM_SURFACE,
    "lh.central.gii": ElementTags.LH_CENTRAL_SURFACE,
    "rh.sphere.reg.gii": ElementTags.RH_SPHERE_REG,
    "rh.sphere.gii": ElementTags.RH_SPHERE,
    "rh.pial.gii": ElementTags.RH_PIAL_SURFACE,
    "rh.white.gii": ElementTags.RH_WM_SURFACE,
    "rh.central.gii": ElementTags.RH_CENTRAL_SURFACE,
}
ELEMENT_TAG_TO_SURFACE_FILE_NAME = {v: k for k, v in SURFACE_FILE_NAME_TO_ELEMENT_TAG.items()}


class SubjectFiles:
    """Locations of the files charm writes for one subject.

    Parameters
    ------------
    fnamehead: str (optional)
        Path to the head mesh.

    subpath: str (optional)
        Path to the m2m folder. Takes precedence over fnamehead when locating
        other files.

    At least one of fnamehead or subpath must be set.

    Notes
    -----
    The subject ID is taken from an ``m2m_<subid>`` directory name. Unlike
    SimNIBS, a directory that does not follow that convention is accepted and
    its basename is used as the ID. The CLI still creates ``m2m_`` folders.

    This class does not check that the files exist.
    """

    def __init__(self, fnamehead: str | None = None, subpath: str | None = None):
        if not fnamehead and not subpath:
            raise ValueError("Either fnamehead or subpath need to be set")

        self.subpath = None
        if fnamehead:
            if not str(fnamehead).endswith(".msh"):
                raise OSError("fnamehead must be a gmsh .msh file")
            self.fnamehead = os.path.normpath(os.path.expanduser(fnamehead))
            if not subpath:
                head_dir, subid = os.path.split(self.fnamehead)
                self.subid = os.path.splitext(subid)[0]
                self.subpath = os.path.normpath(os.path.expanduser(head_dir))
                if not re.search("m2m_(.+)", self.subpath):
                    self.subpath = None

        if subpath:
            self.subpath = os.path.normpath(os.path.expanduser(subpath))
            folder_name = os.path.basename(self.subpath.rstrip(os.sep))
            match = re.fullmatch("m2m_(.+)", folder_name)
            self.subid = match.group(1) if match else folder_name
            if not fnamehead:
                self.fnamehead = os.path.join(self.subpath, self.subid + ".msh")

        # Preparation fails for meshes outside an m2m folder if this stays None.
        if not self.subpath:
            self.subpath = ""

        self.reference_volume = os.path.join(self.subpath, "T1.nii.gz")
        self.T2_reg = os.path.join(self.subpath, "T2_reg.nii.gz")

        self.settings = os.path.join(self.subpath, "settings.ini")
        self.charm_log = os.path.join(self.subpath, "charm_log.html")
        self.summary_report = os.path.join(self.subpath, "charm_report.html")

        self.final_labels = os.path.join(self.subpath, "final_tissues.nii.gz")

        self.mni_transf_folder = os.path.join(self.subpath, "toMNI")

        self.mni2conf_nonl = os.path.join(self.mni_transf_folder, "MNI2Conform_nonl.nii.gz")
        self.conf2mni_nonl = os.path.join(self.mni_transf_folder, "Conform2MNI_nonl.nii.gz")
        self.final_labels_MNI = os.path.join(self.mni_transf_folder, "final_tissues_MNI.nii.gz")

        self.segmentation_folder = os.path.join(self.subpath, "segmentation")

        self.coregistration_matrices = os.path.join(self.segmentation_folder, "coregistrationMatrices.mat")

        self.T1_denoised = os.path.join(self.segmentation_folder, "T1_denoised.nii.gz")
        self.T2_reg_denoised = os.path.join(self.segmentation_folder, "T2_reg_denoised.nii.gz")
        self.T1_bias_corrected = os.path.join(self.segmentation_folder, "T1_bias_corrected.nii.gz")
        self.T2_bias_corrected = os.path.join(self.segmentation_folder, "T2_bias_corrected.nii.gz")
        self.labeling = os.path.join(self.segmentation_folder, "labeling.nii.gz")
        self.template_coregistered = os.path.join(self.segmentation_folder, "template_coregistered.mgz")

        self.label_prep_folder = os.path.join(self.subpath, "label_prep")

        self.T1_upsampled = os.path.join(self.label_prep_folder, "T1_upsampled.nii.gz")
        self.T2_upsampled = os.path.join(self.label_prep_folder, "T2_upsampled.nii.gz")
        self.tissue_labeling_upsampled = os.path.join(self.label_prep_folder, "tissue_labeling_upsampled.nii.gz")
        self.tissue_labeling_before_morpho = os.path.join(self.label_prep_folder, "before_morpho.nii.gz")
        self.upper_mask = os.path.join(self.label_prep_folder, "upper_part.nii.gz")

        self.surface_folder = os.path.join(self.subpath, "surfaces")

        self.hemispheres = HEMISPHERES

        self._standard_surfaces = ("central", "pial", "white", "sphere", "sphere.reg")
        self.surfaces = {s: {h: self.get_surface(h, s) for h in self.hemispheres} for s in self._standard_surfaces}

        self.eeg_cap_folder = os.path.join(self.subpath, "eeg_positions")

    def get_surface_from_element_tag(self, element_tag, subsampling=None) -> Path:
        subsampling = self._parse_subsampling(subsampling)
        return Path(self.surface_folder) / subsampling / ELEMENT_TAG_TO_SURFACE_FILE_NAME[element_tag]

    def get_surface(self, hemi, surface, subsampling=None) -> Path:
        """Path of a surface file, e.g. central, pial, sphere or sphere.reg."""
        if surface == "sphere_reg":
            surface = "sphere.reg"  # backwards compatibility
        subsampling = self._parse_subsampling(subsampling)
        return Path(self.surface_folder) / subsampling / f"{hemi}.{surface}.gii"

    @staticmethod
    def _parse_subsampling(subsampling: None | int) -> str:
        return str(subsampling) if subsampling else ""
