import os

import pytest

from charm.mesh.element_tags import ElementTags
from charm.subject import HEMISPHERES, SubjectFiles


def test_subid_from_m2m_folder(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    assert sf.subid == "ernie"
    assert os.path.basename(sf.fnamehead) == "ernie.msh"


def test_trailing_separator_does_not_break_subid(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie") + os.sep)
    assert sf.subid == "ernie"


def test_non_m2m_folder_falls_back_to_basename(tmp_path):
    """SimNIBS raised here; charm accepts the directory and uses its name."""
    sf = SubjectFiles(subpath=str(tmp_path / "subject01"))
    assert sf.subid == "subject01"


def test_requires_one_of_fnamehead_or_subpath():
    with pytest.raises(ValueError):
        SubjectFiles()


def test_fnamehead_must_be_msh(tmp_path):
    with pytest.raises(IOError):
        SubjectFiles(fnamehead=str(tmp_path / "head.nii.gz"))


def test_standard_surfaces_are_per_hemisphere(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    for surface in ("central", "pial", "white", "sphere", "sphere.reg"):
        for hemi in HEMISPHERES:
            assert sf.surfaces[surface][hemi].name == f"{hemi}.{surface}.gii"


def test_surface_from_element_tag(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    got = sf.get_surface_from_element_tag(ElementTags.LH_CENTRAL_SURFACE)
    assert got.name == "lh.central.gii"


def test_subsampling_adds_a_directory_level(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    assert sf.get_surface("lh", "central", 10000).parent.name == "10000"
    assert sf.get_surface("lh", "central").parent.name == "surfaces"


def test_sphere_reg_alias(tmp_path):
    sf = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    assert sf.get_surface("lh", "sphere_reg").name == "lh.sphere.reg.gii"
