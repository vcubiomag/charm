from pathlib import Path

import pytest

from charm.resources import templates

TESTDATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    # The tests ported from SimNIBS write scratch files to the working directory
    # and leave them behind when they fail.
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def atlas_v1_1():
    return templates.atlas_path("charm_atlas_mni_v1-1")


@pytest.fixture(scope="session")
def mni_image():
    return str(TESTDATA / "MNI_test_ds5.nii.gz")


@pytest.fixture(scope="session")
def cube_image():
    return str(TESTDATA / "cube.nii.gz")


@pytest.fixture(scope="session")
def noisy_cube_image():
    return str(TESTDATA / "cube_noise.nii.gz")


@pytest.fixture(scope="session")
def cube_atlas():
    return TESTDATA / "cube_atlas"
