from charm.resources import config_file, templates


def test_packaged_config_files_exist():
    assert config_file("charm.ini").is_file()


def test_bundled_data_exists():
    assert (templates.atlas_path("charm_atlas_mni_v1-1") / "template.nii.gz").is_file()
    assert templates.mni_volume.is_file()
    assert templates.fsaverage.is_file()
    assert any(templates.eeg_montage_dir.glob("*.csv"))
