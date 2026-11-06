"""Locations of the files shipped inside the wheel: the atlas, templates, EEG caps, .ini defaults, colour LUTs
and the HTML report assets.

It must stay cheap to import, since every module that needs a packaged file comes through here. Paths are not
checked for existence on access.
"""

from pathlib import Path


def package_data(*parts: str) -> Path:
    """A file shipped inside the wheel."""
    return Path(__file__).resolve().parent.joinpath("_data", *parts)


def config_file(name: str = "charm.ini") -> Path:
    """The packaged default settings file."""
    return package_data("config", name)


class Templates:
    """Locations of the templates, atlases and report assets charm uses."""

    @property
    def labeling_LUT(self) -> Path:
        """FreeSurfer colour LUT for segmentation/labeling.nii.gz."""
        return package_data("luts", "labeling_FreeSurferColorLUT.txt")

    @property
    def final_tissues_LUT(self) -> Path:
        """FreeSurfer colour LUT for final_tissues.nii.gz."""
        return package_data("luts", "final_tissues_FreeSurferColorLUT.txt")

    @property
    def jquery(self) -> Path:
        return package_data("html", "jquery.min.js")

    @property
    def brainsprite(self) -> Path:
        return package_data("html", "brainsprite.min.js")

    @property
    def report_logo(self) -> Path:
        return package_data("html", "gui_icon.png")

    def atlas_path(self, name: str) -> Path:
        return package_data("atlases", name)

    @property
    def mni_volume(self) -> Path:
        return package_data("templates", "MNI152_T1_1mm.nii.gz")

    @property
    def fsaverage(self) -> Path:
        """HDF5 file of the fsaverage surfaces."""
        return package_data("templates", "fsaverage.h5")

    @property
    def eeg_montage_dir(self) -> Path:
        return package_data("eeg_caps")


templates = Templates()
