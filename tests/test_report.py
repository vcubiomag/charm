import shutil

import nibabel as nib
import numpy as np
from conftest import TESTDATA

from charm.report import html_writer
from charm.resources import config_file
from charm.subject import SubjectFiles
from charm.utils import nifti


def test_write_report(tmp_path):
    sub_files = SubjectFiles(subpath=str(tmp_path / "m2m_ernie"))
    (tmp_path / "m2m_ernie").mkdir()

    t1 = nifti.load(TESTDATA / "ernie_T1_ds5.nii.gz")
    nib.save(t1, sub_files.reference_volume)
    labels = (np.asarray(t1.dataobj) > 50).astype(np.uint16)
    nib.save(nib.Nifti1Image(labels, t1.affine), sub_files.final_labels)
    shutil.copyfile(config_file("charm.ini"), sub_files.settings)

    html_writer.write_report(sub_files)

    with open(sub_files.summary_report) as f:
        report = f.read()
    assert "Tissue labels" in report
    assert "brainsprite" in report
    assert "{simnibs_fig}" not in report
