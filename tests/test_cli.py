import pytest

from charm.cli import charm


@pytest.fixture
def images(tmp_path):
    for name in ("t1.nii.gz", "t2.nii.gz", "settings.ini", "transform.txt"):
        (tmp_path / name).touch()


def options(*tokens):
    _, bound, _ = charm.app.parse_args(["ernie", *tokens])
    return bound.kwargs.get("options") or charm.Options()


def test_no_arguments_print_help(capsys):
    charm.main([])
    assert "usage" in capsys.readouterr().out.lower()


def test_version(capsys):
    charm.main(["--version"])
    assert capsys.readouterr().out.strip()


def test_fresh_run_requires_t1(images):
    with pytest.raises(SystemExit, match="T1-weighted"):
        charm.main(["ernie"])


def test_register_t2_requires_t2(images):
    with pytest.raises(SystemExit, match="T2-weighted"):
        charm.main(["ernie", "t1.nii.gz", "--register-t2"])


def test_existing_folder_requires_force(images, tmp_path):
    (tmp_path / "m2m_ernie").mkdir()
    with pytest.raises(SystemExit, match="--force"):
        charm.main(["ernie", "t1.nii.gz"])


FLAGS = {
    "register_t2": ("--register-t2", "--registerT2"),
    "init_atlas": ("--init-atlas", "--initatlas"),
    "segment": ("--segment",),
    "mesh": ("--mesh",),
    "surfaces": ("--surfaces",),
    "force": ("--force", "--forcerun"),
    "skip_register_t2": ("--skip-register-t2", "--skipregisterT2"),
    "no_neck": ("--no-neck", "--noneck"),
    "force_qform": ("--force-qform", "--forceqform"),
    "force_sform": ("--force-sform", "--forcesform"),
    "debug": ("--debug",),
}


@pytest.mark.parametrize("attribute,flag", [(a, f) for a, flags in FLAGS.items() for f in flags])
def test_flags_and_their_simnibs_spellings(attribute, flag):
    assert getattr(options(flag), attribute) is True


@pytest.mark.parametrize(
    "attribute,flag,file",
    [
        ("settings", "--settings", "settings.ini"),
        ("settings", "--usesettings", "settings.ini"),
        ("init_transform", "--init-transform", "transform.txt"),
        ("init_transform", "--inittransform", "transform.txt"),
        ("use_transform", "--use-transform", "transform.txt"),
        ("use_transform", "--usetransform", "transform.txt"),
    ],
)
def test_file_options(images, tmp_path, attribute, flag, file):
    assert getattr(options(flag, file), attribute).resolve() == (tmp_path / file).resolve()


def test_images_are_positional(images):
    _, bound, _ = charm.app.parse_args(["ernie", "t1.nii.gz", "t2.nii.gz"])
    assert [str(a) for a in bound.args] == ["ernie", "t1.nii.gz", "t2.nii.gz"]


@pytest.mark.parametrize(
    "tokens",
    [
        ["a", "b", "c", "d"],
        ["a", "--nosuch"],
        ["a", "--no-mesh"],
        ["a", "missing.nii.gz"],
        ["a", "--force-qform", "--force-sform"],
        ["a", "--init-transform", "transform.txt", "--use-transform", "transform.txt"],
    ],
)
def test_invalid_command_lines_are_rejected(images, tokens):
    with pytest.raises(SystemExit) as exc:
        charm.main(tokens)
    assert exc.value.code != 0
