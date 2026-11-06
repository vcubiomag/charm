"""The charm command.

The pipeline is imported inside the command so that --help and --version stay cheap. Every option also accepts its
SimNIBS spelling, so existing scripts keep working.
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Group, Parameter, validators
from cyclopts.types import ExistingFile

from charm import __version__
from charm.logging import setup_cli_logging

STEPS = Group("Steps", sort_key=0, help="Run only these parts of the pipeline. Without any, charm runs all of them.")
REGISTRATION = Group("Registration", sort_key=1)
TRANSFORM = Group(
    "Initial transform", sort_key=2, validator=validators.MutuallyExclusive(), help="At most one of these."
)
HEADER = Group("Header", sort_key=3, validator=validators.MutuallyExclusive(), help="At most one of these.")
RUN = Group("Run", sort_key=4)

app = App(
    name="charm",
    version=__version__,
    version_flags=["--version", "-v"],
    default_parameter=Parameter(negative=(), show_default=False),
)


@Parameter(name="*")
@dataclass
class Options:
    """
    Parameters
    ----------
    register_t2
        Register the T2-weighted image to the T1-weighted image.
    init_atlas
        Register the atlas affinely to the input images. Needs the T1-weighted image unless the T2-weighted image is
        registered in the same run.
    segment
        Create the label image and the registration to MNI space.
    surfaces
        Create the cortical surfaces, update the label image, and register to fsaverage.
    mesh
        Create the head mesh from the label image.
    skip_register_t2
        Copy the T2-weighted image instead of registering it.
    no_neck
        The scans don't include the neck.
    init_transform
        4x4 matrix in a space-delimited text file that takes the MNI template to subject space, to start the affine
        registration from. Used when init_type is 'atlas' in the settings.
    use_transform
        4x4 matrix in a space-delimited text file that takes the MNI template to subject space, used instead of the
        affine registration.
    force_qform
        Replace the sform of the inputs with their qform.
    force_sform
        Replace the qform of the inputs with their sform, which drops shears.
    force
        Delete an existing m2m folder when starting from the first step.
    settings
        Settings file to use instead of the default charm.ini.
    debug
        Write the results of intermediate steps.
    """

    register_t2: Annotated[bool, Parameter(alias="--registerT2", group=STEPS)] = False
    init_atlas: Annotated[bool, Parameter(alias="--initatlas", group=STEPS)] = False
    segment: Annotated[bool, Parameter(group=STEPS)] = False
    surfaces: Annotated[bool, Parameter(group=STEPS)] = False
    mesh: Annotated[bool, Parameter(group=STEPS)] = False
    skip_register_t2: Annotated[bool, Parameter(alias="--skipregisterT2", group=REGISTRATION)] = False
    no_neck: Annotated[bool, Parameter(alias="--noneck", group=REGISTRATION)] = False
    init_transform: Annotated[ExistingFile | None, Parameter(alias="--inittransform", group=TRANSFORM)] = None
    use_transform: Annotated[ExistingFile | None, Parameter(alias="--usetransform", group=TRANSFORM)] = None
    force_qform: Annotated[bool, Parameter(alias="--forceqform", group=HEADER)] = False
    force_sform: Annotated[bool, Parameter(alias="--forcesform", group=HEADER)] = False
    force: Annotated[bool, Parameter(alias="--forcerun", group=RUN)] = False
    settings: Annotated[ExistingFile | None, Parameter(alias="--usesettings", metavar="settings.ini", group=RUN)] = None
    debug: Annotated[bool, Parameter(group=RUN)] = False

    def any_step(self) -> bool:
        return self.register_t2 or self.init_atlas or self.segment or self.surfaces or self.mesh


@app.default
def charm(
    subject: str,
    t1: ExistingFile | None = None,
    t2: ExistingFile | None = None,
    /,
    *,
    options: Options | None = None,
) -> None:
    """Create a head mesh from one or more MR images.

    The results go to m2m_SUBJECT in the current directory, with a report in m2m_SUBJECT/charm_report.html. To
    correct the segmentation, edit m2m_SUBJECT/label_prep/tissue_labeling_upsampled.nii.gz and run charm SUBJECT
    --mesh.

    Parameters
    ----------
    subject
        Subject ID.
    t1
        T1-weighted image.
    t2
        T2-weighted image.
    """
    options = options or Options()
    # The run starts from the first step if it registers the T2, or registers the atlas to a newly supplied T1
    fresh_run = options.register_t2 or (options.init_atlas and t1 is not None)
    if not options.any_step():
        fresh_run = True
        options.register_t2 = t2 is not None
        options.init_atlas = options.segment = options.surfaces = options.mesh = True
    if options.skip_register_t2:
        options.register_t2 = False

    if fresh_run and t1 is None:
        sys.exit("charm: the T1-weighted image is required to start a run")
    if options.register_t2 and t2 is None:
        sys.exit("charm: --register-t2 needs the T2-weighted image")

    subject_dir = Path.cwd() / f"m2m_{subject}"
    if subject_dir.exists():
        if fresh_run:
            if not options.force:
                sys.exit(f"charm: {subject_dir} exists; pass --force to replace it")
            if options.settings is not None and options.settings.resolve().is_relative_to(subject_dir.resolve()):
                sys.exit("charm: move the settings file out of the m2m folder before replacing it")
            shutil.rmtree(subject_dir)
        elif options.settings is None:
            options.settings = subject_dir / "settings.ini"
            if not options.settings.exists():
                sys.exit(f"charm: {options.settings} is missing; pass --settings")

    setup_cli_logging()
    from charm.pipeline.run import run

    run(
        str(subject_dir),
        None if t1 is None else str(t1),
        None if t2 is None else str(t2),
        registerT2=options.register_t2,
        initatlas=options.init_atlas,
        segment=options.segment,
        create_surfaces=options.surfaces,
        mesh_image=options.mesh,
        usesettings=None if options.settings is None else str(options.settings),
        noneck=options.no_neck,
        init_transform=None if options.init_transform is None else str(options.init_transform),
        use_transform=None if options.use_transform is None else str(options.use_transform),
        force_qform=options.force_qform,
        force_sform=options.force_sform,
        options_str=" ".join(sys.argv[1:]),
        debug=options.debug,
    )


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    app(argv or ["--help"], result_action="return_none")


if __name__ == "__main__":
    main()
