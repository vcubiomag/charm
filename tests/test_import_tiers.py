"""Guard the import-cost tiers.

In SimNIBS, importing anything on the CHARM path pulled in the whole
simulation stack. These tests fail if that comes back.
"""

import subprocess
import sys

import pytest

# Anything expensive enough that pulling it in at import time is a regression.
HEAVY = ("numpy", "scipy", "nibabel", "torch", "matplotlib", "cortech")

TIER0_MODULES = [
    "charm",
    "charm.resources",
    "charm.subject",
    "charm.settings",
    "charm.logging",
    "charm.mesh.element_tags",
    "charm.utils.timing",
    "charm.utils.parallel",
    "charm.cli.charm",
]


_MARKER = "___MODULES___"


def imported_modules(statement: str) -> set[str]:
    """Top-level module names loaded by running `statement` in a fresh process."""
    import json

    # The payload is marked because some statements (e.g. --help) also print.
    code = (
        f"{statement}\n"
        "import sys, json\n"
        f"print({_MARKER!r} + json.dumps(sorted({{m.split('.')[0] for m in sys.modules}})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.startswith(_MARKER):
            return set(json.loads(line[len(_MARKER) :]))
    raise AssertionError(f"no module list in output:\n{out.stdout}\n{out.stderr}")


@pytest.mark.parametrize("module", TIER0_MODULES)
def test_tier0_imports_nothing_heavy(module):
    loaded = imported_modules(f"import {module}")
    offenders = sorted(loaded & set(HEAVY))
    assert not offenders, f"import {module} pulled in {offenders}"


@pytest.mark.parametrize("module", ["charm.cli.charm"])
def test_charm_help_is_cheap(module):
    """`charm --help` must not import the scientific stack."""
    loaded = imported_modules(f"import {module} as m\ntry:\n    m.main(['--help'])\nexcept SystemExit:\n    pass")
    offenders = sorted(loaded & set(HEAVY))
    assert not offenders, f"{module} --help pulled in {offenders}"


@pytest.mark.parametrize("module", ["charm.mesh.meshing", "charm.surface.brain_surface", "charm.pipeline.run"])
def test_meshing_path_does_not_need_torch(module):
    """Only surface estimation and TREGA initialization may load torch."""
    loaded = imported_modules(f"import {module}")
    offenders = sorted(loaded & {"torch", "brainnet", "brainsynth"})
    assert not offenders, f"import {module} pulled in {offenders}"
