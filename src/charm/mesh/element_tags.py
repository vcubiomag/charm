from enum import IntEnum


class ElementTypes(IntEnum):
    LINE = 1
    TRIANGLE = 2
    TETRAHEDRON = 4
    POINT = 15


class ElementTags(IntEnum):
    @classmethod
    def from_surface_file_name(cls, surface_name: str, hemisphere: str):
        tag_from_name = {
            ("gray", "lh"): cls.LH_PIAL_SURFACE,
            ("gray", "rh"): cls.RH_PIAL_SURFACE,
            ("pial", "lh"): cls.LH_PIAL_SURFACE,
            ("pial", "rh"): cls.RH_PIAL_SURFACE,
            ("white", "lh"): cls.LH_WM_SURFACE,
            ("white", "rh"): cls.RH_WM_SURFACE,
            ("sphere", "lh"): cls.LH_SPHERE,
            ("sphere", "rh"): cls.RH_SPHERE,
            ("sphere.reg", "lh"): cls.LH_SPHERE_REG,
            ("sphere.reg", "rh"): cls.RH_SPHERE_REG,
            ("central", "lh"): cls.LH_CENTRAL_SURFACE,
            ("central", "rh"): cls.RH_CENTRAL_SURFACE,
        }
        return tag_from_name[(surface_name.lower(), hemisphere.lower())]

    TH_START = 0
    WM = 1
    GM = 2
    CSF = 3
    BONE = 4
    SCALP = 5
    EYE_BALLS = 6
    COMPACT_BONE = 7
    SPONGY_BONE = 8
    BLOOD = 9
    MUSCLE = 10
    CARTILAGE = 11
    FAT = 12
    CREAM = 999
    TH_END = 999

    TH_SURFACE_START = 1000
    WM_TH_SURFACE = TH_SURFACE_START + WM
    GM_TH_SURFACE = TH_SURFACE_START + GM
    CSF_TH_SURFACE = TH_SURFACE_START + CSF
    BONE_TH_SURFACE = TH_SURFACE_START + BONE
    SCALP_TH_SURFACE = TH_SURFACE_START + SCALP
    EYE_BALLS_TH_SURFACE = TH_SURFACE_START + EYE_BALLS
    COMPACT_BONE_TH_SURFACE = TH_SURFACE_START + COMPACT_BONE
    SPONGY_BONE_TH_SURFACE = TH_SURFACE_START + SPONGY_BONE
    BLOOD_TH_SURFACE = TH_SURFACE_START + BLOOD
    MUSCLE_TH_SURFACE = TH_SURFACE_START + MUSCLE
    CARTILAGE_TH_SURFACE = TH_SURFACE_START + CARTILAGE
    FAT_TH_SURFACE = TH_SURFACE_START + FAT
    INTERNAL_AIR_TH_SURFACE = TH_SURFACE_START + 99

    LH_SURFACE_START = 5000
    LH_WM_SURFACE = LH_SURFACE_START + WM
    LH_PIAL_SURFACE = LH_SURFACE_START + GM
    LH_CENTRAL_SURFACE = LH_SURFACE_START + GM + 1
    LH_SPHERE = LH_SURFACE_START + 100
    LH_SPHERE_REG = LH_SURFACE_START + 101
    LH_SURFACE_END = 5499

    RH_SURFACE_START = 7000
    RH_WM_SURFACE = RH_SURFACE_START + WM
    RH_PIAL_SURFACE = RH_SURFACE_START + GM
    RH_CENTRAL_SURFACE = RH_SURFACE_START + GM + 1
    RH_SPHERE = RH_SURFACE_START + 100
    RH_SPHERE_REG = RH_SURFACE_START + 101
    RH_SURFACE_END = 7499

    UNKNOWN_SURFACE = 9999
