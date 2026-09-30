"""CPU-only tests for ExpertClimb's Torch mass/COM kinematics."""

import importlib.util
from pathlib import Path

import unittest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "kinematic.py"
_SPEC = importlib.util.spec_from_file_location("mass_kinematic", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Kinematic = _MODULE.Kinematic

LEGS = ("lb", "lf", "lm", "rb", "rf", "rm")
LINKS = ("thigh", "knee", "ankle", "foot", "ball1", "ball2", "suck", "toe", "empty")
NAMES = ["body"] + [f"l_{leg}_{link}" for leg in LEGS for link in LINKS]


def _model(mass_scale=(1.0, 2.0)):
    masses = torch.ones(len(mass_scale), len(NAMES))
    for index, scale in enumerate(mass_scale):
        masses[index] *= scale
    local_com = torch.zeros(len(NAMES), 3)
    local_com[0] = torch.tensor((0.01, -0.02, 0.03))
    # Give the distal chain non-zero COM offsets so q4 changes the whole COM.
    for index, name in enumerate(NAMES):
        if name.endswith(("foot", "ball1", "ball2", "suck", "toe", "empty")):
            local_com[index] = torch.tensor((0.02, 0.0, 0.01))
    kin = Kinematic(0.072, 0.13, 0.17, device="cpu", body_x=0.1, body_y=0.22)
    kin.configure_mass_model(NAMES, masses, local_com)
    return kin, masses


class TestMassKinematic(unittest.TestCase):
    def test_uniform_mass_scale_changes_total_but_not_com(self):
        kin, masses = _model()
        total_mass, center = kin.mass_properties(torch.zeros(2, 6, 4))
        self.assertTrue(torch.allclose(total_mass[:, 0], masses.sum(dim=1)))
        self.assertTrue(torch.allclose(center[0], center[1]))

    def test_joint_configuration_changes_dynamic_com(self):
        kin, _ = _model((1.0,))
        neutral = torch.zeros(1, 6, 4)
        moved = neutral.clone()
        moved[0, 5, 3] = 0.5
        _, neutral_com = kin.mass_properties(neutral)
        _, moved_com = kin.mass_properties(moved)
        self.assertFalse(torch.allclose(neutral_com, moved_com))

    def test_mass_distribution_changes_com_independently_per_environment(self):
        kin, masses = _model((1.0, 1.0))
        masses[1, NAMES.index("l_rm_foot")] *= 10.0
        local_com = torch.zeros(len(NAMES), 3)
        local_com[NAMES.index("l_rm_foot")] = torch.tensor((0.02, 0.0, 0.01))
        kin.configure_mass_model(NAMES, masses, local_com)
        _, center = kin.mass_properties(torch.zeros(2, 6, 4))
        self.assertFalse(torch.allclose(center[0], center[1]))

    def test_mass_model_rejects_missing_link_and_bad_joint_shape(self):
        kin, masses = _model((1.0,))
        with self.assertRaisesRegex(ValueError, "missing"):
            kin.configure_mass_model(NAMES[:-1], masses[:, :-1], torch.zeros(len(NAMES) - 1, 3))
        with self.assertRaisesRegex(ValueError, "drive_joints"):
            kin.mass_properties(torch.zeros(1, 6, 3))
