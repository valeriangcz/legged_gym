"""Unit tests for ``ExpertClimb`` quasi-static feedforward torque."""
import unittest

# Isaac Gym must be imported before torch in this project environment.
from isaacgym import gymapi  # noqa: F401
import torch

from legged_gym.envs.hex_v4.expert import ExpertClimb
from legged_gym.envs.hex_v4.hex_climb_config import HexClimbCfg


class ExpertFeedforwardTest(unittest.TestCase):
    def setUp(self):
        self.expert = ExpertClimb(HexClimbCfg(), "cpu", 1)
        self.q_cur = self.expert.q_init.view(1,6,4)[...,:3].reshape(1,18)
        self.gravity_R = torch.tensor([[0.0,0.0,-9.81]])
        command = torch.zeros(1,5)
        command[:,0] = 1.0
        self.adhesions, self.q_des, self.tau_ff = self.expert.ProcessCommand(
            command,self.q_cur,torch.zeros_like(self.q_cur),torch.zeros(1,6),
            torch.zeros(1,6),self.gravity_R
        )

    def test_static_force_balance_and_swing_torque(self):
        forces_R = self.expert.static_contact_forces_R
        r_R = self.expert._B2R(self.expert.B_e_cur)[...,:3]
        r_R = r_R-self.expert.R_body_com.view(1,1,3)
        expected_force = -self.expert.total_mass*self.gravity_R

        self.assertTrue(torch.allclose(forces_R.sum(dim=1),expected_force,atol=2e-3))
        self.assertTrue(torch.allclose(
            torch.cross(r_R,forces_R,dim=-1).sum(dim=1),torch.zeros(1,3),atol=2e-3
        ))
        self.assertEqual(tuple(self.q_des.shape),(1,24))
        self.assertEqual(tuple(self.tau_ff.shape),(1,18))
        self.assertTrue(torch.equal(
            self.tau_ff.view(1,6,3)[~self.expert.gaits],
            torch.zeros_like(self.tau_ff.view(1,6,3)[~self.expert.gaits]),
        ))

    def test_limit_and_gait_transition_smoothing(self):
        for _ in range(self.expert._tau_ff_transition_steps):
            self.expert._ComputeQuasiStaticTauFF(self.q_cur.view(6,3),self.gravity_R)
        settled_tau = self.expert.tau_ff.clone()
        self.assertLessEqual(settled_tau.abs().max().item(),self.expert.motor_torque_limit)

        self.expert.gaits.zero_()
        self.expert.gaits[:,self.expert.B_group_index] = True
        self.expert._ComputeQuasiStaticTauFF(self.q_cur.view(6,3),self.gravity_R)
        self.assertEqual(self.expert._tau_ff_transition_remaining.item(),
                         self.expert._tau_ff_transition_steps-1)
        self.assertFalse(torch.equal(self.expert.tau_ff,settled_tau))
        self.assertTrue(torch.isfinite(self.expert.tau_ff).all())


if __name__ == "__main__":
    unittest.main()
