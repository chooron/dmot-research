import unittest

import torch

from dmotpy.models.core.penman import penman_step


DTYPE = torch.float64
DEVICE = torch.device("cpu")
NEARZERO = 1e-6


def tensor(value):
    return torch.tensor([[value]], dtype=DTYPE, device=DEVICE)


def run_step(p, pet, s1, s2, s3, *, phi=0.2, gam=0.6, k1=0.1):
    return penman_step(
        tensor(p),
        tensor(5.0),
        tensor(pet),
        tensor(100.0),
        tensor(phi),
        tensor(gam),
        tensor(k1),
        tensor(s1),
        tensor(s2),
        tensor(s3),
        nearzero=NEARZERO,
    )


class PenmanBoundaryBalanceTests(unittest.TestCase):
    def test_zero_and_tiny_stores_are_not_refilled(self):
        for s1, s3 in ((0.0, 0.0), (0.25 * NEARZERO, 0.5 * NEARZERO)):
            with self.subTest(s1=s1, s3=s3):
                q, ea, s1_new, s2_new, s3_new = run_step(
                    0.0, 0.0, s1, 1.0, s3
                )
                self.assertEqual(float(s1_new), s1)
                self.assertGreaterEqual(float(s3_new), 0.0)
                if s3 == 0.0:
                    self.assertEqual(float(q), 0.0)
                    self.assertEqual(float(s3_new), 0.0)
                else:
                    self.assertLessEqual(float(s3_new), s3)
                self.assertEqual(float(ea), 0.0)
                self.assertGreaterEqual(float(s1_new), 0.0)

    def test_zero_tiny_and_wet_states_close_in_one_step(self):
        for p, pet, s1, s2, s3 in (
            (0.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.25e-6, 1.0, 0.5e-6),
            (2.0, 0.5, 2.0, 5.0, 4.0),
        ):
            with self.subTest(p=p, s1=s1, s3=s3):
                q, ea, s1_new, s2_new, s3_new = run_step(p, pet, s1, s2, s3)
                residual = p - float(q) - float(ea) - (
                    float(s1_new - tensor(s1))
                    - float(s2_new - tensor(s2))
                    + float(s3_new - tensor(s3))
                )
                self.assertLessEqual(abs(residual), 2e-12)
                self.assertGreaterEqual(float(s1_new), 0.0)
                self.assertGreaterEqual(float(s3_new), 0.0)

    def test_wet_state_fluxes_and_s2_deficit_convention_are_preserved(self):
        q, ea, s1_new, s2_new, s3_new = run_step(2.0, 0.5, 2.0, 5.0, 4.0)
        self.assertGreater(float(q), 0.0)
        self.assertGreater(float(ea), 0.0)
        self.assertGreater(float(s1_new), NEARZERO)
        self.assertGreater(float(s3_new), NEARZERO)
        # With no recharge in the lower zone, PET-driven lower ET increases deficit.
        self.assertGreaterEqual(float(s2_new), 0.0)
        # With no recharge, lower-zone ET increases the deficit coordinate.
        _, _, _, s2_dry, _ = run_step(0.0, 0.5, 0.0, 5.0, 0.0)
        self.assertGreater(float(s2_dry), 5.0)

if __name__ == "__main__":
    unittest.main()
