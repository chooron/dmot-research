import unittest

import torch

from dmotpy.models.core.penman import penman_step


DTYPE = torch.float64
DEVICE = torch.device("cpu")
NEARZERO = 1e-6


def tensor(value, *, requires_grad=False):
    return torch.tensor([[value]], dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)


def run_step(s1, s2, s3, gam=0.6, *, p=0.0, pet=0.5, requires_grad=False):
    return penman_step(
        tensor(p),
        tensor(5.0),
        tensor(pet),
        tensor(100.0),
        tensor(0.2),
        tensor(gam, requires_grad=requires_grad),
        tensor(0.1),
        tensor(s1, requires_grad=requires_grad),
        tensor(s2, requires_grad=requires_grad),
        tensor(s3, requires_grad=requires_grad),
        nearzero=NEARZERO,
    )


class PenmanF3Tests(unittest.TestCase):
    def test_lower_et_uses_post_upper_storage_below_threshold_and_not_deficit(self):
        # With PET=0.5, these upper stores leave S1_new at 0.001 and 0.1.
        dry = run_step(0.001, 0.2, 5.0)[1]
        wet = run_step(0.6, 0.2, 5.0)[1]
        # Remove each case's upper-store ET to compare only the lower flux.
        dry_lower = dry - tensor(0.001)
        wet_lower = wet - tensor(0.5)
        self.assertGreater(float(dry_lower), float(wet_lower) + 0.1)
        # Holding upper storage fixed, wet/dry deficit values do not change its gate.
        wet_deficit = run_step(0.001, 1.0, 5.0)[1]
        dry_deficit = run_step(0.001, 50.0, 5.0)[1]
        torch.testing.assert_close(wet_deficit, dry_deficit, rtol=0.0, atol=0.0)

    def test_one_day_signed_storage_balance(self):
        p = tensor(2.0)
        before = (tensor(2.0), tensor(5.0), tensor(4.0))
        q, ea, *after = penman_step(
            p, tensor(5.0), tensor(0.5), tensor(100.0), tensor(0.2),
            tensor(0.4), tensor(0.1), *before, nearzero=NEARZERO,
        )
        # S2 is a deficit store, hence contributes with a negative sign.
        storage_before = before[0] - before[1] + before[2]
        storage_after = after[0] - after[1] + after[2]
        residual = p - q - ea - (storage_after - storage_before)
        self.assertLessEqual(float(residual.abs().max()), 2e-12)

    def test_float64_autograd_matches_central_finite_differences(self):
        gam = tensor(0.6, requires_grad=True)
        s1 = tensor(0.04, requires_grad=True)

        def objective(g, upper):
            q, ea, s1_new, s2_new, s3_new = penman_step(
                tensor(0.0), tensor(5.0), tensor(0.01), tensor(100.0),
                tensor(0.2), g, tensor(0.1), upper, tensor(2.0), tensor(4.0),
                nearzero=NEARZERO,
            )
            return ea + 0.2 * q + 0.01 * (s1_new + s2_new + s3_new)

        value = objective(gam, s1)
        grad_gam, grad_s1 = torch.autograd.grad(value, (gam, s1))
        step = 1e-6
        fd_gam = (objective(tensor(0.6 + step), tensor(0.04)) - objective(tensor(0.6 - step), tensor(0.04))) / (2 * step)
        fd_s1 = (objective(tensor(0.6), tensor(0.04 + step)) - objective(tensor(0.6), tensor(0.04 - step))) / (2 * step)
        self.assertAlmostEqual(float(grad_gam), float(fd_gam), delta=2e-7)
        self.assertAlmostEqual(float(grad_s1), float(fd_s1), delta=2e-7)

    def test_dry_threshold_adjacent_and_wet_outputs_and_gradients_are_finite(self):
        for s1 in (NEARZERO, 0.01 - 1e-7, 0.01, 0.01 + 1e-7, 0.5):
            with self.subTest(s1=s1):
                results = run_step(s1, 2.0, 3.0)
                self.assertTrue(all(torch.isfinite(value).all() for value in results))
                # Inputs used by run_step are intentionally encapsulated; separately
                # exercise gradients through the complete step's sensitive inputs.
                gam = tensor(0.6, requires_grad=True)
                upper = tensor(s1, requires_grad=True)
                deficit = tensor(2.0, requires_grad=True)
                q, ea, s1_new, s2_new, s3_new = penman_step(
                    tensor(0.0), tensor(5.0), tensor(0.5), tensor(100.0),
                    tensor(0.2), gam, tensor(0.1), upper, deficit, tensor(3.0, requires_grad=True),
                    nearzero=NEARZERO,
                )
                loss = q.sum() + ea.sum() + s1_new.sum() + s2_new.sum() + s3_new.sum()
                grads = torch.autograd.grad(loss, (gam, upper, deficit), allow_unused=True)
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
