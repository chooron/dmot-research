"""Focused GSFB F4 checks for preserving the physical recharge threshold."""

import inspect
import unittest

import torch
import torch.nn.functional as F

from dmotpy.models.core import gsfb
from dmotpy.models.flux import recharge as shared_recharge


class GsfbF4ThresholdTests(unittest.TestCase):
    def test_physical_threshold_and_unchanged_softplus_profile(self):
        dtype = torch.float64
        coefficient = torch.tensor(0.4, dtype=dtype)
        source = torch.tensor(3.0, dtype=dtype)
        receiving = torch.tensor(0.5, dtype=dtype)

        # Physical thresholds at/below/above 1 mm; no threshold is rewritten.
        for threshold_value in (0.05, 1.0, 2.0):
            threshold = torch.tensor(threshold_value, dtype=dtype)
            observed = gsfb._gsfb_recharge_5(
                coefficient, threshold, source, receiving
            )
            reference = coefficient * source * F.softplus(
                1.0 - receiving / threshold, beta=10.0
            )
            torch.testing.assert_close(observed, reference, rtol=0.0, atol=1e-15)

        # Legal minimum ndc*Smax is 0.05 mm. The legacy 1-mm floor materially
        # changes this case, while the local law retains the 0.05-mm threshold.
        threshold = torch.tensor(0.05, dtype=dtype)
        candidate = gsfb._gsfb_recharge_5(coefficient, threshold, source, receiving)
        old_floored = shared_recharge.recharge_5(
            coefficient, threshold, source, receiving
        )
        self.assertGreater(abs((candidate - old_floored).item()), 0.5)

    def test_daily_order_and_local_helper_scope(self):
        source = inspect.getsource(gsfb.gsfb_step)
        order = [
            "# 1. Saturated zone → Soil moisture recharge",
            "# 2. Soil Moisture Store  (S1)",
            "# 3. Intermediate Store  (S2)",
            "# 4. Saturated Zone Store  (S3)",
            "# 5. Output",
        ]
        positions = [source.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("_gsfb_recharge_5(c, threshold_s1, S3, S1)", source)
        self.assertNotIn("flux_qdr_pot = recharge_5(", source)
        module_source = inspect.getsource(gsfb)
        self.assertNotIn("from ..flux.recharge import recharge_5", module_source)
        shared_source = inspect.getsource(shared_recharge.recharge_5)
        self.assertIn("torch.clamp(p2, min=1.0)", shared_source)

    def test_daily_water_balance(self):
        dtype = torch.float64
        params = [torch.tensor(v, dtype=dtype) for v in
                  (0.2, 0.05, 1.0, 4.0, 3.0, 0.4, 0.08, 0.2)]
        c, ndc, smax, emax, frate, b, dpf, sdrmax = params
        stores = [torch.tensor(v, dtype=dtype) for v in (0.4, 0.8, 1.5)]
        max_residual = 0.0
        for precipitation, pet in ((0.0, 0.1), (0.7, 0.2), (2.0, 0.0), (0.1, 0.4)):
            p = torch.tensor(precipitation, dtype=dtype)
            t = torch.tensor(5.0, dtype=dtype)
            pet_t = torch.tensor(pet, dtype=dtype)
            old_total = sum(x.item() for x in stores)
            q, ea, *new_stores = gsfb.gsfb_step(
                p, t, pet_t, c, ndc, smax, emax, frate, b, dpf, sdrmax,
                *stores,
            )
            new_total = sum(x.item() for x in new_stores)
            residual = new_total - old_total - precipitation + ea.item() + q.item()
            max_residual = max(max_residual, abs(residual))
            stores = new_stores
        self.assertLessEqual(max_residual, 1e-12)

    def test_float64_autograd_matches_finite_difference_and_extremes_are_finite(self):
        dtype = torch.float64
        c = torch.tensor(0.37, dtype=dtype, requires_grad=True)
        threshold = torch.tensor(0.05, dtype=dtype, requires_grad=True)
        source = torch.tensor(1.5, dtype=dtype, requires_grad=True)
        receiving = torch.tensor(0.02, dtype=dtype, requires_grad=True)
        result = gsfb._gsfb_recharge_5(c, threshold, source, receiving)
        gradients = torch.autograd.grad(result, (c, threshold, source, receiving))
        self.assertTrue(torch.isfinite(result).item())
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
        h = 1e-6
        plus = gsfb._gsfb_recharge_5(
            c.detach(), threshold.detach() + h, source.detach(), receiving.detach()
        )
        minus = gsfb._gsfb_recharge_5(
            c.detach(), threshold.detach() - h, source.detach(), receiving.detach()
        )
        fd = (plus - minus) / (2.0 * h)
        self.assertAlmostEqual(gradients[1].item(), fd.item(), delta=2e-8)

        for ndc_value, smax_value in ((0.05, 1.0), (0.95, 2000.0)):
            extreme_threshold = torch.tensor(ndc_value * smax_value, dtype=dtype,
                                             requires_grad=True)
            value = gsfb._gsfb_recharge_5(
                torch.tensor(200.0, dtype=dtype), extreme_threshold,
                torch.tensor(2000.0, dtype=dtype), torch.tensor(0.0, dtype=dtype),
            )
            gradient = torch.autograd.grad(value, extreme_threshold)[0]
            self.assertTrue(torch.isfinite(value).item())
            self.assertTrue(torch.isfinite(gradient).item())


if __name__ == "__main__":
    unittest.main()
