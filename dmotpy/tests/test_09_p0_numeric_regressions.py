"""Focused regressions for the HBV96 partition and VIC float32 boundary fixes."""
from __future__ import annotations

import unittest

import torch

from models.core.hbv96 import rain_snow_partition
from models.core.vic import vic_step
from models.flux.evap import evap_7
from models.flux.phenology import phenology_2


class TestP0NumericRegressions(unittest.TestCase):
    def test_hbv96_zero_width_partition_has_hard_temperature_boundary(self) -> None:
        precipitation = torch.tensor([6.83, 6.83, 6.83], dtype=torch.float64)
        tt = torch.full((3,), -3.0, dtype=torch.float64)
        tti = torch.zeros(3, dtype=torch.float64)
        temperature = torch.tensor([-3.1, -3.0, -2.9], dtype=torch.float64)

        rain, snow = rain_snow_partition(precipitation, temperature, tt, tti)

        torch.testing.assert_close(rain + snow, precipitation, atol=1e-15, rtol=0.0)
        torch.testing.assert_close(snow, torch.tensor([6.83, 6.83, 0.0], dtype=torch.float64))
        torch.testing.assert_close(rain, torch.tensor([0.0, 0.0, 6.83], dtype=torch.float64))

    def test_hbv96_positive_width_uses_analytic_linear_partition(self) -> None:
        precipitation = torch.tensor([6.83], dtype=torch.float64)
        temperature = torch.tensor([-2.8], dtype=torch.float64)
        tt = torch.tensor([-3.0], dtype=torch.float64)
        tti = torch.tensor([1.0], dtype=torch.float64)
        rain, snow = rain_snow_partition(precipitation, temperature, tt, tti)
        torch.testing.assert_close(snow, torch.tensor([6.83 * 0.3], dtype=torch.float64))
        torch.testing.assert_close(rain + snow, precipitation, atol=1e-15, rtol=0.0)

    def test_hbv96_partition_is_continuous_across_nearzero_width(self) -> None:
        precipitation = torch.tensor([6.83, 6.83], dtype=torch.float64)
        temperature = torch.tensor([-3.0000001, -3.0000001], dtype=torch.float64)
        tt = torch.tensor([-3.0, -3.0], dtype=torch.float64)
        tti = torch.tensor([0.999e-6, 1.001e-6], dtype=torch.float64)
        rain, snow = rain_snow_partition(precipitation, temperature, tt, tti)
        expected = precipitation * torch.clamp(0.5 + (tt - temperature) / tti, 0.0, 1.0)
        torch.testing.assert_close(snow, expected, atol=1e-12, rtol=0.0)
        self.assertLess(abs(float(snow[0] - snow[1])), 2e-3)
        torch.testing.assert_close(rain + snow, precipitation, atol=1e-15, rtol=0.0)

    def test_hbv96_snow_tti_gradient_matches_finite_difference(self) -> None:
        precipitation = torch.tensor([6.83], dtype=torch.float64)
        temperature = torch.tensor([-2.8], dtype=torch.float64)
        tt = torch.tensor([-3.0], dtype=torch.float64)
        tti = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        snow = rain_snow_partition(precipitation, temperature, tt, tti)[1]
        snow.sum().backward()
        self.assertIsNotNone(tti.grad)
        analytic = float(tti.grad)
        self.assertTrue(torch.isfinite(tti.grad).all())
        self.assertAlmostEqual(analytic, 6.83 * 0.2, places=12)

        step = 1e-5
        plus = rain_snow_partition(precipitation, temperature, tt, torch.tensor([1.0 + step], dtype=torch.float64))[1]
        minus = rain_snow_partition(precipitation, temperature, tt, torch.tensor([1.0 - step], dtype=torch.float64))[1]
        finite_difference = float((plus - minus) / (2.0 * step))
        self.assertAlmostEqual(analytic, finite_difference, places=8)

    def test_hbv96_zero_width_backward_is_finite(self) -> None:
        tti = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        precipitation = torch.tensor([6.83], dtype=torch.float64)
        temperature = torch.tensor([-3.1], dtype=torch.float64)
        tt = torch.tensor([-3.0], dtype=torch.float64)
        _, snow = rain_snow_partition(precipitation, temperature, tt, tti)
        snow.sum().backward()
        self.assertIsNotNone(tti.grad)
        self.assertTrue(torch.isfinite(tti.grad).all())
        self.assertEqual(float(tti.grad), 0.0)  # hard branch has structural zero derivative

    def test_hbv96_step_activates_tti_gradient_through_snow_process(self) -> None:
        from models.core.hbv96 import hbv96_step

        dtype = torch.float64
        parameters = [torch.tensor([v], dtype=dtype, requires_grad=True) for v in
                      (-3.0, 1.0, 0.0, 0.2, 1.0, 0.5, 1.0, 100.0, 0.5, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0)]
        states = [torch.tensor([v], dtype=dtype) for v in (1.0, 1.0, 10.0, 5.0, 5.0)]
        outputs = hbv96_step(torch.tensor([5.0], dtype=dtype), torch.tensor([-2.8], dtype=dtype),
                             torch.tensor([0.0], dtype=dtype), *parameters, *states)
        snow_storage = outputs[2] + outputs[3]
        snow_storage.sum().backward()
        tti_grad = parameters[1].grad
        self.assertIsNotNone(tti_grad)
        self.assertTrue(torch.isfinite(tti_grad).all())
        self.assertNotEqual(float(tti_grad), 0.0)

    def test_evap_7_safe_path_preserves_vic_forward_values(self) -> None:
        nearzero = 1e-6
        storage = torch.tensor([nearzero, 2e-6, 0.2, 0.8], dtype=torch.float32)
        capacity = torch.tensor([0.0, 1e-7, 0.2, 0.5], dtype=torch.float32)
        potential = torch.tensor([2.0, 1.0, 0.5, 0.3], dtype=torch.float32)

        actual = evap_7(storage, capacity, potential, safe_denominator=True)
        default = evap_7(storage, capacity, potential)
        legacy = torch.minimum(torch.clamp(storage / capacity, max=1.0) * potential, storage)

        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.equal(actual, legacy))
        self.assertTrue(torch.equal(default, legacy))

    def test_vic_zero_interception_capacity_backward_is_finite(self) -> None:
        dtype = torch.float32
        parameters = [
            torch.tensor([0.2], dtype=dtype, requires_grad=True),
            torch.tensor([1.0], dtype=dtype, requires_grad=True),
            torch.tensor([92.3125], dtype=dtype, requires_grad=True),
            torch.tensor([100.0], dtype=dtype, requires_grad=True),
            torch.tensor([0.5], dtype=dtype, requires_grad=True),
            torch.tensor([2.0], dtype=dtype, requires_grad=True),
            torch.tensor([0.1], dtype=dtype, requires_grad=True),
            torch.tensor([1.0], dtype=dtype, requires_grad=True),
            torch.tensor([0.05], dtype=dtype, requires_grad=True),
            torch.tensor([2.0], dtype=dtype, requires_grad=True),
        ]
        doy = torch.tensor([1.0], dtype=dtype)
        imax = phenology_2(parameters[0], parameters[1], parameters[2], doy,
                           torch.tensor(365.25, dtype=dtype))
        self.assertEqual(float(imax.detach()), 0.0)

        forcing = [
            torch.tensor([5.0], dtype=dtype),
            torch.tensor([10.0], dtype=dtype),
            torch.tensor([2.0], dtype=dtype),
        ]
        states = [
            torch.tensor([1e-6], dtype=dtype),
            torch.tensor([50.0], dtype=dtype),
            torch.tensor([20.0], dtype=dtype),
        ]
        q, ea, *_ = vic_step(*forcing, *parameters, *states, doy=doy)

        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.isfinite(ea).all())
        q.mean().backward()
        self.assertTrue(all(p.grad is not None for p in parameters))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters))


if __name__ == "__main__":
    unittest.main()
