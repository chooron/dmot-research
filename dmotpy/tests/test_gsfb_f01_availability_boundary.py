import unittest

import torch

from dmotpy.models.core.gsfb import _cap_available_flux, gsfb_step
from dmotpy.models.flux.smooth import smooth_cap_flux


class TestGsfbF01AvailabilityBoundary(unittest.TestCase):
    dtype = torch.float64

    def tensor(self, value, requires_grad=False):
        return torch.tensor([value], dtype=self.dtype, requires_grad=requires_grad)

    def step(self, P, PET, S1, S2, S3):
        return gsfb_step(
            P=self.tensor(P),
            T=self.tensor(5.0),
            PET=self.tensor(PET),
            c=self.tensor(0.2),
            ndc=self.tensor(0.3),
            smax=self.tensor(100.0),
            emax=self.tensor(10.0),
            frate=self.tensor(2.0),
            b=self.tensor(0.5),
            dpf=self.tensor(0.1),
            sdrmax=self.tensor(10.0),
            S1=self.tensor(S1),
            S2=self.tensor(S2),
            S3=self.tensor(S3),
        )

    def test_positive_potential_zero_and_tiny_availability(self):
        potential = self.tensor(2.0)
        availability = self.tensor(0.0)
        self.assertEqual(_cap_available_flux(potential, availability, 1e-3).item(), 0.0)

        tiny = self.tensor(1e-12)
        q = _cap_available_flux(potential, tiny, 1e-3)
        self.assertGreaterEqual(q.item(), 0.0)
        self.assertLessEqual(q.item(), tiny.item())
        self.assertEqual(q.item(), tiny.item())

    def test_zero_potential_preserves_smooth_cap_profile(self):
        potential = self.tensor(0.0)
        availability = self.tensor(1.0)
        actual = _cap_available_flux(potential, availability, tau=1e-3)
        expected = smooth_cap_flux(potential, availability, tau=1e-3)
        self.assertTrue(torch.equal(actual, expected))
        self.assertGreater(actual.item(), 0.0)

    def test_normal_availability_preserves_in_bounds_smooth_cap(self):
        potential = self.tensor(0.25)
        availability = self.tensor(1.0)
        actual = _cap_available_flux(potential, availability, 1e-3)
        expected = smooth_cap_flux(potential, availability, tau=1e-3)
        self.assertTrue(torch.equal(actual, expected))

    def test_zero_availability_evaporation_and_one_day_balance(self):
        # This reproduces the formerly leaking nearzero-store, zero-precip day.
        qsim, ea, s1, s2, s3 = self.step(0.0, 1.0643516, 1e-6, 1e-6, 1e-6)
        self.assertEqual(ea.item(), 0.0)
        self.assertTrue(torch.isfinite(torch.stack((qsim, ea, s1, s2, s3))).all())
        residual = (s1 + s2 + s3 - 3e-6) + qsim + ea
        self.assertLessEqual(abs(residual.item()), 1e-14)

    def test_one_day_balance_with_positive_availability(self):
        P = 1.5
        qsim, ea, s1, s2, s3 = self.step(P, 0.8, 3.0, 2.0, 4.0)
        residual = (s1 + s2 + s3 - 9.0) - P + qsim + ea
        self.assertLessEqual(abs(residual.item()), 1e-12)

    def test_float64_autograd_matches_finite_difference(self):
        pet = self.tensor(1.2, requires_grad=True)
        inputs = dict(
            P=self.tensor(0.7), T=self.tensor(5.0), PET=pet,
            c=self.tensor(0.2), ndc=self.tensor(0.3), smax=self.tensor(100.0),
            emax=self.tensor(10.0), frate=self.tensor(2.0), b=self.tensor(0.5),
            dpf=self.tensor(0.1), sdrmax=self.tensor(10.0),
            S1=self.tensor(3.0), S2=self.tensor(2.0), S3=self.tensor(4.0),
        )
        objective = gsfb_step(**inputs)[1]
        gradient = torch.autograd.grad(objective, pet)[0].item()

        def evaluate(pet_value):
            copied = dict(inputs)
            copied["PET"] = self.tensor(pet_value)
            return gsfb_step(**copied)[1].item()

        h = 1e-6
        finite_difference = (evaluate(1.2 + h) - evaluate(1.2 - h)) / (2 * h)
        self.assertGreater(abs(gradient), 0.5)
        self.assertAlmostEqual(gradient, finite_difference, delta=2e-6)

    def test_zero_and_subfloor_states_are_not_refilled(self):
        for store in (0.0, 1e-12):
            with self.subTest(store=store):
                qsim, ea, s1, s2, s3 = self.step(0.0, 0.5, store, store, store)
                self.assertEqual(qsim.item(), 0.0)
                self.assertEqual(ea.item(), 0.0)
                self.assertEqual(s1.item(), store)
                self.assertEqual(s2.item(), store)
                self.assertEqual(s3.item(), store)

    def test_boundary_water_balance_gradient_is_finite(self):
        for store in (0.0, 1e-12):
            with self.subTest(store=store):
                P = self.tensor(0.5, requires_grad=True)
                PET = self.tensor(0.5, requires_grad=True)
                S1 = self.tensor(store, requires_grad=True)
                S2 = self.tensor(store, requires_grad=True)
                S3 = self.tensor(store, requires_grad=True)
                outputs = gsfb_step(
                    P=P, T=self.tensor(5.0), PET=PET,
                    c=self.tensor(0.2), ndc=self.tensor(0.3), smax=self.tensor(100.0),
                    emax=self.tensor(10.0), frate=self.tensor(2.0), b=self.tensor(0.5),
                    dpf=self.tensor(0.1), sdrmax=self.tensor(10.0), S1=S1, S2=S2, S3=S3,
                )
                qsim, ea, s1, s2, s3 = outputs
                residual = s1 + s2 + s3 + qsim + ea - (S1 + S2 + S3 + P)
                gradients = torch.autograd.grad(residual, (P, PET, S1, S2, S3))
                self.assertTrue(torch.isfinite(torch.stack(gradients)).all())
                self.assertLessEqual(abs(residual.item()), 1e-14)

    def test_dry_wet_and_high_forcing_remain_finite(self):
        for forcing_and_state in (
            (0.0, 5.0, 1e-6, 1e-6, 1e-6),
            (10.0, 2.0, 50.0, 10.0, 20.0),
            (1e4, 1e3, 100.0, 100.0, 100.0),
        ):
            outputs = self.step(*forcing_and_state)
            self.assertTrue(torch.isfinite(torch.stack(outputs)).all())
            self.assertGreaterEqual(outputs[0].item(), 0.0)
            self.assertGreaterEqual(outputs[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
