import unittest

import torch

from dmotpy.models.core.tcm import tcm_step
from dmotpy.models.flux.baseflow import baseflow_tcm


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def scalar(value, *, requires_grad=False):
    return torch.tensor(value, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)


def run_step(*, S1=10.0, S2=10.0, S3=1.0, S4=2.0, P=0.0, PET=1.0,
             phi=0.2, rc=100.0, gam=0.7, k1=0.1, fa=0.0, k2=0.2,
             mean_P=0.0, nearzero=1e-6):
    return tcm_step(
        scalar(P), scalar(0.0), scalar(PET), scalar(phi), scalar(rc),
        scalar(gam), scalar(k1), scalar(fa), scalar(k2), scalar(S1),
        scalar(S2), scalar(S3), scalar(S4), nearzero=nearzero,
        mean_P=scalar(mean_P), return_diagnostics=True,
    )


class TestTCMF2F5(unittest.TestCase):
    def test_deep_et_gate_is_active_below_post_upper_storage_threshold(self):
        # S2 is away from its recharge gate; the S2 increment is therefore deep ET.
        dry = run_step(S1=0.02, S2=10.0, PET=1.0)
        wet = run_step(S1=20.0, S2=10.0, PET=1.0)
        dry_et = dry[3] - scalar(10.0)
        wet_et = wet[3] - scalar(10.0)
        self.assertGreater(dry_et.item(), wet_et.item())
        below_gate = 1.0 - torch.sigmoid(50.0 * (dry[2] - scalar(0.01)))
        self.assertAlmostEqual(dry_et.item(), (0.7 * below_gate).item(), places=12)
        self.assertLess(wet_et.item(), 1e-8)

    def test_baseflow_direct_k2_and_caller_availability_cap(self):
        k2, storage = scalar(0.2), scalar(3.0)
        direct = baseflow_tcm(k2, scalar(0.0), storage)
        expected = k2 * storage.square() * torch.sigmoid(
            torch.tensor(50.0, dtype=DTYPE) * storage
        )
        self.assertAlmostEqual(direct.item(), expected.item(), places=12)

        # A large k2 makes the unconstrained quadratic exceed remaining S4.
        capped = run_step(S1=10.0, S2=10.0, S3=0.0, S4=2.0, PET=0.0, k2=1.0)
        q = capped[0]
        self.assertAlmostEqual(q.item(), 2.0, places=8)
        self.assertLessEqual(q.item(), 2.0)

    def test_one_day_signed_water_balance_including_external_abstraction(self):
        initial = (10.0, 4.0, 3.0, 5.0)  # S2 is a deficit, hence negative signed storage.
        P, PET, mean_P, fa = 4.0, 1.2, 2.0, 0.1
        out = run_step(S1=initial[0], S2=initial[1], S3=initial[2], S4=initial[3],
                       P=P, PET=PET, mean_P=mean_P, fa=fa, k2=0.01)
        q, ea, s1, s2, s3, s4, diagnostics = out
        initial_signed = initial[0] - initial[1] + initial[2] + initial[3]
        final_signed = s1.item() - s2.item() + s3.item() + s4.item()
        abstraction = diagnostics["external_losses"].item()
        residual = final_signed - initial_signed - (P - ea.item() - q.item() - abstraction)
        self.assertLessEqual(abs(residual), 2e-6)

    def test_float64_autograd_matches_central_finite_differences(self):
        values = [0.25, 40.0, 0.45, 0.08, 0.15, 0.012]
        names = ("phi", "rc", "gam", "k1", "fa", "k2")

        def objective(parameter_values):
            params = [scalar(v) for v in parameter_values]
            q, ea, s1, s2, s3, s4 = tcm_step(
                scalar(2.3), scalar(4.0), scalar(1.7), *params,
                scalar(12.0), scalar(3.0), scalar(5.0), scalar(8.0),
                mean_P=scalar(2.0),
            )
            return q + 0.31 * ea + 0.013 * (s1 - s2 + s3 + s4)

        grad_params = [scalar(v, requires_grad=True) for v in values]
        q, ea, s1, s2, s3, s4 = tcm_step(
            scalar(2.3), scalar(4.0), scalar(1.7), *grad_params,
            scalar(12.0), scalar(3.0), scalar(5.0), scalar(8.0),
            mean_P=scalar(2.0),
        )
        value = q + 0.31 * ea + 0.013 * (s1 - s2 + s3 + s4)
        gradients = torch.autograd.grad(value, grad_params)
        errors = []
        for i, gradient in enumerate(gradients):
            step = 1e-6 * max(1.0, abs(values[i]))
            plus, minus = values.copy(), values.copy()
            plus[i] += step
            minus[i] -= step
            finite_difference = (objective(plus).item() - objective(minus).item()) / (2 * step)
            errors.append(abs(gradient.item() - finite_difference))
        self.assertLessEqual(max(errors), 2e-5, f"{dict(zip(names, errors))}")

    def test_low_and_high_legal_parameter_and_storage_values_are_finite(self):
        cases = [
            dict(S1=1e-6, S2=1e-6, S3=1e-6, S4=1e-6,
                 phi=0.0, rc=1.0, gam=0.0, k1=0.0, fa=0.0, k2=0.0,
                 P=0.0, PET=0.0, mean_P=0.0),
            dict(S1=2000.0, S2=2000.0, S3=2000.0, S4=2000.0,
                 phi=1.0, rc=2000.0, gam=1.0, k1=1.0, fa=1.0, k2=1.0,
                 P=100.0, PET=100.0, mean_P=100.0),
        ]
        for case in cases:
            with self.subTest(case=case):
                outputs = run_step(**case)
                tensors = outputs[:6]
                self.assertTrue(all(torch.isfinite(x).all().item() for x in tensors))
                self.assertTrue(torch.isfinite(outputs[6]["external_losses"]).all().item())


if __name__ == "__main__":
    unittest.main()
