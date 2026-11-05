import unittest

import torch

from dmotpy.models.core.smar import _smar_deep_evap_below_gate, smar_step
from dmotpy.models.flux.smooth import soft_gate_storage_below


DTYPE = torch.float64
NEARZERO = 1e-6


def tensor(value, *, requires_grad=False):
    return torch.tensor([[value]], dtype=DTYPE, requires_grad=requires_grad)


def run_step(stores, *, precipitation=0.0, pet=4.0):
    return smar_step(
        tensor(precipitation), tensor(10.0), tensor(pet),
        tensor(0.2), tensor(20.0), tensor(100.0), tensor(0.8),
        tensor(0.2), tensor(0.05), tensor(2.0), tensor(10.0),
        *(tensor(value) for value in stores),
        nearzero=NEARZERO,
    )


class SmarF1GateTests(unittest.TestCase):
    def test_below_threshold_gate_activates_deep_et(self):
        threshold = tensor(0.1)
        below, above = tensor(0.001), tensor(0.2)
        self.assertGreater(soft_gate_storage_below(below, threshold).item(), 0.99)
        self.assertLess(soft_gate_storage_below(above, threshold).item(), 0.01)
        deep_flux = _smar_deep_evap_below_gate(
            tensor(0.8), tensor(1.0), tensor(4.0), tensor(5.0), below,
            threshold, NEARZERO,
        )
        self.assertGreater(deep_flux.item(), 3.17)

    def test_daily_water_balance_sequence(self):
        stores = [12.0, 10.0, 8.0, 6.0, 4.0, 20.0]
        max_residual = 0.0
        for precipitation, pet in [(0.0, 4.0), (2.0, 3.0), (0.5, 1.0), (0.0, 2.0)]:
            result = run_step(stores, precipitation=precipitation, pet=pet)
            qsim, ea, *next_stores = result
            storage_change = sum(float(new.item() - old) for new, old in zip(next_stores, stores))
            residual = precipitation - float(ea.item()) - float(qsim.item()) - storage_change
            max_residual = max(max_residual, abs(residual))
            stores = [float(state.item()) for state in next_stores]
        self.assertLessEqual(max_residual, 2e-12)

    def test_float64_autograd_matches_finite_difference(self):
        storage = tensor(0.08, requires_grad=True)
        def evaluate(value):
            return _smar_deep_evap_below_gate(
                tensor(0.8), tensor(2.0), tensor(4.0), tensor(5.0), value,
                tensor(0.1), NEARZERO,
            ).sum()
        analytical = torch.autograd.grad(evaluate(storage), storage)[0].item()
        step = 1e-6
        finite_difference = (evaluate(tensor(0.08 + step)).item() - evaluate(tensor(0.08 - step)).item()) / (2 * step)
        self.assertAlmostEqual(analytical, finite_difference, delta=2e-7)

    def test_full_step_outputs_and_gradients_finite_across_storage_regimes(self):
        for preceding in (0.001, 0.099, 0.101, 0.5):
            stores = [preceding, 5.0, 4.0, 3.0, 2.0, 10.0]
            differentiable = [tensor(value, requires_grad=(i == 0)) for i, value in enumerate(stores)]
            result = smar_step(
                tensor(0.0), tensor(10.0), tensor(4.0), tensor(0.2), tensor(20.0),
                tensor(100.0), tensor(0.8), tensor(0.2), tensor(0.05), tensor(2.0),
                tensor(10.0), *differentiable, nearzero=NEARZERO,
            )
            self.assertTrue(all(torch.isfinite(value).all() for value in result))
            gradient = torch.autograd.grad(sum(value.sum() for value in result), differentiable[0])[0]
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
