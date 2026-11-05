import unittest

import torch

from dmotpy.models.core.tcm import tcm_step


DTYPE = torch.float64
DEVICE = torch.device("cpu")
NEARZERO = 1e-6


def scalar(value):
    return torch.tensor(value, dtype=DTYPE, device=DEVICE)


def run_step(*, S1, S2, S3, S4, P=0.0, PET=0.0, k1=0.1, k2=0.01):
    return tcm_step(
        scalar(P), scalar(0.0), scalar(PET), scalar(0.2), scalar(100.0),
        scalar(0.7), scalar(k1), scalar(0.0), scalar(k2),
        scalar(S1), scalar(S2), scalar(S3), scalar(S4), nearzero=NEARZERO,
        mean_P=scalar(0.0), return_diagnostics=True,
    )


class TestTCMWaterBalanceBoundary(unittest.TestCase):
    def check_closure(self, initial, outputs, precipitation=0.0):
        q, ea, s1, s2, s3, s4, diagnostics = outputs
        initial_signed = initial[0] - initial[1] + initial[2] + initial[3]
        final_signed = (s1 - s2 + s3 + s4).item()
        losses = diagnostics["external_losses"].item()
        residual = final_signed - initial_signed - (precipitation - ea.item() - q.item() - losses)
        self.assertLessEqual(abs(residual), 1e-13, f"water-balance residual={residual:.17g}")
        for state in (s1, s2, s3, s4):
            self.assertGreaterEqual(state.item(), 0.0)

    def test_dry_and_tiny_stores_have_no_unpaired_refill(self):
        for store in (0.0, 1e-12):
            initial = (store, NEARZERO, store, store)
            outputs = run_step(S1=initial[0], S2=initial[1], S3=initial[2], S4=initial[3])
            self.check_closure(initial, outputs)
            self.assertEqual(outputs[2].item(), store)
            self.assertLessEqual(outputs[4].item(), store)
            if store == 0.0:
                self.assertEqual(outputs[5].item(), 0.0)

    def test_wet_state_behavior_and_closure_are_preserved(self):
        initial = (10.0, 4.0, 3.0, 5.0)
        outputs = run_step(S1=initial[0], S2=initial[1], S3=initial[2], S4=initial[3], P=2.0, PET=0.2)
        self.check_closure(initial, outputs, precipitation=2.0)
        self.assertGreater(outputs[2].item(), NEARZERO)
        self.assertGreater(outputs[4].item(), NEARZERO)
        self.assertGreater(outputs[5].item(), NEARZERO)


if __name__ == "__main__":
    unittest.main()
