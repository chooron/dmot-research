"""Minimal API and gradient smoke example; not a scientific test."""

import torch

from dmotpy.models.hydrology_model import HydrologyModel


def main() -> None:
    model = HydrologyModel(
        config={
            "model_name": "gr4j",
            "warm_up": 1,
            "parameter_mapping": "auto",
            "backend": "none",
        },
        device=torch.device("cpu"),
    )
    # [time, basin, channels], with channels ordered P, T, PET.
    forcing = torch.tensor(
        [[[4.0, 8.0, 1.5]], [[0.0, 7.0, 1.0]], [[2.0, 9.0, 1.2]]],
        dtype=torch.float64,
    )
    raw_parameters = torch.full(
        (1, 4), 0.5, dtype=torch.float64, requires_grad=True
    )
    output = model({"x_phy": forcing}, (None, raw_parameters))
    streamflow = output["streamflow"]
    streamflow.sum().backward()
    print("streamflow shape:", tuple(streamflow.shape))
    print("parameter gradient finite:", bool(torch.isfinite(raw_parameters.grad).all()))


if __name__ == "__main__":
    main()
