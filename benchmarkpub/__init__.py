"""Publication-facing benchmark extensions."""

from .specialization import (
    BasinSpecialization,
    SpecializationConfig,
    effective_sample_size,
    loss_from_per_basin_kge,
    weighted_basin_loss,
)

from .basin_split import (
    BasinRoles,
    TrainFittedAttributeNormalizer,
    basin_id_hash,
    resolve_basin_roles,
)

__all__ = [
    "BasinSpecialization",
    "SpecializationConfig",
    "effective_sample_size",
    "loss_from_per_basin_kge",
    "weighted_basin_loss",
    "BasinRoles",
    "TrainFittedAttributeNormalizer",
    "basin_id_hash",
    "resolve_basin_roles",
]
