"""
This module defines the kernel parameters which are used to modify behaviour of kernel functions.
These meta-parameters are integrated in the model state and used in kernel functions.
"""

from numbers import Number

import xarray as xr
from attrs import field, frozen

from seapopym.configuration.abstract_configuration import AbstractKernelParameter
from seapopym.standard.labels import ConfigurationLabels


@frozen(kw_only=True)
class KernelParameter(AbstractKernelParameter):
    """This data class is used to store the parameters of the kernel."""

    angle_horizon_sun: Number = field(
        alias=ConfigurationLabels.angle_horizon_sun,
        default=0.0,
        metadata={"description": "The angle between the horizon and the sun in degrees."},
    )

    compute_initial_conditions: bool = field(
        alias=ConfigurationLabels.compute_initial_conditions,
        default=False,
        metadata={"description": "If True, the initial conditions are computed."},
    )

    compute_preproduction: bool = field(
        alias=ConfigurationLabels.compute_preproduction,
        default=False,
        metadata={"description": "If True, the pre-production is computed."},
    )

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing."""
        if self.compute_initial_conditions and self.compute_preproduction:
            msg = (
                "Select only one of compute_initial_conditions or compute_preproduction."
                "As compute_initial_conditions is included in compute_preproduction, "
            )
            raise ValueError(msg)

    def to_dataset(self) -> xr.Dataset:
        """Convert the kernel parameters to a dictionary."""
        return xr.Dataset(
            {
                ConfigurationLabels.angle_horizon_sun: self.angle_horizon_sun,
                ConfigurationLabels.compute_initial_conditions: self.compute_initial_conditions,
                ConfigurationLabels.compute_preproduction: self.compute_preproduction,
            }
        )
