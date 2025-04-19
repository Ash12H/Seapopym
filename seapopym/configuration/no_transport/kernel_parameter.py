"""
This module defines the kernel parameters which are used to modify behaviour of kernel functions.
These meta-parameters are integrated in the model state and used in kernel functions.
"""

from numbers import Number

from attrs import field, frozen

from seapopym.configuration.abstract_configuration import AbstractKernelParameter


@frozen(kw_only=True)
class KernelParameter(AbstractKernelParameter):
    """This data class is used to store the parameters of the kernel."""

    angle_horizon_sun: Number = field(
        default=0.0, metadata={"description": "The angle between the horizon and the sun in degrees."}
    )

    compute_initial_conditions: bool = field(
        default=True, metadata={"description": "If True, the initial conditions are computed."}
    )

    compute_preproduction: bool = field(
        default=False, metadata={"description": "If True, the pre-production is computed."}
    )
