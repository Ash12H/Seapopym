"""A temperature mask computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import mortality_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def mortality_field(state: SeapopymState) -> xr.Dataset:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - lambda_temperature_0 [functional_group]
    - gamma_lambda_temperature [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]

    Note:
    ----
    The mortality field is computed as follow:
    - lambda = lambda_temperature_0 * exp(gamma_lambda_temperature * T)
    - B_t = B_(t-1) * exp(-dt * lambda)

    Which is equivalent to:
    - tau_m = tau_m_0 * exp(gamma_tau_m * T)
    Where tau_m is equal to 1/lambda, tau_m_0 is equal to 1/lambda_temperature_0 and gamma_tau_m is equal to -gamma_lambda_temperature.

    """
    timestep = state[ConfigurationLabels.timestep]
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    lambda_temperature_0 = state[ConfigurationLabels.lambda_temperature_0]
    gamma_lambda_temperature = state[ConfigurationLabels.gamma_lambda_temperature]

    lambda_ = lambda_temperature_0 * np.exp(
        gamma_lambda_temperature * average_temperature
    )  # lambda = lambda_temperature_0 * exp(gamma_lambda_temperature * T)
    mortality_field = np.exp(-timestep * lambda_)  # B_t = B_(t-1) * exp(-dt * lambda)
    return xr.Dataset({ForcingLabels.mortality_field: mortality_field})


MortalityFieldTemplate = template.template_unit_factory(
    name=ForcingLabels.mortality_field,
    attributs=mortality_field_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)

MortalityFieldKernel = kernel.kernel_unit_factory(
    name="mortality_field", template=[MortalityFieldTemplate], function=mortality_field
)

MortalityFieldKernelLight = kernel.kernel_unit_factory(
    name="mortality_field_light",
    template=[MortalityFieldTemplate],
    function=mortality_field,
    to_remove_from_state=[ForcingLabels.avg_temperature_by_fgroup],
)
