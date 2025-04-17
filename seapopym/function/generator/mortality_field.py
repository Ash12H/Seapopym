"""A temperature mask computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core import template
from seapopym.function.core.kernel import kernel_unit_factory
from seapopym.standard.attributs import mortality_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units

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
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]

    Note:
    ----
    The mortality field is computed as follow:
    - lambda = lambda_0 * exp(gamma_lambda * T)
    - B_t = B_(t-1) * exp(-dt * lambda)

    Which is equivalent to:
    - tau_m = tau_m_0 * exp(gamma_tau_m * T)
    Where tau_m is equal to 1/lambda, tau_m_0 is equal to 1/lambda_0 and gamma_tau_m is equal to -gamma_lambda.

    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    inv_lambda_max = state[ConfigurationLabels.inv_lambda_max]
    inv_lambda_rate = state[ConfigurationLabels.inv_lambda_rate]
    timestep = state[ConfigurationLabels.timestep]

    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature)

    mortality_rate_lambda = (1 / inv_lambda_max) * np.exp(
        -inv_lambda_rate * average_temperature
    )  # lambda = lambda_0 * exp(gamma_lambda * T)
    mortality_field = np.exp(-timestep * mortality_rate_lambda)  # B_t = B_(t-1) * exp(-dt * lambda)
    return xr.Dataset({ForcingLabels.mortality_field: mortality_field})


MortalityFieldTemplate = template.template_unit_factory(
    name=ForcingLabels.mortality_field,
    attributs=mortality_field_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)

MortalityFieldKernel = kernel_unit_factory(
    name="mortality_field", template=[MortalityFieldTemplate], function=mortality_field
)
