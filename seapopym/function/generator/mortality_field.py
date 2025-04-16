"""A temperature mask computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import mortality_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units

if TYPE_CHECKING:
    import xarray as xr


def _mortality_field_helper(state: xr.Dataset) -> xr.DataArray:
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
    return np.exp(-timestep * mortality_rate_lambda)  # B_t = B_(t-1) * exp(-dt * lambda)


def mortality_field_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.mortality_field,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=mortality_field_desc,
        chunks=chunk,
    )


def mortality_field_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = mortality_field_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.mortality_field,
        template=template,
        function=_mortality_field_helper,
    )
