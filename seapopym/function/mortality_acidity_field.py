from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import mortality_acidity_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def mortality_acidity_field(state: SeapopymState) -> xr.Dataset:
    """
    Use the relation between temperature, pH and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature()
    - acidity

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - acidity [time, latitude, longitude] (pH)
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    average_acidity = state[ForcingLabels.avg_acidity_by_fgroup]
    lambda_ph_max = state[ConfigurationLabels.lambda_acidity_0]
    lambda_ph_rate = state[ConfigurationLabels.gamma_lambda_acidity]
    lambda_t_max = state[ConfigurationLabels.lambda_temperature_0]
    lambda_t_rate = state[ConfigurationLabels.gamma_lambda_temperature]
    timestep = state[ConfigurationLabels.timestep]

    part_ph = lambda_ph_max * np.exp(lambda_ph_rate * average_acidity)
    part_t = lambda_t_max * np.exp(lambda_t_rate * average_temperature)
    return xr.Dataset({ForcingLabels.mortality_field: np.exp(-timestep * (part_ph + part_t))})


# def mortality_acidity_field_template(chunk: dict | None = None) -> ForcingTemplate:
#     return ForcingTemplate(
#         name=ForcingLabels.mortality_field,
#         dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
#         attrs=mortality_acidity_field_desc,
#         chunks=chunk,
#     )


# def mortality_acidity_field_kernel(
#     *, chunk: dict | None = None, template: ForcingTemplate | None = None
# ) -> KernelUnits:
#     if template is None:
#         template = mortality_acidity_field_template(chunk=chunk)
#     return KernelUnits(
#         name=ForcingLabels.mortality_field,
#         template=template,
#         function=mortality_acidity_field,  # new function using pH and temperature
#     )


MortalityTemplate = template.template_unit_factory(
    name=ForcingLabels.mortality_field,
    attributs=mortality_acidity_field_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


MortalityTemperatureAcidityKernel = kernel.kernel_unit_factory(
    name="mortality_temperature_acidity", template=[MortalityTemplate], function=mortality_acidity_field
)
