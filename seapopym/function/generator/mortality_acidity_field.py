from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units
from seapopym.standard.attributs import mortality_acidity_field_desc

if TYPE_CHECKING:
    import xarray as xr

def _mortality_acidity_field_helper(state: xr.Dataset) -> xr.DataArray:
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
    average_acidity=state[ForcingLabels.avg_acidity_by_fgroup]
    lambda_pH_max = state[ConfigurationLabels.lambda_pH_max]
    lambda_pH_rate = state[ConfigurationLabels.lambda_pH_rate]
    lambda_T_max = state[ConfigurationLabels.lambda_T_max]
    lambda_T_rate = state[ConfigurationLabels.lambda_T_rate]
    timestep = state[ConfigurationLabels.timestep]

    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature)
    average_acidity = check_units(average_acidity, StandardUnitsLabels.acidity)

    part_pH= lambda_pH_max*np.exp(lambda_pH_rate*average_acidity)
    part_T= lambda_T_max*np.exp(lambda_T_rate*average_temperature)
    return np.exp(-timestep *(part_pH+part_T) )

def mortality_acidity_field_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.mortality_field,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=mortality_acidity_field_desc,
        chunks=chunk,
    )

def mortality_acidity_field_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = mortality_acidity_field_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.mortality_field,
        template=template,
        function=_mortality_acidity_field_helper,  # new function using pH and temperature
    )
