"""Mortality field calculations for ocean acidification effects."""

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


MortalityTemplate = template.template_unit_factory(
    name=ForcingLabels.mortality_field,
    attributs=mortality_acidity_field_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


MortalityTemperatureAcidityKernel = kernel.kernel_unit_factory(
    name="mortality_temperature_acidity", template=[MortalityTemplate], function=mortality_acidity_field
)


def mortality_acidity_bed_field(state: SeapopymState) -> xr.Dataset:
    """
    Generate mortality field using Bednarsek et al. (2022) equation.

    Uses the linear relationship between temperature, pH and mortality from
    Bednarsek equation: mortality = lambda_0_bed + gamma_lambda_temperature_bed * T + gamma_lambda_acidity_bed * pH
    The result is converted to a daily mortality rate.

    Parameters
    ----------
    state : SeapopymState
        The model state containing forcing and configuration data.

    Returns
    -------
    xr.Dataset
        Dataset containing the mortality_field variable with Bednarsek mortality rates.

    Notes
    -----
    - Original Bednarsek equation gives weekly mortality percentage
    - Converted to daily rate by dividing by (100 * 7)
    - Negative values are clipped to 0

    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    average_acidity = state[ForcingLabels.avg_acidity_by_fgroup]
    gamma_lambda_acidity = state[ConfigurationLabels.gamma_lambda_acidity]
    lambda_0 = state[ConfigurationLabels.lambda_0]
    gamma_lambda_temperature = state[ConfigurationLabels.gamma_lambda_temperature]
    timestep = state[ConfigurationLabels.timestep]

    bednarsek = lambda_0 + gamma_lambda_temperature * average_temperature + gamma_lambda_acidity * average_acidity
    daily_rate = bednarsek / (100 * 7)
    with xr.set_options(keep_attrs=True):
        daily_rate = xr.where(daily_rate >= 0, daily_rate, 0)

    return xr.Dataset({ForcingLabels.mortality_field: np.exp(-timestep * (daily_rate))})


MortalityTemperatureAcidityBedKernel = kernel.kernel_unit_factory(
    name="mortality_temperature_acidity_bed", template=[MortalityTemplate], function=mortality_acidity_bed_field
)
