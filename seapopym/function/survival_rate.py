"""Calculate the survival rate of a population over a specified time period."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from seapopym.core import kernel, template

# WARNING check that
from seapopym.standard.attributs import survival_rate_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def survival_rate_bednarsek(state: SeapopymState) -> xr.Dataset:
    """
    # TODO(Jules): Add reference to the Bednarsek paper and documentation
    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    average_acidity = state[ForcingLabels.avg_acidity_by_fgroup]

    survival_rate_0 = state[ConfigurationLabels.survival_rate_0]
    gamma_survival_rate_acidity = state[ConfigurationLabels.gamma_survival_rate_acidity]
    gamma_survival_rate_temperature = state[ConfigurationLabels.gamma_survival_rate_temperature]

    linear_function = (
        survival_rate_0
        + gamma_survival_rate_temperature * average_temperature
        + gamma_survival_rate_acidity * average_acidity
    )
    survival_rate = xr.ufuncs.exp(linear_function) / (1 + xr.ufuncs.exp(linear_function))  # Sigmoid function

    return xr.Dataset({ForcingLabels.survival_rate: survival_rate})


SurvivalRateTemplate = template.template_unit_factory(
    name=ForcingLabels.survival_rate,
    attributs=survival_rate_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)

SurvivalRateBednarsekKernel = kernel.kernel_unit_factory(
    name="survival_rate_bednarsek", template=[SurvivalRateTemplate], function=survival_rate_bednarsek
)
