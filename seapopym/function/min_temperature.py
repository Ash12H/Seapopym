"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import min_temperature_by_cohort_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def min_temperature_by_cohort(state: SeapopymState) -> xr.Dataset:
    """
    Define the minimal temperature of a cohort to be recruited.

    Input
    -----
    - mean_timestep [functional_group, cohort_age]
    - tr_max [functional_group]
    - tr_rate [functional_group]

    Output
    ------
    - min_temperature [functional_group, cohort_age] : a datarray with cohort_age as coordinate and
    minimum temperature as value.

    Note:
    ----
    The minimal temperature for recruitment is defined as:
    - Temperature = log(Tau_r / Tau_r_0) / Gamma_Tau_r
    Which is calculated from the equation Tau_r = Tau_r_0 * exp(Gamma_Tau_r * Temperature)
    Where Tau_r is equal to the cohorte age (delta_t -> Tau_r_0).

    """
    min_temperature = (
        np.log(state[ConfigurationLabels.mean_timestep] / state[ConfigurationLabels.tr_0])
        / state[ConfigurationLabels.gamma_tr]
    )
    return xr.Dataset({ForcingLabels.min_temperature: min_temperature})


MinTemperatureByCohortTemplate = template.template_unit_factory(
    name=ForcingLabels.min_temperature,
    attributs=min_temperature_by_cohort_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.cohort],
)

MinTemperatureByCohortKernel = kernel.kernel_unit_factory(
    name="mortality_field", template=[MinTemperatureByCohortTemplate], function=min_temperature_by_cohort
)
