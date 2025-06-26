"""A temperature mask computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import mask_temperature_desc
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def mask_temperature(state: SeapopymState) -> xr.Dataset:
    """
    It uses the min_temperature.

    Depend on
    ---------
    - min_temperature()
    - average_temperature()

    Input
    -----
    - min_temperature [cohort_age]
    - average_temperature [functional_group, time, latitude, longitude]

    Output
    ------
    - mask_temperature_by_cohort_by_functional_group [functional_group, time, latitude, longitude, cohort_age]

    NOTE(Jules): Warning : average temperature by functional group (because of daily vertical migration) and not by
    layer. We therefore have a function with a high cost in terms of computation and memory space.

    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    min_temperature = state[ForcingLabels.min_temperature]
    mask_temperature = average_temperature >= min_temperature
    return xr.Dataset({ForcingLabels.mask_temperature: mask_temperature})


MaskTemperatureTemplate = template.template_unit_factory(
    name=ForcingLabels.mask_temperature,
    attributs=mask_temperature_desc,
    dims=[
        CoordinatesLabels.functional_group,
        CoordinatesLabels.time,
        CoordinatesLabels.Y,
        CoordinatesLabels.X,
        CoordinatesLabels.cohort,
    ],
    dtype=bool,
)


MaskTemperatureKernel = kernel.kernel_unit_factory(
    name="mask_temperature", template=[MaskTemperatureTemplate], function=mask_temperature
)

MaskTemperatureKernelLight = kernel.kernel_unit_factory(
    name="mask_temperature_light",
    template=[MaskTemperatureTemplate],
    function=mask_temperature,
    to_remove_from_state=[ForcingLabels.min_temperature],
)
