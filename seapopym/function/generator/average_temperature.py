"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core import kernel, template
from seapopym.standard.attributs import average_temperature_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState

AverageTemperatureTemplate = template.template_unit_factory(
    name=ForcingLabels.avg_temperature_by_fgroup,
    attributs=average_temperature_by_fgroup_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


@kernel.kernel_unit_registry_factory(name="average_temperature", template=[AverageTemperatureTemplate])
def average_temperature(state: SeapopymState) -> xr.Dataset:
    """
    Depend on:
    - compute_daylength
    - mask_by_fgroup.

    Input
    -----
    - mask_by_fgroup()      [time, latitude, longitude]
    - compute_daylength()   [functional_group, latitude, longitude] in day
    - day/night_layer       [functional_group]
    - temperature           [time, latitude, longitude, layer] in degC

    Output
    ------
    - avg_temperature [functional_group, time, latitude, longitude] in degC
    """
    temperature = check_units(state[ForcingLabels.temperature], StandardUnitsLabels.temperature.units)
    day_length = check_units(state[ForcingLabels.day_length], StandardUnitsLabels.time.units)
    mask_by_fgroup = state[ForcingLabels.mask_by_fgroup]
    day_layer = state[ConfigurationLabels.day_layer]
    night_layer = state[ConfigurationLabels.night_layer]

    average_temperature = []
    for fgroup in day_layer[CoordinatesLabels.functional_group]:
        day_temperature = temperature.cf.sel(Z=day_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        night_temperature = temperature.cf.sel(Z=night_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        mean_temperature = (day_length * day_temperature) + ((1 - day_length) * night_temperature)
        if "Z" in mean_temperature.cf:
            mean_temperature = mean_temperature.cf.drop_vars("Z")
        mean_temperature = mean_temperature.where(mask_by_fgroup.sel({CoordinatesLabels.functional_group: fgroup}))
        average_temperature.append(mean_temperature)

    average_temperature = xr.concat(average_temperature, dim=CoordinatesLabels.functional_group.value)
    return xr.Dataset({ForcingLabels.avg_temperature_by_fgroup: average_temperature})
