"""An average acidity by fgroup computation wrapper. Use xarray.map_block."""

from __future__ import annotations

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.core import kernel, template

# WARNING check that
from seapopym.standard.attributs import average_acidity_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.types import SeapopymState


def average_acidity(state: SeapopymState) -> xr.Dataset:
    """
    In the open ocean, pH does not exhibit significant temporal or spatial variability.
    However, in coastal shelf and estuarine regions, pH can fluctuate both diurnally
    and spatially due to biological activity and freshwater input.
    Additionally, pH varies with depth.
    This function is designed to capture such potential variability (e.g. could be useful
    in a future version of the model that includes vertical diurnal migration of pteropods.).

    Depend on:
    - compute_daylength
    - mask_by_fgroup.

    Input
    -----
    - mask_by_fgroup()      [time, latitude, longitude]
    - compute_daylength()   [functional_group, latitude, longitude] in day
    - day/night_layer       [functional_group]
    - acidity           [time, latitude, longitude, layer] dimensionless (pH)

    Output
    ------
    - average_acidity [functional_group, time, latitude, longitude] dimensionless (pH)
    """
    acidity = state[ForcingLabels.acidity]
    day_length = state[ForcingLabels.day_length]
    mask_by_fgroup = state[ForcingLabels.mask_by_fgroup]
    day_layer = state[ConfigurationLabels.day_layer]
    night_layer = state[ConfigurationLabels.night_layer]

    average_acidity = []
    for fgroup in day_layer[CoordinatesLabels.functional_group]:
        day_acidity = acidity.cf.sel(Z=day_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        night_acidity = acidity.cf.sel(Z=night_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        mean_acidity = (day_length * day_acidity) + ((1 - day_length) * night_acidity)
        if "Z" in mean_acidity.cf:
            mean_acidity = mean_acidity.cf.drop_vars("Z")
        mean_acidity = mean_acidity.where(mask_by_fgroup.sel({CoordinatesLabels.functional_group: fgroup}))
        average_acidity.append(mean_acidity)

    return xr.Dataset(
        {ForcingLabels.avg_acidity_by_fgroup: xr.concat(average_acidity, dim=CoordinatesLabels.functional_group.value)}
    )


AverageAcidityTemplate = template.template_unit_factory(
    name=ForcingLabels.avg_acidity_by_fgroup,
    attributs=average_acidity_by_fgroup_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
)


AverageAcidityKernel = kernel.kernel_unit_factory(
    name="average_acidity", template=[AverageAcidityTemplate], function=average_acidity
)
