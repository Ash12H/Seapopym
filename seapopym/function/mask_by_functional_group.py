"""Funcitons used to generate a landmask from any forcing data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401  # noqa: F401
import xarray as xr

from seapopym.core import kernel, template
from seapopym.standard.attributs import mask_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymState


def mask_by_fgroup(state: SeapopymState) -> xr.Dataset:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    day_layers = state[ConfigurationLabels.day_layer]
    night_layers = state[ConfigurationLabels.night_layer]
    global_mask = state[ForcingLabels.global_mask]

    masks = []
    for i in day_layers[CoordinatesLabels.functional_group]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = global_mask.cf.sel(Z=day_pos)
        night_mask = global_mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    mask_by_fgroup = xr.DataArray(
        dims=(CoordinatesLabels.functional_group, global_mask.cf["Y"].name, global_mask.cf["X"].name),
        coords={
            CoordinatesLabels.functional_group: day_layers[CoordinatesLabels.functional_group],
            global_mask.cf["Y"].name: global_mask.cf["Y"],
            global_mask.cf["X"].name: global_mask.cf["X"],
        },
        data=masks,
    )
    return xr.Dataset({ForcingLabels.mask_by_fgroup: mask_by_fgroup})


MaskByFunctionalGroupTemplate = template.template_unit_factory(
    name=ForcingLabels.mask_by_fgroup,
    attributs=mask_by_fgroup_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X],
    dtype=bool,
)


MaskByFunctionalGroupKernel = kernel.kernel_unit_factory(
    name="mask_by_fgroup", template=[MaskByFunctionalGroupTemplate], function=mask_by_fgroup
)

MaskByFunctionalGroupKernelLight = kernel.kernel_unit_factory(
    name="mask_by_fgroup_light",
    template=[MaskByFunctionalGroupTemplate],
    function=mask_by_fgroup,
    to_remove_from_state=[ForcingLabels.global_mask],
)
