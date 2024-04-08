"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import mask_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels

if TYPE_CHECKING:
    from seapopym.standard.types import SeapopymForcing


def _mask_by_fgroup_helper(state: xr.Dataset) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    day_layers = state[ConfigurationLabels.day_layer]
    night_layers = state[ConfigurationLabels.night_layer]
    global_mask = state[PreproductionLabels.global_mask]

    masks = []
    for i in day_layers[CoordinatesLabels.functional_group]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = global_mask.cf.sel(Z=day_pos)
        night_mask = global_mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group, global_mask.cf["Y"].name, global_mask.cf["X"].name),
        coords={
            CoordinatesLabels.functional_group: day_layers[CoordinatesLabels.functional_group],
            global_mask.cf["Y"].name: global_mask.cf["Y"],
            global_mask.cf["X"].name: global_mask.cf["X"],
        },
        data=masks,
    )


# def mask_by_fgroup(state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None) -> SeapopymForcing:
#     """Wrap the mask by fgroup computation with a map_block function."""
#     class_type = Template if lazy is None else TemplateLazy
#     template_attributs = {
#         "name": PreproductionLabels.mask_by_fgroup,
#         "dims": [CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X],
#         "attributs": mask_by_fgroup_desc,
#         "chunk": chunk,
#     }
#     if lazy is not None:
#         template_attributs["model_name"] = lazy
#     template = class_type(**template_attributs)

#     return apply_map_block(function=_mask_by_fgroup_helper, state=state, template=template)


def mask_by_fgroup_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=PreproductionLabels.mask_by_fgroup,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=mask_by_fgroup_desc,
        chunks=chunk,
    )


def mask_by_fgroup_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> SeapopymForcing:
    if template is None:
        template = mask_by_fgroup_template(chunk=chunk)
    return KernelUnits(
        name=PreproductionLabels.mask_by_fgroup,
        template=template,
        function=_mask_by_fgroup_helper,
    )
