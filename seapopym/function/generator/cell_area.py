"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.cell_area import mesh_cell_area
from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import compute_cell_area_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels
from seapopym.standard.types import SeapopymForcing


def _cell_area_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Compute the cell area from the latitude and longitude.

    Input
    ------
    - latitude [latitude]
    - longitude [longitude]
    - resolution

    Output
    ------
    - cell_area [latitude, longitude]s
    """
    resolution = (state[ConfigurationLabels.resolution_latitude], state[ConfigurationLabels.resolution_longitude])
    resolution = np.asarray(resolution)
    resolution = float(resolution) if resolution.size == 1 else tuple(resolution)
    return mesh_cell_area(state.cf[CoordinatesLabels.Y], state.cf[CoordinatesLabels.X], resolution)


# def cell_area(state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None) -> SeapopymForcing:
#     """Wrap the average temperature by functional group computation with a map_block function."""
#     class_type = Template if lazy is None else TemplateLazy
#     template_attributs = {
#         "name": PreproductionLabels.cell_area,
#         "dims": [CoordinatesLabels.Y, CoordinatesLabels.X],
#         "attributs": compute_cell_area_desc,
#         "chunk": chunk,
#     }
#     if lazy is not None:
#         template_attributs["model_name"] = lazy
#     template = class_type(**template_attributs)
#     return apply_map_block(function=_cell_area_helper, state=state, template=template)


def cell_area_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=PreproductionLabels.cell_area,
        dims=[CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=compute_cell_area_desc,
        chunks=chunk,
    )


def cell_area_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> SeapopymForcing:
    if template is None:
        template = cell_area_template(chunk=chunk)
    return KernelUnits(
        name=PreproductionLabels.cell_area,
        template=template,
        function=_cell_area_helper,
    )
