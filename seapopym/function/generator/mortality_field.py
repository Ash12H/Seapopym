"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.template import Template, TemplateLazy, apply_map_block
from seapopym.standard.attributs import mortality_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels
from seapopym.standard.types import ForcingName, SeapopymForcing
from seapopym.standard.units import StandardUnitsLabels, check_units


def _mortality_field_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    average_temperature = state[PreproductionLabels.avg_temperature_by_fgroup]
    inv_lambda_max = state[ConfigurationLabels.inv_lambda_max]
    inv_lambda_rate = state[ConfigurationLabels.inv_lambda_rate]
    timestep = state[ConfigurationLabels.timestep]

    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature)
    return np.exp(-timestep * (np.exp(inv_lambda_rate * average_temperature) / inv_lambda_max))


def mortality_field(state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None) -> SeapopymForcing:
    """Wrap the average temperature by functional group computation with a map_block function."""
    class_type = Template if lazy is None else TemplateLazy
    template_attributs = {
        "name": PreproductionLabels.mortality_field,
        "dims": [CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        "attributs": mortality_field_desc,
        "chunk": chunk,
    }
    if lazy is not None:
        template_attributs["model_name"] = lazy
    template = class_type(**template_attributs)
    return apply_map_block(function=_mortality_field_helper, state=state, template=template)
