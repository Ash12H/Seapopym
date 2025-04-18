"""
This module contains the function used to compute the recruited population in  the NotTransport model.
They are run in sequence in timeseries order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core import kernel, template
from seapopym.function.generator.compiled_functions.production_compiled_functions import (
    production,
    production_export_initial,
    production_export_preproduction,
)
from seapopym.standard.attributs import preproduction_desc, recruited_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from collections.abc import Iterable

    from seapopym.standard.types import SeapopymDims, SeapopymForcing, SeapopymState

PRODUCTION_DIMS = [CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
INITIAL_CONDITION_DIMS = [CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.cohort]
PREPRODUCTION_DIMS = [CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.cohort]


def _production_helper_init_forcing(fgroup_data: xr.Dataset) -> dict[str, np.ndarray]:
    """Initialise the forcing data used in the Numba function that compute production."""

    def standardize_forcing(forcing: xr.DataArray, nan: object = 0.0, dtype: type = np.float64) -> np.ndarray:
        """Refer to Numba documentation about array typing."""
        return np.nan_to_num(x=forcing.data, nan=nan).astype(dtype)

    if ConfigurationLabels.initial_condition_production not in fgroup_data:
        initial_condition = None
    else:
        initial_condition = standardize_forcing(fgroup_data[ConfigurationLabels.initial_condition_production])
    return {  # NOTE(Jules): the keys correspond to the parameters of the numba functions
        "primary_production": standardize_forcing(fgroup_data[ForcingLabels.primary_production_by_fgroup]),
        "mask_temperature": standardize_forcing(fgroup_data[ForcingLabels.mask_temperature]),
        "timestep_number": standardize_forcing(fgroup_data[ConfigurationLabels.timesteps_number], False, bool),
        "initial_production": initial_condition,
    }


def _production_helper_format_output(
    fgroup_data: SeapopymState, dims: Iterable[SeapopymDims], data: np.ndarray
) -> SeapopymForcing:
    """Convert the output of the Numba function to a DataArray."""
    coords = {fgroup_data.cf[dim_name].name: fgroup_data.cf[dim_name] for dim_name in dims}
    formated_data = xr.DataArray(coords=coords, dims=coords.keys())
    formated_data = CoordinatesLabels.order_data(formated_data)
    formated_data.data = data
    return formated_data


# TODO(Jules): Add a version of this function that doesn't keep preproduction data
def production(
    data: SeapopymState,
    *,
    export_preproduction: bool = False,
    export_initial_production: bool = True,
) -> xr.Dataset:
    """Compute the production using a numba jit function."""
    data = data.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    results_recruited = []
    if export_preproduction or export_initial_production:
        results_extra = []

    for fgroup in data[CoordinatesLabels.functional_group]:
        fgroup_data = data.sel({CoordinatesLabels.functional_group: fgroup}).dropna(CoordinatesLabels.cohort)
        param = _production_helper_init_forcing(fgroup_data)

        if export_preproduction:
            output_recruited, output_extra = production_export_preproduction(**param)
            results_extra.append(_production_helper_format_output(fgroup_data, PREPRODUCTION_DIMS, output_extra))

        # NOTE(Jules):  Implicite -> if both are True then export_preproduction  PREPRODUCTION_DIMS is prioritized
        #               because init is included
        elif export_initial_production:
            output_recruited, output_extra = production_export_initial(**param)
            results_extra.append(_production_helper_format_output(fgroup_data, INITIAL_CONDITION_DIMS, output_extra))

        else:
            output_recruited = production(**param)

        results_recruited.append(_production_helper_format_output(fgroup_data, PRODUCTION_DIMS, output_recruited))
    results = {ForcingLabels.recruited: xr.concat(results_recruited, dim=data[CoordinatesLabels.functional_group])}
    if export_preproduction or export_initial_production:
        results[ForcingLabels.preproduction] = xr.concat(results_extra, dim=data[CoordinatesLabels.functional_group])
    return xr.Dataset(results)


RecruitedTemplate = template.template_unit_factory(
    name=ForcingLabels.recruited,
    attributs=recruited_desc,
    dims=[CoordinatesLabels.functional_group, *PRODUCTION_DIMS],
)
InitialProductionTemplate = template.template_unit_factory(
    name=ForcingLabels.preproduction,
    attributs=preproduction_desc,
    dims=[CoordinatesLabels.functional_group, *INITIAL_CONDITION_DIMS],
)


ProductionKernel = kernel.kernel_unit_factory(
    name="production", template=[RecruitedTemplate, InitialProductionTemplate], function=production
)
