"""
This module contains the function used to compute the recruited population in  the NotTransport model.
They are run in sequence in timeseries order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.core import kernel, template
from seapopym.function.compiled_functions import production_compiled_functions
from seapopym.standard.attributs import preproduction_desc, recruited_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

if TYPE_CHECKING:
    from collections.abc import Iterable

    from seapopym.standard.types import SeapopymDims, SeapopymForcing, SeapopymState

PRODUCTION_DIMS = [CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
INITIAL_CONDITION_DIMS = [
    CoordinatesLabels.Y,
    CoordinatesLabels.X,
    CoordinatesLabels.cohort,
]
PREPRODUCTION_DIMS = [
    CoordinatesLabels.time,
    CoordinatesLabels.Y,
    CoordinatesLabels.X,
    CoordinatesLabels.cohort,
]


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
        "timestep_number": standardize_forcing(
            fgroup_data[ConfigurationLabels.timesteps_number], nan=False, dtype=bool
        ),
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


def production(state: SeapopymState) -> xr.Dataset:
    """Compute the production using a numba jit function."""
    state = state.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    results_recruited = []

    for fgroup in state[CoordinatesLabels.functional_group]:
        fgroup_data = state.sel({CoordinatesLabels.functional_group: fgroup})
        param = _production_helper_init_forcing(fgroup_data)
        output_recruited = production_compiled_functions.production(**param)
        results_recruited.append(_production_helper_format_output(fgroup_data, PRODUCTION_DIMS, output_recruited))

    results = {ForcingLabels.recruited: xr.concat(results_recruited, dim=state[CoordinatesLabels.functional_group])}
    return xr.Dataset(results)


def production_initial_condition(state: SeapopymState) -> xr.Dataset:
    """Compute the production using a numba jit function. Export the initial conditions."""
    state = state.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    results_recruited = []
    results_extra = []

    for fgroup in state[CoordinatesLabels.functional_group]:
        fgroup_data = state.sel({CoordinatesLabels.functional_group: fgroup})
        param = _production_helper_init_forcing(fgroup_data)
        output_recruited, output_extra = production_compiled_functions.production_export_initial(**param)
        results_recruited.append(_production_helper_format_output(fgroup_data, PRODUCTION_DIMS, output_recruited))
        results_extra.append(_production_helper_format_output(fgroup_data, INITIAL_CONDITION_DIMS, output_extra))

    return xr.Dataset(
        {
            ForcingLabels.recruited: xr.concat(results_recruited, dim=state[CoordinatesLabels.functional_group]),
            ForcingLabels.preproduction: xr.concat(results_extra, dim=state[CoordinatesLabels.functional_group]),
        }
    )


def production_unrecruited(state: SeapopymState) -> xr.Dataset:
    """Compute the production using a numba jit function. Export the unrecruited production (pre-production)."""
    state = state.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    results_recruited = []
    results_extra = []

    for fgroup in state[CoordinatesLabels.functional_group]:
        fgroup_data = state.sel({CoordinatesLabels.functional_group: fgroup})
        param = _production_helper_init_forcing(fgroup_data)
        output_recruited, output_extra = production_compiled_functions.production_export_preproduction(**param)
        results_extra.append(_production_helper_format_output(fgroup_data, PREPRODUCTION_DIMS, output_extra))
        results_recruited.append(_production_helper_format_output(fgroup_data, PRODUCTION_DIMS, output_recruited))

    return xr.Dataset(
        {
            ForcingLabels.recruited: xr.concat(results_recruited, dim=state[CoordinatesLabels.functional_group]),
            ForcingLabels.preproduction: xr.concat(results_extra, dim=state[CoordinatesLabels.functional_group]),
        }
    )


RecruitedTemplate = template.template_unit_factory(
    name=ForcingLabels.recruited, attributs=recruited_desc, dims=[CoordinatesLabels.functional_group, *PRODUCTION_DIMS]
)
InitialProductionTemplate = template.template_unit_factory(
    name=ForcingLabels.preproduction,
    attributs=preproduction_desc,
    dims=[CoordinatesLabels.functional_group, *INITIAL_CONDITION_DIMS],
)
PreproductionTemplate = template.template_unit_factory(
    name=ForcingLabels.preproduction,
    attributs=preproduction_desc,
    dims=[CoordinatesLabels.functional_group, *PREPRODUCTION_DIMS],
)


ProductionKernel = kernel.kernel_unit_factory(name="production", template=[RecruitedTemplate], function=production)
ProductionInitialConditionKernel = kernel.kernel_unit_factory(
    name="production_initial_condition",
    template=[RecruitedTemplate, InitialProductionTemplate],
    function=production_initial_condition,
)
ProductionUnrecruitedKernel = kernel.kernel_unit_factory(
    name="production_unrecruited", template=[RecruitedTemplate, PreproductionTemplate], function=production_unrecruited
)
ProductionKernelLight = kernel.kernel_unit_factory(
    name="production_light",
    template=[RecruitedTemplate],
    function=production,
    to_remove_from_state=[ForcingLabels.primary_production_by_fgroup, ForcingLabels.mask_temperature],
)

ProductionInitialConditionKernelLight = kernel.kernel_unit_factory(
    name="production_initial_condition_light",
    template=[RecruitedTemplate, InitialProductionTemplate],
    function=production_initial_condition,
    to_remove_from_state=[ForcingLabels.primary_production_by_fgroup, ForcingLabels.mask_temperature],
)

ProductionUnrecruitedKernelLight = kernel.kernel_unit_factory(
    name="production_unrecruited_light",
    template=[RecruitedTemplate, PreproductionTemplate],
    function=production_unrecruited,
    to_remove_from_state=[ForcingLabels.primary_production_by_fgroup, ForcingLabels.mask_temperature],
)
