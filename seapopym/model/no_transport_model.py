"""The LMTL model without ADRE equations."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING, Callable

import numpy as np
import xarray as xr

from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.parameter import NoTransportParameters
from seapopym.function.core import landmask
from seapopym.function.generator import pre_production
from seapopym.function.generator.biomass import compute_biomass
from seapopym.function.generator.production import compute_production
from seapopym.model.base_model import BaseModel
from seapopym.standard.labels import ConfigurationLabels, PreproductionLabels

if TYPE_CHECKING:
    from pathlib import Path

    from dask.distributed import Client, Future


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(
        self: NoTransportModel, configuration: NoTransportConfiguration | NoTransportParameters | None = None
    ) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        super().__init__()
        if isinstance(configuration, NoTransportParameters):
            self._configuration = NoTransportConfiguration(configuration)
        elif isinstance(configuration, NoTransportConfiguration):
            self._configuration = configuration
        else:
            msg = "The configuration must be an instance of NoTransportConfiguration or NoTransportParameters."
            raise TypeError(msg)
        self.state = None

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration | None:
        """The parameters structure is an attrs class."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._configuration.environment_parameters.client.client

    @classmethod
    def parse(cls: NoTransportModel, configuration_file: str | Path | IO) -> NoTransportModel:
        return NoTransportModel(NoTransportConfiguration.parse(configuration_file))

    def initialize(self: NoTransportModel) -> None:
        self.configuration.environment_parameters.client.initialize_client()

    def generate_configuration(self: NoTransportModel) -> None:
        self.state: xr.Dataset = self.configuration.model_parameters

    # TODO(Jules): Rename this method to save_state ?
    def save_configuration(self: NoTransportModel) -> None:
        """Save the configuration."""

    def pre_production(self: NoTransportModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

        def apply_if_not_already_computed(forcing_name: str, function: Callable, *args: list, **kargs: dict) -> Future:
            if forcing_name in self.state:
                return self.state[forcing_name]
            return self.client.submit(function, *args, **kargs)

        mask = apply_if_not_already_computed(
            PreproductionLabels.mask_global,
            landmask.landmask_from_nan,
            forcing=self.state[ConfigurationLabels.temperature],
        )

        mask_fgroup = apply_if_not_already_computed(
            PreproductionLabels.mask_by_fgroup,
            pre_production.mask_by_fgroup,
            day_layers=self.state[ConfigurationLabels.day_layer],
            night_layers=self.state[ConfigurationLabels.night_layer],
            mask=mask,
        )

        day_length = apply_if_not_already_computed(
            PreproductionLabels.day_length,
            pre_production.compute_daylength,
            time=self.state.cf["T"],
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
        )

        avg_tmp = apply_if_not_already_computed(
            PreproductionLabels.avg_temperature_by_fgroup,
            pre_production.average_temperature_by_fgroup,
            daylength=day_length,
            mask=mask_fgroup,
            day_layer=self.state[ConfigurationLabels.day_layer],
            night_layer=self.state[ConfigurationLabels.day_layer],
            temperature=self.state[ConfigurationLabels.temperature],
        )

        primary_production_by_fgroup = apply_if_not_already_computed(
            PreproductionLabels.primary_production_by_fgroup,
            pre_production.apply_coefficient_to_primary_production,
            primary_production=self.state[ConfigurationLabels.primary_production],
            functional_group_coefficient=self.state[ConfigurationLabels.energy_transfert],
        )

        min_temperature_by_cohort = apply_if_not_already_computed(
            PreproductionLabels.min_temperature_by_cohort,
            pre_production.min_temperature_by_cohort,
            mean_timestep=self.state[ConfigurationLabels.mean_timestep],
            tr_max=self.state[ConfigurationLabels.temperature_recruitment_max],
            tr_rate=self.state[ConfigurationLabels.temperature_recruitment_rate],
        )

        mask_temperature = apply_if_not_already_computed(
            PreproductionLabels.mask_temperature,
            pre_production.mask_temperature_by_cohort_by_functional_group,
            min_temperature_by_cohort=min_temperature_by_cohort,
            average_temperature=avg_tmp,
        )

        cell_area = apply_if_not_already_computed(
            PreproductionLabels.cell_area,
            pre_production.compute_cell_area,
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
            resolution=(
                self.state[ConfigurationLabels.resolution_latitude],
                self.state[ConfigurationLabels.resolution_longitude],
            ),
        )

        mortality_field = apply_if_not_already_computed(
            PreproductionLabels.mortality_field,
            pre_production.compute_mortality_field,
            average_temperature=avg_tmp,
            inv_lambda_max=self.state[ConfigurationLabels.inv_lambda_max],
            inv_lambda_rate=self.state[ConfigurationLabels.inv_lambda_rate],
            timestep=self.state[ConfigurationLabels.timestep],
        )

        # NOTE(Jules): Some forcing are not used in the production-process so we do not keep them in memory.
        results = self.client.gather(
            {
                # PreproductionLabels.mask_global: mask,
                PreproductionLabels.mask_by_fgroup: mask_fgroup,
                # PreproductionLabels.day_length: day_length,
                # PreproductionLabels.avg_temperature_by_fgroup: avg_tmp,
                PreproductionLabels.primary_production_by_fgroup: primary_production_by_fgroup,
                # PreproductionLabels.min_temperature_by_cohort: min_temperature_by_cohort,
                PreproductionLabels.mask_temperature: mask_temperature,
                PreproductionLabels.cell_area: cell_area,
                PreproductionLabels.mortality_field: mortality_field,
            }
        )

        self.state = xr.merge([self.state, results])

    def production(self: NoTransportModel) -> None:
        """Run the production process that is not explicitly parallel."""

        def _preproduction_converter() -> np.ndarray | None:
            """
            The production process requires a specific format, we then convert parameters to a np.ndarray that contains
            the indices of the timestamps to export. None is returned if no timestamps are required.
            """
            timestamps = self.configuration.environment_parameters.output.pre_production.timestamps
            data = self.state.cf["T"]

            if timestamps is None:
                return None
            if timestamps == "all":
                return np.arange(data.size)
            if np.all([isinstance(x, int) for x in timestamps]):
                selected_dates = data.isel(time=timestamps)
            elif np.all([isinstance(x, str) for x in timestamps]):
                selected_dates = data.sel(time=timestamps, method="nearest")
            else:
                msg = "The timestamps must be either 'all', a list of integers or a list of strings."
                raise TypeError(msg)
            return np.arange(data.size)[data.isin(selected_dates)]

        self.state = compute_production(
            data=self.state,
            chunk=self.configuration.environment_parameters.chunk.as_dict(),
            export_preproduction=_preproduction_converter(),
        )

    def post_production(self: NoTransportModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""
        biomass = xr.map_blocks(compute_biomass, self.state)
        self.state = xr.merge([self.state, biomass]).persist()

    def save_output(self: NoTransportModel) -> None:
        """Save the outputs of the model."""
        # 1. biomass
        # param_biomass = self.configuration.environment_parameters.output.biomass
        # biomass_field = self.state[PostproductionLabels.biomass]

        # 2. production

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.configuration.environment_parameters.client.close_client()