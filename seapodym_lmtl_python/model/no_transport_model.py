"""The LMTL model without ADRE equations."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import IO, Callable

import xarray as xr
from dask.distributed import Client, Future

from seapodym_lmtl_python.configuration.no_transport import client as no_transport_client
from seapodym_lmtl_python.configuration.no_transport.configuration import NoTransportConfiguration, NoTransportLabels
from seapodym_lmtl_python.configuration.no_transport.parameters import NoTransportParameters
from seapodym_lmtl_python.model.base_model import BaseModel
from seapodym_lmtl_python.pre_production import pre_production
from seapodym_lmtl_python.pre_production.core import landmask


class ForcingLabels(StrEnum):
    """
    A single place to store all labels as :
    - used in the NoTransportModel class
    - declared in no_transport_parameters
    module.
    """

    # Forcing
    mask_global = "mask"
    mask_by_fgroup = "mask_fgroup"
    day_length = "day_length"
    avg_temperature_by_fgroup = "average_temperature_by_fgroup"
    primary_production_by_fgroup = "primary_production_by_fgroup"
    min_temperature_by_cohort = "min_temperature_by_cohort"
    # Parameters
    temperature = "temperature"
    primary_production = "primary_production"
    energy_transfert = "energy_transfert"


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(
        self: NoTransportModel,
        configuration: NoTransportConfiguration | NoTransportParameters | None = None,
        client: Client | None = None,
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
        self._client = client

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration | None:
        """The parameters structure is an attrs class."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._client

    @client.setter
    def client(self: NoTransportModel, client: Client) -> None:
        """The dask Client setter."""
        if self._client is not None:
            warning_message = (
                f"The model has already a client running at '{self._client.dashboard_link}'."
                f"\nWe are then closing it and starting the new one at '{client.dashboard_link}'"
            )
            logging.warning(warning_message)
            self._client.close()
            self._client = client
        self._client = client

    @classmethod
    def parse(cls: NoTransportModel, configuration_file: str | Path | IO) -> NoTransportModel:
        return NoTransportModel(NoTransportConfiguration.parse(configuration_file))

    def initialize(self: NoTransportModel) -> None:
        self.client = no_transport_client.init_client_locally(self.configuration)

    def generate_configuration(self: NoTransportModel) -> None:
        self.state: xr.Dataset = self.configuration.as_dataset()

    def save_configuration(self: NoTransportModel) -> None:
        """Save the configuration."""

    def pre_production(self: NoTransportModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

        def apply_if_not_already_computed(forcing_name: str, function: Callable, *args: list, **kargs: dict) -> Future:
            if forcing_name in self.state:
                return self.state[forcing_name]
            return self.client.submit(function, *args, **kargs)

        # 1. Global mask
        mask = apply_if_not_already_computed(
            ForcingLabels.mask_global,
            landmask.landmask_from_nan,
            forcing=self.state[ForcingLabels.temperature],
        )

        # 2. Mask by functional group
        mask_fgroup = apply_if_not_already_computed(
            ForcingLabels.mask_by_fgroup,
            pre_production.mask_by_fgroup,
            day_layers=self.state[NoTransportLabels.day_layer],
            night_layers=self.state[NoTransportLabels.night_layer],
            mask=mask,
        )

        # 3. Mask by functional group
        day_length = apply_if_not_already_computed(
            ForcingLabels.day_length,
            pre_production.compute_daylength,
            time=self.state.cf["T"],
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
        )

        # 4. Average temperature by functional group
        avg_tmp = apply_if_not_already_computed(
            ForcingLabels.avg_temperature_by_fgroup,
            pre_production.average_temperature_by_fgroup,
            daylength=day_length,
            mask=mask_fgroup,
            day_layers=self.state[NoTransportLabels.day_layer],
            night_layers=self.state[NoTransportLabels.day_layer],
            temperature=self.state[ForcingLabels.temperature],
        )

        primary_production_by_fgroup = apply_if_not_already_computed(
            ForcingLabels.primary_production_by_fgroup,
            pre_production.apply_coefficient_to_primary_production,
            primary_production=self.state[ForcingLabels.primary_production],
            functional_group_coefficient=self.state[NoTransportLabels.energy_transfert],
        )

        # min_temperature_by_cohort = apply_if_not_already_computed(
        #     ForcingLabels.min_temperature_by_cohort,
        #     pre_production.min_temperature_by_cohort,
        #     cohort_coordinates=self.state[],
        #     tr_max=self.state[],
        #     tr_rate=self.state[],
        # )

    def production(self: NoTransportModel) -> None:
        """Run the production process that is not explicitly parallel."""

    def post_production(self: NoTransportModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""

    def save_output(self: NoTransportModel) -> None:
        """Save the outputs of the model."""

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        no_transport_client.close_client_locally(self.client)
