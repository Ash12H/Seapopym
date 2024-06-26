from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from seapopym.configuration.no_transport.parameter import (
    ForcingParameters,
    FunctionalGroups,
    NoTransportParameters,
)
from seapopym.configuration.parameters import parameter_functional_group
from seapopym.configuration.parameters.parameter_environment import EnvironmentParameter
from seapopym.configuration.parameters.parameter_forcing import ForcingUnit
from seapopym.exception.parameter_exception import DifferentForcingTimestepError
from seapopym.standard import coordinates

time_1 = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))
time_2 = coordinates.new_time(xr.cftime_range(start="2020", freq="2D", periods=2))
latitude_1 = coordinates.new_latitude(np.array([0, 1, 2]))
longitude_1 = coordinates.new_longitude(np.array([0, 1, 2]))
latitude_2 = coordinates.new_latitude(np.array([0, 2, 4]))
longitude_2 = coordinates.new_longitude(np.array([0, 2, 4]))
layer = coordinates.new_layer()


@pytest.fixture()
def forcing_time_1_space_1_temperature():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_1, "longitude": longitude_1},
            data=np.full((time_1.size, latitude_1.size, longitude_1.size), 0, dtype=float),
            attrs={"units": "degC"},
            name="temperature",
        )
    )


@pytest.fixture()
def forcing_time_1_space_1_primary_production():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_1, "longitude": longitude_1},
            data=np.full((time_1.size, latitude_1.size, longitude_1.size), 0, dtype=float),
            attrs={"units": "kg / m^2 / day"},
            name="primary_production",
        )
    )


# @pytest.fixture()
# def forcing_time_2_space_1_temperature():
#     return ForcingUnit(
#         forcing=xr.DataArray(
#             dims=("time", "latitude", "longitude"),
#             coords={"time": time_2, "latitude": latitude_1, "longitude": longitude_1},
#             data=np.full((time_2.size, latitude_1.size, longitude_1.size), 0, dtype=float),
#             attrs={"units": "degC"},
#             name="temperature",
#         )
#     )


@pytest.fixture()
def forcing_time_2_space_1_primary_production():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_2, "latitude": latitude_1, "longitude": longitude_1},
            data=np.full((time_2.size, latitude_1.size, longitude_1.size), 0, dtype=float),
            attrs={"units": "kg / m^2 / day"},
            name="primary_production",
        )
    )


@pytest.fixture()
def forcing_time_1_space_2_temperature():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_2, "longitude": longitude_2},
            data=np.full((time_1.size, latitude_2.size, longitude_2.size), 0, dtype=float),
            attrs={"units": "degC"},
            name="temperature",
        )
    )


@pytest.fixture()
def forcing_time_1_space_2_primary_production():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_2, "longitude": longitude_2},
            data=np.full((time_1.size, latitude_2.size, longitude_2.size), 0, dtype=float),
            attrs={"units": "kg / m^2 / day"},
            name="primary_production",
        )
    )


class TestForcingParameters:
    def test_forcing_parameters_initialization(
        self: TestForcingParameters,
        forcing_time_1_space_1_temperature: ForcingUnit,
        forcing_time_1_space_1_primary_production: ForcingUnit,
        forcing_time_1_space_2_temperature: ForcingUnit,
        forcing_time_1_space_2_primary_production: ForcingUnit,
    ) -> None:
        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_1_temperature,
            primary_production=forcing_time_1_space_1_primary_production,
        )
        assert forcing_param.timestep == 1
        assert forcing_param.resolution == (1, 1)

        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_2_temperature,
            primary_production=forcing_time_1_space_2_primary_production,
        )
        assert forcing_param.timestep == 1
        assert forcing_param.resolution == (2, 2)

    def test_forcing_parameters_initialization_with_different_space(
        self, forcing_time_1_space_1_temperature, forcing_time_1_space_2_primary_production
    ):
        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_1_temperature,
            primary_production=forcing_time_1_space_2_primary_production,
        )
        assert forcing_param.resolution == (1, 1)

    def test_forcing_parameters_initialization_with_different_time(
        self, forcing_time_1_space_1_temperature, forcing_time_2_space_1_primary_production
    ):
        with pytest.raises(DifferentForcingTimestepError):
            ForcingParameters(
                temperature=forcing_time_1_space_1_temperature,
                primary_production=forcing_time_2_space_1_primary_production,
            )


class TestNoTransportParameters:
    def test_no_transport_parameters_initialization(
        self, forcing_time_1_space_1_temperature, forcing_time_1_space_1_primary_production
    ):
        f_param = ForcingParameters(
            temperature=forcing_time_1_space_1_temperature, primary_production=forcing_time_1_space_1_primary_production
        )
        g_param = FunctionalGroups(
            functional_groups=[
                parameter_functional_group.FunctionalGroupUnit(
                    name="phytoplankton",
                    energy_transfert=0.5,
                    functional_type=parameter_functional_group.FunctionalGroupUnitRelationParameters(
                        cohorts_timesteps=[1, 2, 3, 3, 1],
                        inv_lambda_max=10,
                        inv_lambda_rate=0.5,
                        temperature_recruitment_max=10,
                        temperature_recruitment_rate=-0.5,
                    ),
                    migratory_type=parameter_functional_group.FunctionalGroupUnitMigratoryParameters(
                        day_layer=1, night_layer=1
                    ),
                )
            ],
        )
        e_param = EnvironmentParameter()
        NoTransportParameters(
            forcing_parameters=f_param, functional_groups_parameters=g_param, environment_parameters=e_param
        )
