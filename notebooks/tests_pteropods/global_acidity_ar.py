# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from seapopym.configuration.acidity import (
    FunctionalTypeParameter,
    FunctionalGroupParameter,
    AcidityConfiguration,
    ForcingParameter,
    FunctionalGroupUnit,
)
from seapopym.configuration.no_transport import (
    ForcingUnit,
    MigratoryTypeParameter,
    ChunkParameter,
    EnvironmentParameter,
    KernelParameter
)

from seapopym.model import AcidityModel
from seapopym.standard import coordinates
from seapopym.standard.labels import ConfigurationLabels
from seapopym.standard.units import StandardUnitsLabels

from dask.distributed import Client, get_client

def main():
    try:
        client= get_client()
    except ValueError:
        client=Client(
            n_workers=10,
            threads_per_worker=1,
            memory_limit="20GB",
        )

    # Load data
    # Load aragonite
    path_ar='/scratch/sroyer/SEAPOPYM/d_filled_aragonite_global_1998_2022.nc'
    da_ar=xr.open_dataarray(path_ar)

    acidity =da_ar
    acidity.attrs['units'] = "dimensionless"
    acidity["time"].attrs["standard_name"]="time"
    acidity["time"].attrs["axis"]="T"
    acidity["latitude"].attrs["standard_name"]="latitude"
    acidity["latitude"].attrs["axis"]="Y"
    acidity["longitude"].attrs["standard_name"]="longitude"
    acidity["longitude"].attrs["axis"]="X"

    # création masque terre / mer
    # # Pour chaque pixel, on regarde combien de fois il y a une valeur non-NaN dans le temps
    nb_valid = acidity.notnull().sum(dim="time")
    # Le point est valide si aragonite a au moins une valeur non-NaN dans le temps
    mask_ocean = nb_valid > 0
    mask_ocean = xr.DataArray(mask_ocean, coords=nb_valid.coords, dims=("latitude", "longitude"))
    
    acidity=acidity.expand_dims(layer=[0])
    acidity["layer"].attrs["standard_name"]="layer"
    acidity["layer"].attrs["axis"]="Z"

    # Load T, NPP and pld 
    path="/scratch/sroyer/SEAPOPYM/Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr"
    ds_forcings=xr.open_zarr(path)
    ds_forcings = ds_forcings.chunk({
        "time": -1,
        "latitude": 10,
        "longitude": 10,
    })
    # Variables 
    ds_T=ds_forcings['temperature'].sel(depth=0) # epipelagic layer
    ds_npp=ds_forcings['primary_production']
    ds_pld=ds_forcings['pelagic_layer_depth'].sel(depth=0) # epipelagic layer

    # apply mask
    with xr.set_options(keep_attrs=True):
        # temperature
        T_masked = ds_T.where(mask_ocean)
        # primary production 
        npp_masked = ds_npp.where(mask_ocean)
        # pealgic layer depth
        pld_masked = ds_pld.where(mask_ocean)

    temperature = T_masked.expand_dims(layer=[0])
    temperature.attrs["units"] = StandardUnitsLabels.temperature
    temperature.name = None
    temperature["time"].attrs["standard_name"]="time"
    temperature["time"].attrs["axis"]="T"
    temperature["latitude"].attrs["standard_name"]="latitude"
    temperature["latitude"].attrs["axis"]="Y"
    temperature["longitude"].attrs["standard_name"]="longitude"
    temperature["longitude"].attrs["axis"]="X"
    temperature["layer"].attrs["standard_name"]="layer"
    temperature["layer"].attrs["axis"]="Z"
    temperature=temperature.assign_coords({"time": temperature.time.dt.floor("D")})

    primary_production=npp_masked
    primary_production.attrs["units"] = "mg m-2 day-1"
    primary_production.name=None
    primary_production["time"].attrs["standard_name"]="time"
    primary_production["time"].attrs["axis"]="T"
    primary_production["latitude"].attrs["standard_name"]="latitude"
    primary_production["latitude"].attrs["axis"]="Y"
    primary_production["longitude"].attrs["standard_name"]="longitude"
    primary_production["longitude"].attrs["axis"]="X"
    primary_production=primary_production.assign_coords({"time": primary_production.time.dt.floor("D")})

    dataset = xr.Dataset({"temperature": temperature, "primary_production": primary_production, "acidity":acidity})
    dataset = dataset.drop_vars("depth", errors="ignore")

    # Set model parameters
    day_layer = 0
    night_layer = 0
    tr_0 = 14.500203
    gamma_tr = -0.287597
    lambda_temperature_0 = 0.004056
    gamma_lambda_temperature = 0.371879
    lambda_acidity_0 = 0.316360
    gamma_lambda_acidity = -0.697753
    energy_transfert= 0.00413

    f_groups = FunctionalGroupParameter(
        functional_group=[
            FunctionalGroupUnit(
                name=f"D{day_layer}N{night_layer}",
                energy_transfert=energy_transfert,
                migratory_type=MigratoryTypeParameter(day_layer=day_layer, night_layer=night_layer),
                functional_type=FunctionalTypeParameter(
                    lambda_temperature_0=lambda_temperature_0,
                    gamma_lambda_temperature=gamma_lambda_temperature,
                    lambda_acidity_0=lambda_acidity_0,
                    gamma_lambda_acidity=gamma_lambda_acidity,
                    tr_0=tr_0,
                    gamma_tr=gamma_tr,
                ),
            ),
        ]
    )
    p_param = ForcingParameter(
        temperature=ForcingUnit(forcing=dataset["temperature"]),
        primary_production=ForcingUnit(forcing=dataset["primary_production"]),
        acidity=ForcingUnit(forcing=dataset["acidity"]),
    )
    environement_parameter=EnvironmentParameter(
        chunk=ChunkParameter(latitude=10,longitude=10)
    )
    kernel_parameter=KernelParameter(compute_initial_conditions=True)
    parameters = AcidityConfiguration(
        forcing=p_param,
        functional_group=f_groups,
        kernel=kernel_parameter,
        environment=environement_parameter
    )

    model = AcidityModel.from_configuration(configuration=parameters)

    model.run()
    print("Model run done")
    with xr.set_options(keep_attrs=True):
        biomass = model.state.biomass.isel(functional_group=0)
        biomass = biomass.chunk({
            "time": -1,
            "latitude": 10,
            "longitude": 10,
        })
        biomass = biomass.where(mask_ocean) # mask biomass
        


    print("Biomass stats:", biomass.min().values, biomass.max().values)
    print("PLD stats:", pld_masked.min().values, pld_masked.max().values)
    print("Biomass nan count:", biomass.isnull().sum().values)
    print("PLD nan count:", pld_masked.isnull().sum().values)

    # convert biomass from kg C m-2 to kg C m-3 
    # rm hour in pld time 
    pld_masked=pld_masked.assign_coords(time=pld_masked.time.dt.floor("D"))
    # biomass in float32 (like pld)
    biomass=biomass.astype("float32")

    # divide by the epipelagic layer depth
    biomass_m3=biomass/pld_masked
    biomass_m3.attrs['units']="kilogram / meter**3"

    print(biomass_m3)
    print("Biomass kgCm-3 stats:", biomass_m3.min().values, biomass_m3.max().values)
    print("Biomass kgCm-3 nan count:", biomass_m3.isnull().sum().values)
    #biomass_m3=biomass_m3.compute()
    biomass_m3.to_zarr(
        "/scratch/sroyer/SEAPOPYM/output/global_acidity_ar_1998_2022.zarr",
        mode='w',
    )
    print("Biomass saved")
if __name__=="__main__":
    main()
