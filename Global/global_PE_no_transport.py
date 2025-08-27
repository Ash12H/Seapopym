# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gc 
from seapopym.configuration.no_transport import (
    ForcingUnit,
    MigratoryTypeParameter,
    FunctionalTypeParameter,
    FunctionalGroupParameter,
    NoTransportConfiguration,
    ForcingParameter,
    FunctionalGroupUnit,
    ChunkParameter,
    EnvironmentParameter,
    KernelParameter
)

from seapopym.model import NoTransportModel
from seapopym.standard import coordinates
from seapopym.standard.labels import ConfigurationLabels
from seapopym.standard.units import StandardUnitsLabels

from dask.distributed import Client, get_client
#import os
#name_npp = os.environ.get('NAME_NPP')
import sys
name_npp=sys.argv[1]
print(name_npp)
# CSV file that contain the top 10 individus
indiv_path="top10_no_transport.csv" 
# the zarr export file name (file that will contain the modelized biomass)
export_file_name="no_transport"

# path forcings 
path_T_vgpm_pld="/scratch/sroyer/SEAPOPYM/Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr"
forcing_npp = {
    "vgpm": "Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr",
    "cafe": "CAFE_daily_global_1deg.nc",
    "cbpm_westberry": "CbPM_west_daily_global_1deg.nc",
    "cbpm_behrenfeld": "CbPM_behr_daily_global_1deg.nc"
}
path_npp=forcing_npp[name_npp]
def load_T_vgpm_pld(path):
    ds_forcings=xr.open_zarr(path)
    ds_forcings = ds_forcings.chunk({
        "time": -1,
        "latitude": 5,
        "longitude": 5,
    })
    # Variables 
    ds_T=ds_forcings['temperature'].sel(depth=0) # epipelagic layer
    ds_vgpm=ds_forcings['primary_production']
    ds_pld=ds_forcings['pelagic_layer_depth'].sel(depth=0) # epipelagic layer
    # mask land / ocean
    mask=ds_forcings['mask']
    # apply mask
    with xr.set_options(keep_attrs=True):
        # temperature
        ds_T = ds_T.where(mask)
        # primary production 
        ds_vgpm = ds_vgpm.where(mask)
        # pealgic layer depth
        ds_pld = ds_pld.where(mask)
    return ds_T, ds_vgpm, ds_pld, mask

def format_variable(var,dimension=3): 
    var.name = None
    var["time"].attrs["standard_name"]="time"
    var["time"].attrs["axis"]="T"
    var["latitude"].attrs["standard_name"]="latitude"
    var["latitude"].attrs["axis"]="Y"
    var["longitude"].attrs["standard_name"]="longitude"
    var["longitude"].attrs["axis"]="X"
    var=var.assign_coords({"time": var.time.dt.floor("D")})
    if dimension==4:
        var=var.expand_dims(layer=[0])
        var["layer"].attrs["standard_name"]="layer"
        var["layer"].attrs["axis"]="Z"
    return var

def create_functional_groups_from_csv(csv_path,NPP_product):
    df = pd.read_csv(csv_path)

    f_groups_list = []
    for idx, row in df.iterrows():
        f_group = FunctionalGroupUnit(
                    name=f"indiv{idx}{NPP_product}",  # nom = numéro du jeu de paramètres
                    energy_transfert=row['energy_transfert'],
                    migratory_type=MigratoryTypeParameter(
                        day_layer=0,
                        night_layer=0
                    ),
                    functional_type=FunctionalTypeParameter(
                        lambda_temperature_0=row['lambda_T_0'],
                        gamma_lambda_temperature=row['gamma_lambda_T'],
                        tr_0=row['tr_0'],
                        gamma_tr=row['gamma_tr']
                    )
                )
        f_groups_list.append(f_group)
        break
    return FunctionalGroupParameter(functional_group=f_groups_list)

def common_mask(primary_production,temperature,ds_pld):
    nb_valid = primary_production.notnull().sum(dim="time")
    # Le point est valide si au moins une valeur non-NaN dans le temps
    mask = nb_valid > 0
    mask = xr.DataArray(mask, coords=nb_valid.coords, dims=("latitude", "longitude"))
    with xr.set_options(keep_attrs=True):
        # temperature
        temperature = temperature.where(mask)
        # primary production 
        primary_production= primary_production.where(mask)
        # pealgic layer depth
        ds_pld = ds_pld.where(mask)
    return primary_production,temperature,ds_pld

def config_model(f_group_path,NPP_product,ds):
    f_groups=create_functional_groups_from_csv(f_group_path,NPP_product)
    p_param = ForcingParameter(
        temperature=ForcingUnit(forcing=ds["temperature"]),
        primary_production=ForcingUnit(forcing=ds["primary_production"]),
    )
    environement_parameter=EnvironmentParameter(
        chunk=ChunkParameter(latitude=5,longitude=5)
    )
    kernel_parameter=KernelParameter(compute_initial_conditions=True)
    parameters = NoTransportConfiguration(
        forcing=p_param, 
        functional_group=f_groups,
        kernel=kernel_parameter,
        environment=environement_parameter
    )
    model = NoTransportModel.from_configuration(configuration=parameters)
    return model
def convert_biomass(biomass,pld):
    pld=pld.assign_coords(time=pld.time.dt.floor("D"))
    biomass=biomass.astype("float32")
    print(biomass)
    print(pld)
    # divide by the epipelagic layer depth
    biomass_m3=biomass/pld
    biomass_m3.attrs['units']="kilogram / meter**3"
    return biomass_m3

def main():
    try:
        client= get_client()
    except ValueError:
        client=Client(
            n_workers=10,
            threads_per_worker=1,
            memory_limit="20GB",
        )

    print('Loading Data')
    ds_T, ds_vgpm, ds_pld, mask=load_T_vgpm_pld(path_T_vgpm_pld)

    temperature=format_variable(ds_T,dimension=4)
    temperature.attrs["units"] = "celsius" 
    
    vgpm_ref=format_variable(ds_vgpm,dimension=3)

    vgpm_ref.attrs["units"] = "mg m-2 day-1"

    del ds_vgpm
    del ds_T
    gc.collect()

    print(f"Processing {name_npp} NPP forcing")
    if name_npp=='vgpm':
        primary_production=vgpm_ref
        del vgpm_ref
    else :
        ds_pp = xr.open_dataarray("/scratch/sroyer/SEAPOPYM/NPP/"+path_npp)
        ds_pp = ds_pp.chunk({"time": -1, "latitude": 5, "longitude": 5}) 
        ds_pp=ds_pp.where(mask)
        primary_production=format_variable(ds_pp,dimension=3)
        del ds_pp
        primary_production.attrs["units"] = "mg m-2 day-1"
        primary_production,temperature,ds_pld=common_mask(primary_production,temperature,ds_pld)

    gc.collect() 

    dataset = xr.Dataset({"temperature": temperature, "primary_production": primary_production})
    dataset = dataset.drop_vars("depth", errors="ignore")

    model = config_model(indiv_path,name_npp,dataset)

    model.run()
    print(f"Model run with {name_npp} done")

    with xr.set_options(keep_attrs=True):
        biomass = model.state.biomass
        biomass = biomass.chunk({
            "time": -1,
            "latitude": 5,
            "longitude": 5,
        })
        
    # convert biomass from kg C m-2 to kg C m-3 
    biomass_m3=convert_biomass(biomass,ds_pld)
    print(biomass_m3)
    biomass_m3.to_zarr(
        f"/scratch/sroyer/SEAPOPYM/output/{export_file_name}_{name_npp}.zarr",
        mode='w',
    )
    print(f"{name_npp} Biomass saved")
    

if __name__=="__main__":
    main()
