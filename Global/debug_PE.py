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
import sys

# Add debugging flag
DEBUG = True

def debug_print(msg, data=None):
    if DEBUG:
        print(f"DEBUG: {msg}")
        if data is not None:
            if hasattr(data, 'shape'):
                print(f"  Shape: {data.shape}")
            if hasattr(data, 'min') and hasattr(data, 'max'):
                try:
                    print(f"  Min: {data.min().values}, Max: {data.max().values}")
                    print(f"  Mean: {data.mean().values}")
                    print(f"  Non-null count: {data.notnull().sum().values}")
                except:
                    print(f"  Could not compute stats")

name_npp = sys.argv[1] if len(sys.argv) > 1 else 'vgpm'
print(f"NPP product: {name_npp}")

# CSV file that contain the top 10 individuals
indiv_path = "top10_no_transport.csv" 
# the zarr export file name (file that will contain the modelized biomass)
export_file_name = "no_transport"

# path forcings 
path_T_vgpm_pld = "/scratch/sroyer/SEAPOPYM/Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr"
forcing_npp = {
    "vgpm": "Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr",
    "cafe": "CAFE_daily_global_1deg.nc",
    "cbpm_westberry": "CbPM_west_daily_global_1deg.nc",
    "cbpm_behrenfeld": "CbPM_behr_daily_global_1deg.nc"
}
path_npp = forcing_npp[name_npp]

def load_T_vgpm_pld(path):
    debug_print(f"Loading data from {path}")
    ds_forcings = xr.open_zarr(path)
    debug_print("Original dataset loaded", ds_forcings)
    
    ds_forcings = ds_forcings.chunk({
        "time": -1,
        "latitude": 5,
        "longitude": 5,
    })
    
    # Variables 
    ds_T = ds_forcings['temperature'].sel(depth=0)  # epipelagic layer
    ds_vgpm = ds_forcings['primary_production']
    ds_pld = ds_forcings['pelagic_layer_depth'].sel(depth=0)  # epipelagic layer
    
    debug_print("Temperature data", ds_T)
    debug_print("Primary production data", ds_vgpm)
    debug_print("PLD data", ds_pld)
    
    # mask land / ocean
    mask = ds_forcings['mask']
    debug_print("Mask data", mask)
    
    # apply mask
    with xr.set_options(keep_attrs=True):
        # temperature
        ds_T = ds_T.where(mask)
        # primary production 
        ds_vgpm = ds_vgpm.where(mask)
        # pelagic layer depth
        ds_pld = ds_pld.where(mask)
    
    debug_print("After masking - Temperature", ds_T)
    debug_print("After masking - Primary production", ds_vgpm)
    debug_print("After masking - PLD", ds_pld)
    
    return ds_T, ds_vgpm, ds_pld, mask

def format_variable(var, dimension=3): 
    debug_print(f"Formatting variable with dimension {dimension}", var)
    var.name = None
    var["time"].attrs["standard_name"] = "time"
    var["time"].attrs["axis"] = "T"
    var["latitude"].attrs["standard_name"] = "latitude"
    var["latitude"].attrs["axis"] = "Y"
    var["longitude"].attrs["standard_name"] = "longitude"
    var["longitude"].attrs["axis"] = "X"
    var = var.assign_coords({"time": var.time.dt.floor("D")})
    if dimension == 4:
        var = var.expand_dims(layer=[0])
        var["layer"].attrs["standard_name"] = "layer"
        var["layer"].attrs["axis"] = "Z"
    debug_print("After formatting", var)
    return var

def create_functional_groups_from_csv(csv_path, NPP_product):
    debug_print(f"Creating functional groups from {csv_path}")
    
    # Check if CSV exists and is readable
    try:
        df = pd.read_csv(csv_path)
        debug_print(f"CSV loaded successfully with {len(df)} rows")
        print("CSV content:")
        print(df.head())
        print("CSV columns:", df.columns.tolist())
    except Exception as e:
        debug_print(f"ERROR loading CSV: {e}")
        return None

    f_groups_list = []
    for idx, row in df.iterrows():
        debug_print(f"Processing individual {idx}")
        print(f"Parameters for individual {idx}:")
        print(f"  energy_transfert: {row['energy_transfert']}")
        print(f"  lambda_T_0: {row['lambda_T_0']}")
        print(f"  gamma_lambda_T: {row['gamma_lambda_T']}")
        print(f"  tr_0: {row['tr_0']}")
        print(f"  gamma_tr: {row['gamma_tr']}")
        
        # Check for invalid parameters
        if any(pd.isna([row['energy_transfert'], row['lambda_T_0'], 
                       row['gamma_lambda_T'], row['tr_0'], row['gamma_tr']])):
            debug_print(f"WARNING: NaN values found in row {idx}")
        
        f_group = FunctionalGroupUnit(
            name=f"indiv{idx}{NPP_product}",
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
    
    debug_print(f"Created {len(f_groups_list)} functional groups")
    return FunctionalGroupParameter(functional_group=f_groups_list)

def common_mask(primary_production, temperature, ds_pld):
    debug_print("Applying common mask")
    nb_valid = primary_production.notnull().sum(dim="time")
    debug_print("Valid points per location", nb_valid)
    
    # Le point est valide si au moins une valeur non-NaN dans le temps
    mask = nb_valid > 0
    mask = xr.DataArray(mask, coords=nb_valid.coords, dims=("latitude", "longitude"))
    debug_print("Spatial mask", mask)
    
    with xr.set_options(keep_attrs=True):
        # temperature
        temperature = temperature.where(mask)
        # primary production 
        primary_production = primary_production.where(mask)
        # pelagic layer depth
        ds_pld = ds_pld.where(mask)
    
    debug_print("After common masking - Temperature", temperature)
    debug_print("After common masking - Primary production", primary_production)
    debug_print("After common masking - PLD", ds_pld)
    
    return primary_production, temperature, ds_pld

def config_model(f_group_path, NPP_product, ds):
    debug_print("Configuring model")
    f_groups = create_functional_groups_from_csv(f_group_path, NPP_product)
    
    if f_groups is None:
        debug_print("ERROR: Failed to create functional groups")
        return None
    
    debug_print("Dataset for model configuration", ds)
    
    p_param = ForcingParameter(
        temperature=ForcingUnit(forcing=ds["temperature"]),
        primary_production=ForcingUnit(forcing=ds["primary_production"]),
    )
    
    environment_parameter = EnvironmentParameter(
        chunk=ChunkParameter(latitude=5, longitude=5)
    )
    kernel_parameter = KernelParameter(compute_initial_conditions=True)
    
    parameters = NoTransportConfiguration(
        forcing=p_param, 
        functional_group=f_groups,
        kernel=kernel_parameter,
        environment=environment_parameter
    )
    
    debug_print("Model configuration created")
    model = NoTransportModel.from_configuration(configuration=parameters)
    debug_print("Model instantiated")
    return model

def convert_biomass(biomass, pld):
    debug_print("Converting biomass units", biomass)
    pld = pld.assign_coords(time=pld.time.dt.floor("D"))
    biomass = biomass.astype("float32")
    
    debug_print("Biomass before conversion", biomass)
    debug_print("PLD for conversion", pld)
    
    # Check if PLD has valid values
    pld_valid = pld.notnull().sum()
    debug_print(f"Valid PLD values: {pld_valid.values}")
    
    # divide by the epipelagic layer depth
    biomass_m3 = biomass / pld
    biomass_m3.attrs['units'] = "kilogram / meter**3"
    
    debug_print("Biomass after conversion", biomass_m3)
    return biomass_m3

def main():
    try:
        client = get_client()
    except ValueError:
        client = Client(
            n_workers=10,
            threads_per_worker=1,
            memory_limit="20GB",
        )

    print('Loading Data')
    ds_T, ds_vgpm, ds_pld, mask = load_T_vgpm_pld(path_T_vgpm_pld)

    temperature = format_variable(ds_T, dimension=4)
    temperature.attrs["units"] = "celsius" 
    
    vgpm_ref = format_variable(ds_vgpm, dimension=3)
    vgpm_ref.attrs["units"] = "mg m-2 day-1"

    del ds_vgpm
    del ds_T
    gc.collect()

    print(f"Processing {name_npp} NPP forcing")
    if name_npp == 'vgpm':
        primary_production = vgpm_ref
        del vgpm_ref
    else:
        ds_pp = xr.open_dataarray("/scratch/sroyer/SEAPOPYM/NPP/" + path_npp)
        ds_pp = ds_pp.chunk({"time": -1, "latitude": 5, "longitude": 5}) 
        ds_pp = ds_pp.where(mask)
        primary_production = format_variable(ds_pp, dimension=3)
        del ds_pp
        primary_production.attrs["units"] = "mg m-2 day-1"
        primary_production, temperature, ds_pld = common_mask(primary_production, temperature, ds_pld)

    gc.collect() 

    dataset = xr.Dataset({"temperature": temperature, "primary_production": primary_production})
    dataset = dataset.drop_vars("depth", errors="ignore")
    
    debug_print("Final dataset for model", dataset)

    model = config_model(indiv_path, name_npp, dataset)
    
    if model is None:
        print("ERROR: Model configuration failed!")
        return

    print("Running model...")
    model.run()
    print(f"Model run with {name_npp} done")

    with xr.set_options(keep_attrs=True):
        biomass = model.state.biomass
        debug_print("Model biomass output", biomass)
        
        biomass = biomass.chunk({
            "time": -1,
            "latitude": 5,
            "longitude": 5,
        })
        
    # convert biomass from kg C m-2 to kg C m-3 
    biomass_m3 = convert_biomass(biomass, ds_pld)
    debug_print("Final biomass for saving", biomass_m3)
    
    # Additional check before saving
    total_biomass = biomass_m3.sum()
    debug_print(f"Total biomass sum: {total_biomass.values}")
    
    biomass_m3.to_zarr(
        f"/scratch/sroyer/SEAPOPYM/output/{export_file_name}_{name_npp}.zarr",
        mode='w',
    )
    print(f"{name_npp} Biomass saved")
    
    # Libérer la mémoire
    del biomass, biomass_m3, model, dataset, primary_production, temperature
    gc.collect()

    client.close()

if __name__ == "__main__":
    main()