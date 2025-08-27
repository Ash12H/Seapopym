import xarray as xr
import numpy as np
import pyseaflux
import gc

directory = "/scratch/sroyer/SEAPOPYM/output/"

def write_filenames(model):
    npp_products = ['vgpm', 'cafe', 'cbpm_westberry', 'cbpm_behrenfeld']
    return [f"{model}_{npp}.zarr" for npp in npp_products]

file_names = write_filenames('no_transport') \
           + write_filenames('no_transport_aragonite') \
           + write_filenames('acidity_aragonite')

# Calcul de l’aire par maille (lat × lon)
lon = np.arange(-180, 180)
lat = np.arange(-80, 81)
area = pyseaflux.area.area_grid(lat, lon, return_dataarray=True)  # DataArray(lat, lon)
area = area.rename({"lon": "longitude", "lat": "latitude"})
area = area.chunk({"latitude": 5,"longitude": 5})
# Ouverture couche épipélagique
ds_pld = xr.open_zarr("/scratch/sroyer/SEAPOPYM/Sophie_GLOBAL_MULTIYEAR_BGC_001_033.zarr")
ds_pld=ds_pld.chunk({ "time": 30, "latitude": 5,"longitude": 5})
ds_pld = ds_pld.sel(
    time=slice('1999-01-01','2022-12-31'),
    latitude=slice(-80,80),
    depth=0  # épipélagique
).drop_vars("depth",errors='ignore')

# Masquage terre / mer
da_pld = ds_pld['pelagic_layer_depth']
da_pld = da_pld.where(ds_pld['mask'])

# Calcul du volume (area [m2] * depth [m])
volume = area * da_pld  # (time, lat, lon)

member_data = []
member_counter = 0

for name in file_names:
    print(f"Traitement de {name}")
    ds = xr.open_zarr(directory + name)

    ds = ds.sel(
        time=slice('1999-01-01', '2022-12-31'), 
        latitude=slice(-80, 80)
        )
    ds = ds.chunk({ "time": 30, "latitude": 5,"longitude": 5})
    da_biomass = ds['__xarray_dataarray_variable__']  # shape (functional_group, time, lat, lon)

    # Calcul de l'intégrale biomasse × volume
    #  (fg, time, lat, lon) * (time, lat, lon)
    biomass_volume = da_biomass * volume  # (fg, time, lat, lon)

    # Somme spatiale (fg, time)
    integrated = biomass_volume.sum(dim=['latitude', 'longitude'], skipna=True)

    # Ajout d’une dimension membre
    for fg in range(10):
        da_fg = integrated.isel(functional_group=fg)
        da_fg = da_fg.expand_dims(dim={"member": [member_counter]})  # shape (member, time)
        member_data.append(da_fg)
        member_counter += 1

    ds.close()
    del ds, integrated, da_biomass, biomass_volume
    gc.collect()

# Concatène tout
integrated_all = xr.concat(member_data, dim="member")

# Métadonnées
model_labels = ['no_transport'] * 40 + ['no_transport_aragonite'] * 40 + ['acidity_aragonite'] * 40
npp_labels = ['vgpm', 'cafe', 'cbpm_westberry', 'cbpm_behrenfeld'] * 10 * 3  # 10 FG par fichier, 3 modéles
fg_labels = list(range(10)) * 12

integrated_all = integrated_all.assign_coords(
    model=("member", model_labels),
    npp=("member", npp_labels),
    fg=("member", fg_labels)
)
# Sauvegarde
integrated_all.to_zarr(directory + "biomass_integrated_volume_time_PE_Global_1999_2022.zarr", mode='w')
