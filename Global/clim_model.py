# Imports 
import xarray as xr
import numpy as np
import gc

directory = "/scratch/sroyer/SEAPOPYM/output/"

# Crée une liste de noms de fichiers
def write_filenames(model):
    npp_products = ['vgpm', 'cafe', 'cbpm_westberry', 'cbpm_behrenfeld']
    return [f"{model}_{npp}.zarr" for npp in npp_products]

# Tous les fichiers : 3 modèles x 4 produits NPP = 12 fichiers, 
# chacun avec 10 FG = 120 membres
file_names = write_filenames('no_transport') \
           + write_filenames('no_transport_aragonite') \
           + write_filenames('acidity_aragonite')

member_data = []
member_counter = 0

for name in file_names:
    print(f"Traitement de {name}")
    ds = xr.open_zarr(directory + name)
    
    # Garde les années 1999–2022 (1an pour la stabilisation du modéle)
    ds = ds.sel(time=slice('1999-01-01', '2022-12-31'))

    # Ajoute le mois comme coordonnée
    ds = ds.assign_coords(month=ds['time'].dt.month)
    # Climatologie mensuelle
    ds_monthly = ds.groupby("month").mean(dim='time', keep_attrs=True)

    # Ajoute la dimension "member" pour les 10 functional groups
    for fg in range(10):
        # Sélectionne un seul functional group
        ds_fg = ds_monthly.isel(functional_group=fg)
        # Ajoute une dimension member
        ds_fg = ds_fg.expand_dims(dim={"member": [member_counter]})
        member_data.append(ds_fg)
        member_counter += 1

    ds.close()
    del ds, ds_monthly
    gc.collect()

# Concatène tous les membres le long de la nouvelle dimension "member"
clim = xr.concat(member_data, dim="member")

#Métadonnées
model_labels = ['no_transport'] * 40 + ['no_transport_aragonite'] * 40 + ['acidity_aragonite'] * 40
npp_labels = ['vgpm', 'cafe', 'cbpm_westberry', 'cbpm_behrenfeld'] * 10 * 3  # 10 FG par fichier, 3 modéles
fg_labels = list(range(10)) * 12

clim = clim.assign_coords(
    model=("member", model_labels),
    npp=("member", npp_labels),
    fg=("member", fg_labels)
)
# Sauvegarde dans un .zarr
clim.to_zarr(directory + "clim_PE_global_1999_2022.zarr", mode='w')


