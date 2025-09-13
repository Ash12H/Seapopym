# apply_ufunc vs map_blocks - Guide d'Optimisation Seapopym

## Contexte

Dans l'architecture actuelle de Seapopym, le module `core/` utilise `xarray.map_blocks` pour distribuer les calculs. Cette analyse explore l'opportunité d'utiliser `xarray.apply_ufunc` pour optimiser les performances sur les fonctions vectorisables.

## Analyse Comparative

### map_blocks (Implémentation Actuelle)

**Principe :** Applique la fonction chunk par chunk
```python
def _map_block_with_dask(self: KernelUnit, state: SeapopymState) -> xr.Dataset:
    result_template = self.template.generate(state)
    return xr.map_blocks(self.function, state, template=result_template)
    # ↑ N appels de fonction (N = nombre de chunks)
```

**Avantages :**
- ✅ Flexibilité maximale (toute fonction Python)
- ✅ Gestion de logique complexe
- ✅ Opérations inter-variables Dataset
- ✅ Architecture actuelle robuste

**Inconvénients :**
- ⚠️ Overhead des appels multiples de fonction
- ⚠️ Pas de vectorisation native
- ⚠️ Marshalling des données entre chunks

### apply_ufunc (Optimisation Proposée)

**Principe :** Vectorisation native avec parallélisation Dask
```python
def biomass_with_apply_ufunc(state: xr.Dataset) -> xr.Dataset:
    biomass = xr.apply_ufunc(
        biomass_sequence,  # Fonction Numba
        state['recruited'], state['mortality_field'],
        input_core_dims=[['time'], ['time']],
        output_core_dims=[['time']],
        dask='parallelized',
        vectorize=True,
        keep_attrs=True
    )
    return xr.Dataset({'biomass': biomass})
```

**Avantages :**
- ⚡ Vectorisation native Numpy/Numba
- ⚡ 1 appel de fonction vs N appels
- ⚡ Parallélisation Dask optimisée
- ⚡ Gain performance estimé : **2-3x plus rapide**

**Inconvénients :**
- ❌ Limité aux fonctions vectorisables
- ❌ Pas de logique conditionnelle complexe
- ❌ Gestion dimensions plus rigide

## Guide de Décision

### ✅ Utilisez apply_ufunc quand :

**Fonctions numériques pures :**
```python
# Calculs mathématiques vectorisables
def mortality_field(temperature, acidity):
    return base_mortality * np.exp(-temperature/10) * (1 + acidity**2)

# Fonctions Numba compilées
@numba.guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)')
def biomass_calculation(recruited, mortality, result):
    for i in range(recruited.shape[0]):
        result[i] = recruited[i] * np.exp(-mortality[i])
```

**Indicateurs positifs :**
- Opérations algébriques simples (`+`, `-`, `*`, `/`, `np.exp`, `np.log`)
- Fonctions Numba `@guvectorize` ou `@vectorize`
- Pas de logique conditionnelle (`if`, `for`, `while`)
- Transformation 1:1 des données

### ❌ Utilisez map_blocks quand :

**Logique complexe non-vectorisable :**
```python
# Conditions complexes
def global_mask_function(state: xr.Dataset) -> xr.Dataset:
    if state.bathymetry.max() > 0:
        return handle_land_areas(state)
    else:
        return standard_processing(state)

# Opérations stateful
def production_with_memory(state: xr.Dataset) -> xr.Dataset:
    accumulated_biomass = 0  # État persistent
    for t in state.time:
        # Logique dépendante de l'histoire
        pass

# I/O et opérations externes
def load_forcing_data(state: xr.Dataset) -> xr.Dataset:
    external_data = xr.open_dataset("forcing.nc")  # I/O
    return state.merge(external_data)
```

**Indicateurs négatifs :**
- Logique de contrôle (`if`, `for`, `while`)
- Opérations xarray complexes (`xr.where`, `xr.concat`, `interp`, `reindex`)
- I/O operations (`open_dataset`, `to_netcdf`)
- Manipulations de géométries irrégulières
- Fonctions avec état/mémoire inter-chunks

## Implémentation Recommandée

### KernelUnit Optimisé avec apply_ufunc

```python
@dataclass
class OptimizedKernelUnit:
    """KernelUnit optimisé pour fonctions vectorisables."""
    
    name: str
    template: Template
    function: Callable  # Fonction Numba
    input_core_dims: list[list[str]]
    output_core_dims: list[list[str]]
    input_vars: list[str]
    output_vars: list[str]
    
    def run(self, state: SeapopymState) -> xr.Dataset:
        input_arrays = [state[var] for var in self.input_vars]
        
        results = xr.apply_ufunc(
            self.function,
            *input_arrays,
            input_core_dims=self.input_core_dims,
            output_core_dims=self.output_core_dims,
            dask='parallelized',
            output_dtypes=[np.float64] * len(self.output_vars),
            vectorize=True,
            keep_attrs=True,
            dask_gufunc_kwargs={'allow_rechunk': True}
        )
        
        # Conversion Dataset avec métadonnées
        if len(self.output_vars) == 1:
            results = {self.output_vars[0]: results}
        else:
            results = {var: results[i] for i, var in enumerate(self.output_vars)}
            
        return xr.Dataset(results)
```

### Factory Simplifié

```python
def optimized_kernel_unit_factory(
    name: str,
    function: Callable,
    input_vars: list[str],
    output_vars: list[str], 
    input_core_dims: list[list[str]],
    output_core_dims: list[list[str]],
    template: list[type[TemplateUnit]]
):
    """Factory pour KernelUnit avec apply_ufunc."""
    
    class CustomOptimizedKernelUnit(OptimizedKernelUnit):
        def __init__(self, chunk: dict[str, int]):
            super().__init__(
                name=name, function=function,
                input_vars=input_vars, output_vars=output_vars,
                input_core_dims=input_core_dims,
                output_core_dims=output_core_dims,
                template=Template([t(chunk) for t in template])
            )
    
    return CustomOptimizedKernelUnit

# Usage
BiomassKernelOptimized = optimized_kernel_unit_factory(
    name="biomass",
    function=biomass_sequence,  # Fonction Numba existante
    input_vars=["recruited", "mortality_field"],
    output_vars=["biomass"],
    input_core_dims=[["time", "age"], ["time", "age"]],
    output_core_dims=[["time", "age"]],
    template=[BiomassTemplate]
)
```

### Architecture Hybride

```python
class SmartKernelUnit:
    """KernelUnit avec détection automatique de la méthode optimale."""
    
    def run(self, state: SeapopymState) -> xr.Dataset:
        if self._should_use_apply_ufunc():
            return self._run_with_apply_ufunc(state)
        else:
            return self._run_with_map_blocks(state)
    
    def _should_use_apply_ufunc(self) -> bool:
        """Heuristiques pour choisir la méthode."""
        func_source = inspect.getsource(self.function)
        
        vectorizable_patterns = [
            '@numba.guvectorize', '@numba.vectorize',
            'np.exp', 'np.log', 'np.sqrt'
        ]
        
        complex_patterns = [
            'if ', 'for ', 'while ',
            'xr.where', 'open_dataset'
        ]
        
        vectorizable_score = sum(p in func_source for p in vectorizable_patterns)
        complex_score = sum(p in func_source for p in complex_patterns)
        
        return vectorizable_score > complex_score
```

## Répartition Estimée pour Seapopym

### Fonctions Candidates apply_ufunc (~60%)
**Gain performance attendu : 2-3x**

```python
# Calculs numériques purs - OPTIMISABLES
✅ biomass_calculation
✅ mortality_field_calculation  
✅ temperature_averaging
✅ production_rates
✅ growth_functions
✅ recruitment_functions
```

### Fonctions Nécessitant map_blocks (~40%)
**Préserver flexibilité existante**

```python
# Logique complexe - GARDER map_blocks  
❌ global_masking
❌ boundary_conditions
❌ vertical_migration
❌ spatial_interpolation
❌ data_loading
❌ multi_variable_interactions
```

## Plan d'Action Recommandé

### Phase 1 : Proof of Concept (2 semaines)
1. **Implémenter BiomassKernelOptimized** avec apply_ufunc
2. **Benchmarker** sur données réelles vs version actuelle
3. **Valider** que résultats sont identiques
4. **Mesurer** gains de performance

### Phase 2 : Migration Sélective (4 semaines)
1. **Identifier** les fonctions vectorisables avec plus gros impact performance
2. **Migrer** progressivement vers apply_ufunc
3. **Développer** les heuristiques de détection automatique
4. **Tests** de régression complets

### Phase 3 : Architecture Hybride (6 semaines)
1. **Implémenter** SmartKernelUnit avec auto-détection
2. **Optimiser** les patterns de chunking pour apply_ufunc
3. **Documentation** et guides d'usage
4. **Benchmarks** complets performance

### Critères de Succès

**Performance :**
- Gain de 2-3x sur fonctions optimisées
- Pas de régression sur fonctions map_blocks
- Temps de simulation total réduit de 30-50%

**Compatibilité :**
- API existante préservée
- Migration transparente
- Résultats numériques identiques

**Maintenabilité :**
- Code plus simple pour nouvelles fonctions vectorisables
- Documentation claire des patterns d'optimisation
- Tests automatisés performance

## Conclusion

L'optimisation apply_ufunc représente une **opportunité majeure d'accélération** pour Seapopym. En ciblant les fonctions numériques pures (~60% du code), on peut obtenir des gains substantiels tout en préservant la flexibilité de map_blocks pour la logique complexe.

**Recommandation :** Implémenter l'architecture hybride avec migration progressive, en commençant par les fonctions à plus fort impact performance comme le calcul de biomasse.

Cette approche offre le **meilleur des deux mondes** : performance optimisée ET flexibilité maximale.