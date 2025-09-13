# Roadmap Optimisé pour Algorithmes Génétiques - Seapopym

## 🎯 **Contexte d'Usage Critique**

**Workflow d'optimisation** :
```
Algorithme Génétique (externe) 
    ↓ génère N jeux de paramètres
    ↓ pour chaque paramètre_set:
        → Créer modèle Seapopym(paramètre_set, forçages_fixes)
        → Calculer biomasse_finale
        → Extraire score fitness
    ↓ répéter sur plusieurs générations
```

**Contraintes critiques identifiées** :
- **Mêmes forçages** réutilisés pour tous les modèles (optimisation possible)
- **Création/destruction répétée** de modèles (risque memory leak)
- **Score basé sur biomasse** (performance calcul final critique)
- **Volume élevé** simulations (centaines/milliers par génération)

---

## 🚨 **Points Critiques à Absolument Traiter**

### **1. Memory Leak Prevention (CRITIQUE ⭐⭐⭐)**

**Problème** : Création répétée de modèles → accumulation progressive mémoire
```python
# DANGER - Pattern actuel probablement utilisé
for param_set in genetic_population:
    model = NoTransportModel(param_set, forcings)  # New instance each time
    result = model.run()                           # Potentially keeps references
    score = extract_fitness(result.biomass)       # Large arrays in memory
    # model goes out of scope but may not be freed immediately
```

**Solution obligatoire** : Context managers + explicit cleanup
```python
# SAFE - Pattern à implémenter
@contextmanager
def genetic_simulation(param_set, shared_forcings):
    model = create_optimized_model(param_set, shared_forcings)
    try:
        yield model
    finally:
        model.cleanup()  # Explicit resource release
        gc.collect()     # Force garbage collection

for param_set in genetic_population:
    with genetic_simulation(param_set, forcings) as model:
        result = model.run()
        score = extract_fitness(result.biomass)
        # Auto-cleanup guaranteed
```

### **2. Shared Forcings Optimization (PERFORMANCE ⭐⭐⭐)**

**Opportunité majeure** : Forçages identiques → chargement unique
```python
# BEFORE - Inefficient
for param_set in population:
    model = Model(param_set)
    model.load_forcings(forcing_files)  # Reload same data every time
    
# AFTER - Optimized  
shared_forcings = PreloadedForcings(forcing_files)  # Load once
for param_set in population:
    model = Model(param_set, forcings=shared_forcings)  # Reference shared data
```

**Gains attendus** : 10x faster model initialization + 90% moins mémoire

### **3. Biomass Extraction Performance (CRITIQUE ⭐⭐⭐)**

**Goulot d'étranglement probable** : Calcul/extraction finale biomasse
```python
# Optimisations essentielles
- apply_ufunc pour calcul biomasse (2-3x speedup)
- Extraction sélective (éviter copy full datasets)
- Compiled functions pour métriques fitness
- Cache intermédiaires si possible
```

---

## 📋 **Roadmap Révisé pour Usage Génétique**

### **Phase 0 : Genetic Algorithm Profiling** ⭐⭐⭐ (1 semaine)

**Tâche 0.1 : Workflow Analysis**
```python
# Profiler EXACTEMENT votre workflow actuel
- Temps par simulation individuelle
- Memory usage pattern over generations  
- Identifier les 3 plus gros bottlenecks
- Mesurer overhead model creation/destruction
```

**Tâche 0.2 : Shared Resources Identification**
```python
# Cataloguer tout ce qui peut être partagé
- Forcings data (obviament)
- Coordinate definitions
- Template structures
- Compiled functions
- Configuration objects immuables
```

**Tâche 0.3 : Memory Pattern Baseline**
```python
# Test actuel sur mini genetic run
def profile_genetic_memory():
    memory_samples = []
    for generation in range(5):
        for individual in small_population:
            mem_before = psutil.virtual_memory().used
            score = run_simulation(individual)
            mem_after = psutil.virtual_memory().used
            memory_samples.append(mem_after - mem_before)
    return analyze_memory_pattern(memory_samples)
```

### **Phase 1 : Genetic-Optimized Architecture** ⭐⭐⭐ (2 semaines)

**Tâche 1.1 : Shared Resource Manager**
```python
class GeneticResourceManager:
    def __init__(self, forcing_files):
        self.forcings = self._load_shared_forcings(forcing_files)
        self.templates = self._precompile_templates()
        self.coords = self._setup_coordinates()
    
    def create_model(self, params) -> SeapopymModel:
        """Fast model creation using shared resources."""
        return SeapopymModel(
            parameters=params,
            forcings=self.forcings,        # Shared reference
            templates=self.templates,      # Pre-built
            coordinates=self.coords        # Reused
        )
```

**Tâche 1.2 : Memory-Safe Simulation Context**
```python
@contextmanager  
def genetic_individual_simulation(resource_manager, params):
    model = resource_manager.create_model(params)
    try:
        yield model
    finally:
        model._clear_computation_cache()  # Clean intermediate results
        del model  # Explicit deletion
        # Note: shared resources remain in memory
```

**Tâche 1.3 : Fast Fitness Extraction**
```python
def extract_fitness_optimized(seapopym_result):
    """Ultra-fast biomass → fitness conversion."""
    # Use apply_ufunc for vectorized operations
    # Avoid unnecessary data copying
    # Return scalar/small array only
    return compiled_fitness_function(seapopym_result.biomass)
```

### **Phase 2 : Performance Critical Path** ⭐⭐⭐ (2 semaines)

**Tâche 2.1 : apply_ufunc Priority Migration**
```python
# Focus UNIQUEMENT sur functions utilisées dans biomass calc
Priority_Order = [
    "biomass.py",           # Final computation
    "production.py",        # Core calculation  
    "mortality_field.py",   # Parameter-sensitive
]
# Objectif: 2x speedup minimum sur ces 3 fonctions
```

**Tâche 2.2 : Compiled Fitness Functions**
```python
@numba.jit(nopython=True)
def compiled_fitness_metrics(biomass_array):
    """JIT-compiled fitness calculations."""
    # Ultra-fast biomass processing
    # Custom metrics optimized for your GA
    return fitness_score

# Integration with seapopym
def fast_fitness_extraction(seapopym_state):
    biomass = seapopym_state[ForcingLabels.biomass].values
    return compiled_fitness_metrics(biomass)
```

**Tâche 2.3 : Kernel Chain Optimization**
```python
# Optimize kernel execution for repeated similar runs
class GeneticOptimizedKernel:
    def __init__(self):
        self._cached_intermediates = {}
        
    def run(self, params, shared_forcings):
        # Cache invariant computations
        # Skip redundant calculations
        # Optimize for parameter-only variations
```

### **Phase 3 : Genetic Algorithm Integration** ⭐⭐ (1 semaine)

**Tâche 3.1 : High-Level GA Interface**
```python
class SeapopymGeneticOptimizer:
    def __init__(self, forcing_files, model_type="no_transport"):
        self.resources = GeneticResourceManager(forcing_files)
        
    def evaluate_population(self, parameter_population):
        """Evaluate entire GA population efficiently."""
        scores = []
        for params in parameter_population:
            with genetic_individual_simulation(self.resources, params) as model:
                result = model.run()
                score = extract_fitness_optimized(result)
                scores.append(score)
        return np.array(scores)
        
    def cleanup_generation(self):
        """Call between generations to prevent memory accumulation."""
        gc.collect()
        self.resources.clear_caches()
```

**Tâche 3.2 : Memory Monitoring Integration**
```python
# Protection automatique contre explosion mémoire
def monitored_genetic_evaluation(population, max_memory_gb=4):
    memory_samples = []
    
    for i, params in enumerate(population):
        if psutil.virtual_memory().used > max_memory_gb * GB:
            warning(f"Memory limit approaching at individual {i}")
            gc.collect()  # Emergency cleanup
            
        score = evaluate_individual(params)
        memory_samples.append(psutil.virtual_memory().used)
    
    return scores, memory_samples
```

### **Phase 4 : Validation & Benchmarks** ⭐⭐ (1 semaine)

**Tâche 4.1 : Genetic Algorithm Stress Test**
```python
def test_genetic_scalability():
    # Simulate realistic GA run
    populations = [50, 100, 200]  # Typical GA sizes
    generations = [10, 20, 50]    # Duration tests
    
    for pop_size in populations:
        for gen_count in generations:
            memory_peak = run_genetic_simulation(pop_size, gen_count)
            assert memory_peak < MEMORY_LIMIT
            
def test_no_memory_accumulation():
    # Critical: memory should be stable across generations
    memory_per_generation = []
    for gen in range(20):
        mem_before = get_memory_usage()
        run_single_generation(population_size=100)
        mem_after = get_memory_usage()
        memory_per_generation.append(mem_after)
    
    # Memory should NOT grow linearly with generations
    assert np.std(memory_per_generation) < ACCEPTABLE_VARIANCE
```

---

## 🎯 **Métriques de Succès Spécifiques**

### **Performance Targets**
```python
# Baselines à atteindre (ajustables selon votre hardware)
TARGET_METRICS = {
    "simulation_time": "<10s per individual",      # Actuel probablement 30-60s
    "memory_per_simulation": "<256MB",             # Éviter accumulation
    "population_100_time": "<15min total",        # Population entière
    "memory_stability": "<5% variance générations", # Pas de leak
    "initialization_speedup": ">5x faster",       # Shared forcings
    "biomass_extraction": ">3x faster",           # apply_ufunc
}
```

### **Validation Checklist**
- [ ] Aucune accumulation mémoire sur 50 générations
- [ ] Shared forcings correctement réutilisés
- [ ] Temps simulation linéaire avec taille population  
- [ ] Results identiques avant/après optimisation
- [ ] API simple pour intégration GA externe
- [ ] Monitoring automatique ressources
- [ ] Recovery graceful en cas problème mémoire

---

## 🚀 **Quick Wins Prioritaires**

**Implémentation immédiate recommandée** :

1. **Context manager pour simulations** (2-3h dev)
2. **Shared forcings preloading** (1 jour dev)  
3. **Memory monitoring hooks** (0.5 jour dev)
4. **Fast fitness extraction** (1 jour dev)

Ces 4 points couvriront 80% de vos gains performance/mémoire !

---

## ⚠️ **Pièges à Éviter Absolument**

1. **Ne pas optimiser l'architecture kernel/template** → Elle est déjà excellente
2. **Ne pas paralléliser les simulations** → GA externe gère déjà ça
3. **Ne pas complexifier l'API** → Garder interface simple
4. **Ne pas toucher aux calculs scientifiques** → Focus sur infrastructure

**Principe directeur** : Optimiser l'**infrastructure d'exécution**, pas la **science computationnelle** qui est déjà au top ! 🎯