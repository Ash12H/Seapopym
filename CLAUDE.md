# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Environment Setup:**
```bash
poetry install                    # Install all dependencies
poetry install --with dev,doc    # Install with development and documentation dependencies
```

**Code Quality:**
```bash
poetry run ruff check ./seapopym     # Lint code using Ruff
poetry run ruff format ./seapopym    # Format code (equivalent to Black)
make lint                            # Alternative: use Makefile
make format                          # Alternative: use Makefile
```

**Documentation:**
```bash
make doc                          # Generate documentation using Sphinx
# Requires pandoc for notebook conversion
# Outputs to docs/build/html/
```

**Publishing:**
```bash
make publish_test                 # Publish to TestPyPI
make publish                      # Publish to PyPI
```

**Notebooks:**
```bash
poetry run jupyter notebook      # Launch Jupyter Notebook
poetry run jupyter lab           # Launch Jupyter Lab
```

## Core Architecture

Seapopym implements a **sophisticated layered architecture** built around configuration-driven model execution for simulating marine ecosystem dynamics.

### Architecture Flow
```
Configuration Layer → Model Layer → Kernel Layer → Function Layer
```

**Central Data Structure**: `SeapopymState` (alias for `xr.Dataset`) flows through the entire pipeline, containing:
- Forcing data (temperature, primary production, acidity)
- Model parameters for functional groups
- Intermediate and final computation results
- Support for both in-memory and distributed (Dask) computation

### Key Design Patterns

**1. Factory-Based Kernel Construction**
```python
# Models built by composing ordered sequences of computational functions
AcidityKernel = kernel_factory(
    class_name="AcidityKernel",
    kernel_unit=[
        function.GlobalMaskKernel,
        function.MaskByFunctionalGroupKernel,
        function.DayLengthKernel,
        # ... more kernel units
    ],
)
```

**2. Abstract Base Class Hierarchy**
- `AbstractConfiguration` → `NoTransportConfiguration` → `AcidityConfiguration`
- Each configuration type is a specialized "flavor" of the marine ecosystem model

**3. Template System for xarray Operations**
Templates enable efficient distributed computation while maintaining metadata:
```python
BiomassTemplate = template.template_unit_factory(
    name=ForcingLabels.biomass,
    attributs=biomass_desc,
    dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, ...]
)
```

### Domain Architecture

**Functional Groups & Cohorts**: Marine organisms modeled as functional groups (zooplankton, micronekton) with age-structured cohorts featuring:
- Temperature and environmental relationships
- Time-dependent recruitment and mortality  
- Day/night vertical migration patterns

**Configuration Types**:
- **No Transport**: Base model without advection-diffusion transport
- **Acidity**: Extends base model with ocean acidification effects

### Computational Pipeline
1. **GlobalMaskKernel**: Apply spatial masks for valid ocean areas
2. **MaskByFunctionalGroupKernel**: Create functional group masks
3. **Environmental Processing**: Temperature, acidity, day length calculations
4. **Production Calculations**: Primary production, mortality, recruitment
5. **Biomass Integration**: Final biomass using compiled Numba functions

### Performance Architecture
- **Numba Integration**: Critical computations use Numba for performance (`/function/compiled_functions/`)
- **Dask Integration**: Full distributed computation support via `xarray.map_blocks`
- **Lazy Loading**: On-demand environmental data loading with configurable chunking

### Key Design Decisions

**1. CF-Compliant Data Standards**
- Heavy use of cf-xarray for climate science conventions
- Standardized coordinate names (T, X, Y, Z)
- Proper unit handling with pint-xarray

**2. Immutable Configuration**
- `@frozen` attrs classes for parameters with extensive validation
- Enum-based labels (`CoordinatesLabels`, `ForcingLabels`) prevent string errors

**3. Functional Parameter Validation**
```python
lambda_temperature_0: ParameterUnit = field(
    converter=partial(verify_init, unit="1/day", parameter_name=...),
    validator=validators.ge(0),
)
```

## Module Organization

- `/standard/`: Common types, labels, coordinates, and units
- `/core/`: Kernel execution engine and template system  
- `/configuration/`: Parameter classes by model type (acidity, no_transport)
- `/function/`: Individual computational functions and compiled kernels
- `/model/`: High-level model classes that orchestrate execution

## Extension Patterns

**Adding New Configuration Types:**
1. Create directory under `/configuration/`
2. Implement forcing, functional group, and environment parameters
3. Extend `AbstractConfiguration` 
4. Create model class extending `BaseModel`

**Adding New Kernel Functions:**
1. Implement function in `/function/` with signature `(SeapopymState) -> xr.Dataset`
2. Create TemplateUnit factory for output structure
3. Create KernelUnit factory combining function and template
4. Add to kernel composition in model files

## Unique Features

1. **Dynamic Kernel Composition**: Models defined by ordered sequences of kernel units
2. **Distributed-First Design**: Built from ground up for Dask distributed computation
3. **Cohort Age Modeling**: Sophisticated age-structure with configurable timestep aggregation
4. **Multi-dimensional Environmental Relationships**: Complex temperature/acidity/mortality interactions
5. **Lazy Forcing Data**: Environmental data loaded on-demand with remote file support

This architecture enables flexible marine ecosystem modeling while maintaining computational efficiency and scientific accuracy.

## Git Commit Guidelines

**Important**: When making commits, do NOT include Claude as co-author. Use only standard commit messages without the Claude Code attribution footer.