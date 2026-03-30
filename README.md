# How Long to Boil an Egg? FEM Modeling with Python

A finite element simulation of heat transfer during egg boiling, solving the axisymmetric transient heat equation:

$$\rho(T)\,C_p(T)\,\frac{\partial T}{\partial t} = \nabla \cdot \bigl(\kappa(T)\,\nabla T\bigr)$$

Built with [scikit-fem](https://github.com/kinnala/scikit-fem) and [gmsh](https://gmsh.info/), this project models the temperature evolution inside an egg immersed in boiling water, predicting cooking times for soft-, medium-, and hard-boiled eggs.

## Reference

This project is a Python re-implementation inspired by the Wolfram Language blog post:

> **How Long to Boil an Egg? FEM Modeling with Wolfram Language**
> Wolfram Blog, September 15, 2025
> https://blog.wolfram.com/2025/09/15/how-long-to-boil-an-egg-fem-modeling-with-wolfram-language/

## Project Structure

```
fem_boil_egg/
├── environment.yml               # Conda environment specification
├── requirements.txt              # pip dependencies
├── egg_fem/                      # Core Python package
│   ├── __init__.py
│   ├── materials.py              # Temperature-dependent material properties
│   ├── geometry.py               # Mesh generation (circular + egg-shaped)
│   └── solver.py                 # Axisymmetric FEM heat solver
└── notebooks/
    ├── 01_simple_model.ipynb     # Simple circular geometry, constant properties
    ├── 02_realistic_model.ipynb  # Egg-shaped geometry, multi-region, T-dependent
    └── 03_cooking_analysis.ipynb # Cooking time predictions & parameter studies
```

## Setup

### With conda (recommended)

```bash
conda env create -f environment.yml
conda activate fem-boil-egg
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install jupyterlab
```

### Run the notebooks

```bash
jupyter lab notebooks/
```

## Notebooks

### 1. [Simple Model](notebooks/01_simple_model.ipynb)

Introduces the FEM approach with a semicircular cross-section and constant material properties. Demonstrates mesh generation, solver setup, and basic visualization.

### 2. [Realistic Model](notebooks/02_realistic_model.ipynb)

Adds a realistic egg-shaped geometry (parametric B-spline profile), separate yolk and white regions with distinct temperature-dependent material properties, and refined meshing near the shell boundary.

### 3. [Cooking Analysis](notebooks/03_cooking_analysis.ipynb)

Uses the validated model to answer practical questions:
- Room temperature (20 C) vs. refrigerated (8 C) eggs
- Effect of egg size (small, medium, large)
- Predicted cooking times for soft-, medium-, and hard-boiled eggs

## Model Details

### Geometry

The egg is modeled as an axisymmetric body in the (r, z) half-plane. The outer profile uses a parametric curve and the yolk is a semicircular sub-region offset slightly below center.

### Material Properties

| Property             | Egg White        | Egg Yolk         |
|----------------------|------------------|------------------|
| Density              | 1034-1040 kg/m^3 | 1020-1030 kg/m^3 |
| Thermal conductivity | 0.50-0.60 W/(m K)| 0.32-0.36 W/(m K)|
| Specific heat        | ~3800 J/(kg K)   | ~3300 J/(kg K)   |

All properties vary with temperature (linear fits from experimental data).

### Solver

- Backward Euler (implicit) time integration
- Lagged nonlinear coefficients (properties evaluated at T^n to solve for T^{n+1})
- Dirichlet boundary at the shell (T = 100 C), natural symmetry condition at r = 0

### Key Results

| Cooking Stage | Room Temp Start | Fridge Start |
|---------------|-----------------|--------------|
| Soft-boiled (63 C center)   | ~8 min  | ~9-10 min |
| Medium-boiled (65 C center) | ~9 min  | ~10-11 min |
| Hard-boiled (70 C center)   | ~10 min | ~12 min   |

## License

This project is provided for educational purposes.
