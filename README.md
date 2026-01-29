# Sphere Packing Optimization Project

This project reproduces experiments for the paper **"An Efficient Solution Space Exploring Method for Sphere Packing Problem"** (Jianrong Zhou et al.).

The goal is to find the minimum radius $R$ for a container sphere to enclose $n$ non-overlapping unit spheres (radius $r=1$).

## Contributors
This project was a joint effort by:
- **Mohammadmohsen Abbaszadeh** ([@HisEgo](https://github.com/HisEgo))
- **Sina Daneshgar**

## Directory Structure
We have unified the documentation to make it easier to monitor and maintain.
- **`src/`**: Source code implementations.
  - **`matlab/`**: Original MATLAB implementations.
    - `bfgs_solver.m`: Quasi-Newton (BFGS) method.
    - `gd_solver.m`: Gradient Descent method with Potential Energy Function.
  - **`python/`**: Python translations of the algorithms using `numpy` and `matplotlib`.
    - `bfgs_solver.py`
    - `gd_solver.py`
- **`docs/`**: Documentation and Reports.
  - **`report_fa.tex`**: The complete original report in Persian (Latex source).
  - **`report_en.tex`**: English version of the report.
  - **`figs/`**: Shared figures and images for the reports.
  - **`refs.bib`**: Bibliography file.
  - **`reference_paper.pdf`**: The reference paper by Zhou et al.

## Requirements
- **Python**: Numpy, Matplotlib
- **MATLAB**: Core

## How to Run Python Scripts
```bash
python src/python/gd_solver.py
python src/python/bfgs_solver.py
```

## Methods
### Gradient Descent
Uses a potential energy function $F(X)$ comprising a repulsive term (Coulomb-like) and a harmonic attraction term to the center.
$$ F(x) = \sum_{i<j} \frac{1}{\|x_i - x_j\|} + \sum_{i} \|x_i\|^2 $$

### BFGS (Quasi-Newton)
Minimizes a penalty-based energy function directly dealing with overlaps and boundary violations.
This approach iteratively optimizes both the sphere positions and the container radius.
