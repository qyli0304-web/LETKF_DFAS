# LETKF_DFAS

This research-oriented Python platform performs dynamic sampling and data assimilation for atmospheric tracer concentration fields using an improved Local Ensemble Transform Kalman Filter (LETKF) together with a Dynamic Forward–Adjoint Sampling (DFAS) method. The framework provides end-to-end coordination of forward and adjoint transport simulations (Polyphemus/Polair3D), observation preprocessing, detector-selection strategies, and analysis-state updates.

## Features

- ETKF and LETKF implementations for ensemble-based data assimilation.
- Adaptive detector selection with forward/backward uncertainty fields.
- Integration with Polyphemus preprocessing and model execution.
- Tools for geospatial masks and key-area analysis.

## Repository Structure

- fullrun.py
  Main entry point to initialize the model, observations, and assimilation driver.
- src/
  Core modules: modeling, assimilation, filters, configuration parsing, utilities.
- obs/
  Observation post-processing utilities and plotting scripts.
- config_template/
  Template configuration and preprocessing helpers.

## Installation

1. Create a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. External dependencies (not installed by pip):

   - Polyphemus/Polair3D and its preprocessing binaries (paths configured in `model.cfg`).
   - System libraries required by geopandas, shapely, rasterio, and cartopy (GEOS, PROJ, GDAL, GEOTIFF). Use your OS package manager to install the matching C libraries.

## Configuration

- Model configuration: `model.cfg`
- Observation configuration: `obs.cfg`
- Assimilation configuration: `assimulation.cfg`
- Key areas: `keyarea.cfg`
- Polyphemus templates: `config_template/`

Ensure the paths inside these files are valid for your environment. Some scripts assume specific absolute paths to Polyphemus binaries; adapt as needed.

## Usage

- Run a full assimilation cycle:

  ```bash
  python fullrun.py
  ```

This will:

- Initialize the model, generate or synchronize run directories, and run forward and backward adjoint simulations.
- Initialize observation manager and parse configurations.
- Initialize assimilation driver and execute the assimilation procedure.

Outputs are written under forward/backward results directories configured in `model.cfg`.

## Citing

If you use this repository in academic work, please cite it appropriately. Suggested citation format:

- Project: LETKF_DFAS
- Authors: Qingyun Li
- URL: [https://github.com/your-org/LETKF_DFAS](https://github.com/qyli0304-web/LETKF_DFAS.git)
- Year: 2025



## Disclaimer

This repository contains research code and scripts that interface with external modeling systems. Paths and configurations may require adaptation to your environment.
