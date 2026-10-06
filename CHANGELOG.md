# Change Log

Spright uses calendar versioning (`YY.0M.0D`) since v22.12.10. The two versions before that
(v0.5.0 and v0.6.0) used semantic versioning.

## Unreleased

### Added
- Docstrings for `rmestimator.py` and `analytical_model.py`.
- GPLv3 license headers to all source and test files.
- `docs` optional dependency group (`pip install spright[docs]`).
- Python 3.13 to the CI test matrix.

### Changed
- The version is now derived from git tags with `setuptools-scm` instead of being hardcoded
  in `pyproject.toml`.
- The documentation reads its version from the installed package.
- Read the Docs installs the package before building the documentation.
- Notebooks and IDE files are excluded from the source distribution.

## v25.06.03 (2025-06-03)

### Added
- RM relations based on the Exoplanet.eu catalogue (2024.01.09): 'exoeu_m_z19', 'exoeu_m_a21',
  'exoeu_fgk_z19', and 'exoeu_fgk_a21'.
- Exoplanet.eu CSV catalogue reader.
- Unit tests for the `RMRelation` methods and the Exoplanet.eu reader.

### Changed
- Updated the TEPCat M and FGK models (both 'z19' and 'a21' water models) to the TEPCat
  catalogue of 2024.08.30.
- The available RM relations are now 'stpm', 'tepcat_m_z19', 'tepcat_m_a21', 'tepcat_fgk_z19',
  'tepcat_fgk_a21', and the four 'exoeu' relations.
- Improved TEPCat catalogue reading.
- Refactored `plot_radius_density`.
- Removed the `byteswap` and `newbyteorder` calls from relation map loading.

### Fixed
- `RMEstimator` plotting.
- `RelationMap` visualisation.
- `DensityMap` saving.

## v24.6.19 (2024-06-19)

### Added
- Input parameters can be floats, `(mean, sigma)` tuples, `ufloat`s, or frozen `scipy.stats`
  distributions.

### Changed
- Improved plotting and docstrings.

### Fixed
- Density prediction bug.

## v24.02.29 (2024-02-29)

### Added
- Relation maps separated by model component.
- Option to calculate the radius-density ICDF for a subset of model components.
- `CITATION.cff`.

### Changed
- Moved the analytical model and the log likelihood functions into their own modules.

### Fixed
- Bug in the `spright` command line script.
- Numerical stability issue.

## v23.11.01 (2023-11-01)

### Added
- Planet class prediction and plotting methods to `RMRelation`.
- Sphinx documentation.

### Changed
- Improved the `Distribution` class.

### Fixed
- Missing dependency.

## v23.10.28 (2023-10-28)

### Changed
- Improved `plot_model_means` and the `spright` command line script.

### Fixed
- Interpolation issue.

## v23.10.25 (2023-10-25)

### Added
- Model comparison notebooks and a notebook to convert the Zeng19 models.

### Fixed
- Compatibility with Python 3.8 and 3.9.

## v23.10.24 (2023-10-24)

### Fixed
- Bug with unimodal distributions.
- Circular import problem.
- `RMEstimator` fix following the `RadiusDensityModel` change.

## v23.10.23 (2023-10-23)

### Changed
- Reworked `RadiusDensityModel`, and updated the model functions, `LPF`, and `RMEstimator` to
  use it.
- Removed old files.

## v23.10.19.01 (2023-10-19)

### Fixed
- `pyproject.toml`.

## v23.10.19 (2023-10-19)

### Added
- `spright` command line script.
- Sub-Neptune density constraint to the `LPF`.
- `RMEstimator` mock data tests.
- Safeguards to mass and radius sample generation.

### Changed
- Updated the radius-density-mass relation maps.
- Improved catalogue handling in `RMRelation`.
- Changed how the posterior model means are calculated.
- Improved `Distribution` printing.

### Fixed
- Relation maps now work on macOS as well as on Linux.
- Bug in the saved radius-mass maps.
- `plot_model_means` and `RMRelation._identify_modes`.
- Missing `__init__.py` and dependency.

## v23.09.19 (2023-09-19)

### Added
- Function to create mock datasets.

### Fixed
- Bug in the ICDF boundary computation.

## v23.09.06 (2023-09-06)

### Added
- Notebook visualising the analytical model parameterisation.

### Changed
- Made `map_r_to_xy` more robust.

## v23.09.04 (2023-09-04)

### Added
- Option to set the random seed in `RMEstimator`.
- Colour map argument to `plot_map`.

### Changed
- Changed the model and `LPF` parameterisation, and revised the `LPF` priors.
- Updated the default radius-density maps.

### Fixed
- Analytical model log likelihood.
- `RMEstimator` sampling bug: the chains were always started from the optimisation result.

## v23.08.31 (2023-08-31)

### Changed
- `RMEstimator` now uses a global optimiser.
- Recalculated the main 'stpm' maps.

## v23.08.29 (2023-08-30)

### Fixed
- `version.py` still used the old package name.

## v23.06.28 (2023-06-28)

### Changed
- Added the `RMRelation.sample` method back as deprecated.

## v23.06.27 (2023-06-27)

### Added
- `RelationMap` class that handles the CDF and ICDF calculation, with loading support.

### Changed
- Reworked `RMRelation`: the single `sample` method is replaced by `predict_mass`,
  `predict_radius`, `predict_density`, and `predict_rv_semi_amplitude`.
- `RMEstimator` and `RMRelation` now use `RelationMap`.
- Updated the STPM and FGK models.

### Fixed
- `RMRelation` and `RelationMap` bugs.

## v23.06.24 (2023-06-19)

### Added
- GitHub Actions test workflow and a first basic test.
- Missing dependencies.

### Changed
- Renamed the package directory from `moot` to `spright`.

### Fixed
- ICDF calculation bug.

## v23.06.16 (2023-06-16)

### Added
- Radius-mass map creation.
- Beta prior for the water-rich planet population width.

### Changed
- Renamed the package from `moot` to `spright`.
- Changed the model parameterisation.
- Sub-Neptune densities are now represented by a power law.
- Fixed the Student's t-distribution degrees of freedom to five.
- Separated the catalogue reading functions from `RMEstimator`.

## v22.12.12 (2022-12-12)

### Added
- Example notebooks.

### Changed
- The mixture weights are now calculated using interpolation inside a triangle.

## v22.12.10 (2022-12-10)

First version using calendar versioning (`YY.0M.0D`). The earlier versions used semantic
versioning.

### Added
- Model scenario without water worlds.
- Utility methods to calculate and plot the model means.
- Two planet data tables by Luque.

### Changed
- Changed to use the Zeng radius-density models.
- The version is now read from the package metadata.
- Improved `RMRelation` and `RMEstimator`.

## v0.6.0 (2022-10-29)

### Added
- First version of the default radius-density map.

### Changed
- Major core cleanup.
- Cleaned up the `LPF` and improved its priors.
- Improved the information stored in the saved RM model files.

## v0.5.0 (2022-10-24)

First versioned release.

### Added
- `RadiusMassRelation`, separated from the `LPF`.
- `Distribution` class.
- `pyproject.toml`.

### Changed
- Renamed the package from `mmmbop` to `moot`.
