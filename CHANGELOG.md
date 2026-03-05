# Changelog

Observes [Semantic Versioning](https://semver.org/spec/v2.0.0.html) standard and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) convention.

## [1.3.7] - 2026-03-05

+ Fix - `update_kpms_dj_config()` and `dj_generate_config()` now route flat hyperparameter kwargs (`kappa`, `latent_dim`, `sigmasq_loc`, etc.) into their nested config dicts (`trans_hypparams`, `ar_hypparams`, `cen_hypparams`). Previously, these were added as top-level keys only; keypoint-moseq's `init_model()` reads the nested dicts and ignores flat `**kwargs`. This caused all models to be initialized with default hyperparameters (`latent_dim=10`, `sigmasq_loc=0.5`) regardless of the values specified in `PreFitTask` or `FullFitTask`. Affects `PreFit`, `FullFit`, and `Inference`.

## [1.3.6] - 2026-03-04

+ Fix - `compute_syllable_metrics()` now applies the data mask when computing syllable durations from checkpoints. Checkpoints store z as a 2D padded array (N_videos, max_T); z values at padded positions are random (sampled from the prior during Gibbs sampling). Without masking, these spurious short segments corrupted the median duration statistic — e.g. 67ms instead of 300ms for datasets with unequal video lengths (~40% padding). Affects `PreFitQA` and `FullFitQA`.

## [1.3.5] - 2026-03-03

+ Fix - `compute_syllable_metrics()` now applies frequency filtering (MIN_FREQUENCY = 0.5%) when counting syllables, matching the convention in Weinreb et al. 2024. Previously, `num_syllables` counted all unique syllable IDs (always ~100 with `num_states=100`), ignoring rare states. Now only syllables whose instance frequency >= 0.5% of total instances are counted, consistent with `get_frequencies(runlength=True)` from jax-moseq and the filtering already used in `MotionSequence` and `TrajectoryPlot`.
+ Update - Refactor `compute_syllable_metrics()` loop to a single code path — `np.diff` handles all cases (single frame, uniform sequence, normal) without branching.

## [1.3.4] - 2026-03-03

+ Fix - Normalize `BodyParts` blob fields (`anterior_bodyparts`, `posterior_bodyparts`, `use_bodyparts`) at all fetch sites. When inserted via the dashboard (dash-datajoint-components), blob attributes are stored as string representations (e.g. `"['nose', 'head', 'tail_base']"`) instead of native Python lists, breaking downstream `set()` and `.index()` calls in `PreProcessing`, `Inference`, `TrajectoryPlot`, etc. Added `BodyParts.normalize_bodyparts_blob()` static method and applied it at all 8 fetch sites across `moseq_train`, `moseq_infer`, and `moseq_report`.

## [1.3.3] - 2026-02-20

+ Fix - `FullFit.make_compute` was not applying `full_kappa` to the model hyperparameters. The PreFit model (with `pre_kappa` baked in) was loaded but `update_hypparams(model, kappa=full_kappa)` was never called, causing all FullFit models to train with `pre_kappa` regardless of the `full_kappa` value in `FullFitTask`. This resulted in identical QA metrics and model scores across different kappa values. Introduced in v1.3.0 when `find_prefit_model`/`initialize_model_for_fitting` were replaced with `load_prefit_model`.

## [1.3.2] - 2026-02-20

+ Fix - `TrajectoryPlot` grid movie generation crashes with `ValueError: frame number requested outside video bounds` when DLC coordinate files have more frames than the actual video (see [keypoint-moseq#149](https://github.com/dattalab/keypoint-moseq/issues/149)). Build `video_frame_indexes` that clamps frame indices to each video's actual length before calling `generate_grid_movies`.

## [1.3.0] - 2026-02-12

> **BREAKING CHANGES** - This version contains breaking schema changes. Tables must be dropped and recreated.

### Breaking Changes
+ **BREAKING**: `FullFitTask` now depends on `PreFit` instead of `PCAFit` — the pipeline flow is always PreFit → FullFitTask → FullFit
+ **BREAKING**: Rename `pre_latent_dim` to `latent_dim` in `PreFitTask`/`PreFit` — shared by both stages
+ **BREAKING**: Remove `full_latent_dim` from `FullFitTask`/`FullFit` — inherited from `PreFit` via FK
+ **BREAKING**: Remove `full_kappa` secondary attribute from `FullFitQA` — now available as PK attribute via `FullFit` FK

### Changes
+ Remove `FullFit.key_source` property — no longer needed since PreFit dependency is structural
+ Simplify PreFit model loading in `FullFit.make_compute` — fetch directly via FK chain instead of searching
+ Replace `find_prefit_model` and `initialize_model_for_fitting` with simpler `load_prefit_model` in `kpms_reader.py`
+ Update tests to reflect renamed attributes

## [1.2.1] - 2025-11-26
+ Fix - Fix `KeyError` in `FullFit` when querying `PreFitTask`
+ Fix - Set `set_mixed_map_gpus` to 6 in `FullFit` to reduce GPU memory usage
+ Fix - Minor fix in position of `datetime.now` in `PreProcessing`
+ Update - Refactor `TrajectoryPlot` to use `video_paths` if available and complete, or use `video_dir` as fallback
+ Update - Add 3-part make pattern in `TrajectoryPlot`, refactor into helper functions and cleanup in `moseq_report`
+ Update - Refactor `FullFit` to use 3-part make pattern
+ Update - Set `save_every_n_iters` in `moseq_train` for improved checkpoint management
+ Update - Increase video duration in `PreProcessingQA` to 10 seconds by default
+ Add - Save coordinates and confidences as pickle files during `trigger` mode in `Inference`
+ Add - Revert overlaying keypoints over the grid movies
+ Add - Add clarifying comments in codebase
+ Add - Apply black formatting over `moseq_infer`
+ Add - Cleanup `moseq_infer.py` and `moseq_train.py`
+ Add - Add comments in `kpms_reader.py`
+ Add - Refactor `viz_utils.py` module


## [1.2.0] - 2025-11-13
> **BREAKING CHANGES** - This version contains breaking changes due to keypoint-moseq upgrade and API refactoring. Please review the changes below and update your code accordingly.

### Breaking Changes
+ **BREAKING**: Rename VideoRecording to RecordingSet for clearer meaning in `moseq_infer` schema
+ **BREAKING**: Integrate `PreProcessingReport` after `PreProcessing` in `moseq_train`, rather than in `moseq_report`
+ **BREAKING**: Rename `Bodyparts` to `BodyParts` to match naming conventions

### New Features and Fixes
+ Feat - External storage of files (and configuration file versions) via attach and filepath, covering items like model_file, config_file, coordinates_file, and confidences_file
+ Update - Addition of duration fields in heavily computed tables for performance reports
+ Feat - Include a method to infer the output directory in `InferenceTask`
+ Update - Enhance Inference computation for robustness, supporting various keypointset files and adding outlier_removal
+ Feat - Introduce a separate `MotionSequence` table for inference to address connection loss issues
+ Feat - Add `BehavioralSummary` and `TrajectoryPlot` to `moseq_report` schema to analyze and store visualization outputs from inference
+ Update - Implement specific JAX configurations to unify table precision and ensure correct generation and ingestion of JAX-moseq data objects
+ Feat -Add a method in PCATask to infer the kpms_project_output_dir
+ Update - Include sanity checks for proper usage of stored files in downstream tables, e.g., coordinates
+ Update -Correct bodyparts handling to analyze only the specified ones in the BodyParts pipeline instead of the base config
+ Fix - outlier_removal application on data
+ Feat - add PreProcessingQA to generate and store quality assurance data, including a table of NaNs per bodypart
+ Update - Enable `SelectedFullFit` to automatically rank models using the Marginal Log Likelihood (MLL) score
+ Update - Improve kpms_reader helper functions for strict directory and config validation
+ Update - Enhance plotting utilities in viz_utils for better return values
+ Update - Implement stricter input validation
+ Update - Modify plotting functions to provide more useful output
+ Update - Enforce absolute path requirements in file operations

## [1.1.0] - 2025-10-15

> **BREAKING CHANGES** - This version contains breaking changes due to keypoint-moseq upgrade and API refactoring. Please review the changes below and update your code accordingly.

### Breaking Changes
+ **BREAKING**: Remove `recording_name` attribute from `moseq_report` since it will be added in `VideoFile` in `moseq_train`
+ **BREAKING**: Add new `pose_estimation_path` attribute in `KeypointSet.VideoFile`

### New Features and Fixes
+ Fix - Update `moseq_report` and `moseq_train` to use new `QA` dir name instead of `quality_assurance`
+ Fix - Update logic in `PreProcessing` to check for outlier plots
+ Fix - add `poppler` as system dependency in `conda_env.yml`
+ Update - code cleanup related to `copy_pdf_to_png` function
+ Update - remove unused helper functions in `viz_utils` module after these changes

## [1.0.2] - 2025-10-07
+ Update - `kpms` as extra dependency (includes `keypoint-moseq`)
+ Fix - Version pin `jax<0.7.0`

## [1.0.1] - 2025-09-23
+ Feat - Add support to generate PNG version of fitting progress plots in `PreFit`, `FullFit`, and `moseq_report` schema
+ Fix - Update path handling to use `Path` objects and `dj.logger`

## [1.0.0] - 2025-09-10

> **BREAKING CHANGES** - This version contains breaking changes due to keypoint-moseq upgrade and API refactoring. Please review the changes below and update your code accordingly.

### Breaking Changes
+ **BREAKING**: Upgrade keypoint-moseq from pinned 0.4.8 version to the latest version from source with breaking changes adding new features that are not compatible with the previous kpms versions
+ **BREAKING**: Rename `kpms_reader` functions to generate, load, and update kpms config files
+ **BREAKING**: Rename `PCAPrep` to `PreProcessing` table and add new attributes
+ **BREAKING**: Add feature to remove outlier keypoints in `PreProcessing` table with new `outlier_scale_factor` attribute in `PCATask`, only available in latest version of kpms
+ **BREAKING**: Add `sigmasq_loc` feature in `PreFit` and `FullFit` to automatically estimate sigmasq_loc (prior controlling the centroid movement across frames), only available in latest version of kpms

### New Features and Fixes
+ Feat - Add support to load from both DLC `config.yml` and `config.yaml` file extensions
+ Feat - Add new `mosesq_report` schema with comprehensive reporting capabilities
+ Feat - Refactor `PreProcessing` table to use 3-part make function, add a new `Video` part table, and add new attributes `video_duration`, `frame_rate` and `average_frame_rate` to store these new computations
+ Feat - Move and refactor `viz_utils` into new `plotting` module
+ Feat - Update `model_name` varchar and folder naming
+ Feat - Migrate from `setup.py` to `pyproject.toml` and `conda_env.yml` for modern Python packaging standards
+ Fix - Update devcontainer to use Python 3.11 and upgrade dependencies
+ Fix - Remove JAX dependencies from `pyproject.toml`
+ Fix - Update pre-commit hooks for improved linting and consistency
+ Fix - Correct generation of `kpms_dj_config.yml` and refactor `moseq_train` and `moseq_infer` to use the renamed functions
+ Fix - Refactor `moseq_infer` and `moseq_train`, and implement a three-part make function in the most resource-intensive functions
+ Fix - Update folder naming logic to use string of combined primary attributes instead of datetime in `PreFit` and `FullFit`
+ Fix - Improve path and directory handling using `Path` objects and robust existence checks
+ Fix - Remove redundancy of variables in `PreProcessing` table
+ Fix - Update deprecated datetime usage
+ Fix - Fix filename generation in `Inference` table
+ Fix - Bugfix in `Model` imported foreign key
+ Fix - Update tutorial_pipeline
+ Add - Update docstrings across all modules
+ Add - Update pipeline images to reflect new architecture

## [0.3.2] - 2025-08-25
+ Feat - modernize packaging and environment management migrating from `setup.py` to `pyproject.toml`and `env.yml`
+ Fix - JAX compatibility issues
+ Add - update pre-commit hooks

## [0.3.1] - 2025-06-27

+ Fix - `setup.py` to install `keypoint-moseq` as a required dependency

## [0.3.0] - 2025-06-07

+ Feat - Created new `SelectedFullFit` table to register selected trained models for downstream inference
+ Feat - Updated pipeline architecture by inverting dependency direction:`moseq_infer.PoseEstimationMethod` is moved from `moseq_infer` to `moseq_train` and now the new table `SelectedFullFit` is a nullable foreign key in `moseq_infer.Model`.
+ Fix - Updated imports and foreign key references across schemas to match new structure.
+ Fix - `pre-commit` hooks to exclude removal of used dependencies into the table definitions (`F401`).
+ Add - Refactored `setup.py` to organize dependencies more cleanly and into optional dependencies.
+ Add - Refactored schema activation logic.
+ Add - Adjusted `tutorial_pipeline.py` to reflect the new hierarchy
+ Add - Aligned schema design with DataJoint Element conventions to support modular reuse and testing
+ Add - Update `images` according to these changes
+ Add - Update `tutorial.ipynb` to reflect these changes.
+ Add - Style and minor bug fixes.

## [0.2.3] - 2025-04-12

+ Fix - `moseq_train` to import `keypoint_moseq` functions inside `trigger` mode

## [0.2.2] - 2025-01-24

+ Fix - `url_site` in `mkdocs.yaml` to point to the correct URL
+ Fix - revert GHA semantic release

## [0.2.1] - 2024-08-30

+ Fix - `mkdocs` build issues
+ Fix - `reader` module imports by adding `__init__.py`
+ Fix - Move KPMS installation to `extras_require` in `setup` for consistency with other Elements
+ Update - markdown files in `mkdocs`
+ Update- Dockerfile

## [0.2.0] - 2024-08-16

+ Add - `load` functions and new secondary attributes for tutorial purposes
+ Add - `outbox` results in the public s3 bucket to be mounted in Codespaces
+ Update - tutorial content
+ Fix - `scipy.linalg` deprecation in latest release by adjusting version in `setup.py`
+ Update -  `pre_kappa` and `full_kappa` to integer to simplify equality comparisons
+ Update - `images` of the pipeline

## [0.1.1] - 2024-03-21

+ Update - Schemas and tables renaming
+ Update - Move `PreFit` and `FullFit` to `moseq_train`
+ Update - Additional attributes and data type modification from `time` to `float` for `duration` to eliminate datetime formatting code
+ Update - Code refactoring in `make` functions and enhanced path handling
+ Update - `docs`, docstrings and table definitions
+ Update - `tutorial.ipynb` according to these changes and verify full functionality with Codespaces
+ Update - pipeline `images` according to these changes
+ Fix - `Dockerfile` environment variables
+ Update - Activation of one schema with two modules by updating `tutorial_pipeline.ipynb`
+ Update - remove PyPI release from `release.yml`
+ Update - README

## [0.1.0] - 2024-03-20

+ Add - `CHANGELOG` and version for first release
+ Add - DevContainer configuration for GitHub Codespaces
+ Add - Updated documentation in `docs` for schemas and tutorial
+ Add - `kpms_reader` readers
+ Add - `element_moseq` pipeline architecture and design containing `kpms_pca` and `kpms_model` modules
+ Add - `images` with flowchart and pipeline images
+ Add - `tutorial.ipynb` consistent across DataJoint Elements that can be launched using GitHub Codespaces
+ Add - `tutorial_pipeline.py` script for notebooks to import and activate schemas
+ Add - spelling, markdown, and pre-commit config files
+ Add - GitHub Actions that call reusable workflows in the `datajoint/.github` repository
+ Add - `LICENSE`, `CONTRIBUTING`, `CODE_OF_CONDUCT`
+ Add - `README` consistent across DataJoint Elements
+ Add - `setup.py` with `extras_require` and `tests` features
