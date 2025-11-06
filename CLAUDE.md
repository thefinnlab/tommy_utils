# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`tommy_utils` is a Python utility package for neuroscience research, specifically focused on fMRI analysis and encoding model development. This package is maintained by Tommy Botch at the FinnLab, Dartmouth College.

## Environment Setup

**Install the package:**
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate tommy_utils

# Install the package in development mode
pip install -e .
```

**Key dependencies:**
- Python 3.9+
- PyTorch (deep learning)
- Nibabel, Nilearn (neuroimaging)
- Himalaya (ridge regression for neuroimaging)
- Transformers, Gensim (NLP models)
- Seaborn, Matplotlib (visualization)
- See `environment.yml` for complete list

## Architecture

### Module Organization

**Note:** The package has been refactored for better modularity. See `REFACTORING_SUMMARY.md` for details.

The package is organized into domain-specific modules:

1. **`config/`** - Centralized configuration ⭐ NEW
   - Model dictionaries (ENCODING_FEATURES, CLM_MODELS_DICT, etc.)
   - Path utilities (get_data_dir(), get_phonemes_path(), etc.)

2. **`encoding/`** - Core encoding model infrastructure ⭐ REFACTORED
   - **`features/`** - Feature extraction subpackage
     - `language.py` - Phoneme, word, and transformer features
     - `vision.py` - CNN and CLIP vision features
     - `audio.py` - Spectral and mel-spectrogram features
   - **`pipeline.py`** - Model building with optimized helper functions
   - **`crossval.py`** - Cross-validation strategies (leave-one-run-out)
   - **`evaluation.py`** - Model metrics and scoring
   - **`io.py`** - Save/load model parameters
   - **`utils.py`** - Helper functions (Lanczos interpolation, data handling)
   - Ridge regression and kernel ridge regression pipelines using Himalaya
   - Custom group-level solvers for multi-subject modeling
   - HRF delay modeling via the `Delayer` class

3. **`nlp.py`** - Natural language processing utilities
   - Loading and managing word embeddings (GloVe, Word2Vec, FastText)
   - Transformer model interfaces (GPT-2, BERT, RoBERTa, LLaMA, etc.)
   - Contextualized word embedding extraction
   - Word prediction and probability metrics
   - Semantic similarity calculations

4. **`plotting.py`** - Visualization for neuroimaging and statistics
   - Brain surface plotting using `surfplot` and `neuromaps`
   - Volume-to-surface transformations (MNI152 to fsaverage/fslr/civet)
   - Statistical plots (scatter-boxplots, scatter-barplots, KDE-boxplots)
   - Correlation matrices and brain map visualizations
   - Depth map creation for cortical surfaces

5. **`statistics.py`** - Statistical testing for neuroimaging
   - Permutation testing (block permutation, timeshift permutation)
   - P-value calculation from null distributions
   - Array-wise correlation (from BrainIAK)
   - Multiple comparison correction

6. **`fmriprep.py`** - fMRIPrep confound extraction
   - CompCor component extraction (aCompCor, tCompCor)
   - Flexible confound selection from fMRIPrep outputs

7. **`afni.py`** - AFNI regressor creation
   - Amplitude-modulated regressors
   - Duration-modulated regressors

8. **`atlas.py`** - Brain atlas utilities
   - Loading and combining brain atlases (Fedorenko, Glasser, visual ROIs)
   - Atlas manipulation (masking, parcellation, region extraction)
   - Converting data to parcels using atlas definitions
   - Handling overlapping regions with priority-based combination

9. **`custom_solvers.py`** - Custom Himalaya solvers
   - Group-level banded ridge regression
   - Group-level multiple kernel ridge regression

10. **`delayer.py`** - HRF delay modeling
   - Creates time-lagged features for encoding models

11. **`misc.py`** - Miscellaneous utilities

## Refactoring Benefits

### Lazy Loading
The refactored `encoding` module uses lazy imports to dramatically improve import times:
```python
import tommy_utils  # Fast - doesn't load torch/transformers yet
from tommy_utils.encoding.features import create_vision_features  # Loads torch only when called
```

### Helper Functions
The `pipeline.py` module now uses helper functions with **kwargs to eliminate duplication:
- `_create_base_solver_params()` - Creates solver parameter dictionaries
- `_validate_group_level_shapes()` - Validates group-level modeling inputs
- `_create_ridge_model()` - Unified model instantiation with flexible kwargs

### Modular Structure
Each component has a clear responsibility:
- `features/` - Feature extraction only
- `pipeline.py` - Model building only
- `evaluation.py` - Metrics only
- `io.py` - Serialization only

## Key Workflows

### Encoding Model Pipeline

The typical encoding model workflow:

1. **Feature Extraction** - Use functions in `encoding.py` to extract features:
   - Vision: `create_vision_features()` for video/image features
   - Audio: `create_spectral_features()` for audio spectrograms
   - Language: `create_transformer_features()` or `create_word_features()` for text

2. **Feature Organization** - Use `create_banded_features()` to organize multiple feature spaces for banded ridge regression

3. **Cross-Validation Setup** - Use `generate_leave_one_run_out()` to create CV folds that respect run structure

4. **Model Building** - Use `build_encoding_pipeline()` to construct a complete pipeline with:
   - StandardScaler (mean-centering)
   - Delayer (HRF modeling)
   - BandedRidgeCV or MultipleKernelRidgeCV (depending on feature/sample ratio)

5. **Model Evaluation** - Use `get_all_banded_metrics()` to compute correlations, R², and residuals

6. **Parameter Saving** - Use `save_model_parameters()` to serialize trained models

### Brain Visualization

For visualizing brain maps:

1. **Volume to Surface** - Use `vol_to_surf()` or `numpy_to_surface()` to transform volumetric data
2. **Layer Creation** - Use `make_layers_dict()` to organize data, colormaps, and transparency
3. **Plotting** - Use `plot_surf_data()` with optional depth maps for publication-quality figures

### Statistical Testing

For significance testing:

1. Create null distribution using `block_permutation_test()` or `timeshift_permutation_test()`
2. Calculate p-values using `p_from_null()`
3. Apply multiple comparison correction as needed

### Atlas-Based Analysis

For working with brain atlases:

1. **Load Individual Atlases** - Use atlas-specific loaders:
   - `load_fedorenko_atlas()` for language, ToM, or MD networks
   - `load_glasser_atlas()` for HCP multi-modal parcellation
   - `load_visual_rois()` for NSD visual streams or Kastner2015 atlases

2. **Combine Multiple Atlases** - Use `load_combined_atlas()` to merge atlases with priority ordering

3. **Convert to Parcels** - Use `data_to_parcel()` to aggregate voxel/vertex-wise data into parcels

4. **Create Masks** - Use `create_atlas_mask()` to extract specific regions of interest

## Important Implementation Details

### Himalaya Integration

This package extends Himalaya with custom solvers:
- `BandedRidgeCV.ALL_SOLVERS['group_level_random_search']` - registered in `encoding.py:48`
- `MultipleKernelRidgeCV.ALL_SOLVERS['group_level_random_search']` - registered in `encoding.py:49`

These enable group-level modeling where all subjects share the same features but have different brain responses.

### Backend Management

Himalaya uses configurable backends (NumPy, PyTorch, CuPy). The `get_backend()` function determines the current backend. When working with model outputs, always use `backend.to_cpu()` and `backend.asarray_like()` for type consistency.

### Feature Space Configuration

When using multiple feature spaces (e.g., visual + audio + language):
- Features are concatenated horizontally
- Each feature space gets its own regularization parameter
- Use tuples of `(name, preprocessing_pipeline, slice)` to configure `ColumnTransformerNoStack`

### Model Selection Logic

In `build_encoding_pipeline()`:
- If `n_samples > n_features`: Use BandedRidgeCV (more efficient)
- If `n_samples < n_features`: Use MultipleKernelRidgeCV (kernelized)
- Can force BandedRidge with `force_banded_ridge=True`

### Surface Mesh Densities

When working with surface data, be aware of valid densities:
- fsaverage: '3k', '10k', '41k', '164k'
- fslr: '4k', '8k', '32k', '164k'
- civet: '41k', '164k'

### Data Directory Structure

The package includes data files organized by module in `tommy_utils/data/`:
- `data/nlp/` - NLP-related data (CMU phoneme dictionary)
- `data/atlases/` - Brain atlas files:
  - `fedorenko/language/` - Language network atlas files
  - `fedorenko/MD/` - Multiple demand network atlas files
  - `fedorenko/ToM/` - Theory of mind network atlas files
  - `glasser/` - HCP multi-modal parcellation atlas
  - `nsd_streams/` - NSD visual streams atlas
  - `Kastner2015/` - Kastner 2015 visual atlas

Atlas functions in `atlas.py` automatically use the packaged data directory, but you can override this by passing a custom `data_dir` parameter.

## Development Practices

### Modifying Encoding Pipelines

When adding new feature extraction methods:
1. Add the model name to `ENCODING_FEATURES` dict in `encoding.py`
2. Create a feature extraction function following the pattern `create_*_features()`
3. Return `(times, features)` where times are onset timestamps

### Adding New Solvers

To add a custom solver to Himalaya:
1. Define the solver function following Himalaya's interface
2. Register it: `BandedRidgeCV.ALL_SOLVERS['your_solver_name'] = your_solver_function`

### Working with Transformers

The package supports both causal LM (GPT-style) and masked LM (BERT-style) models:
- CLM models: Use `load_clm_model()` - models defined in `CLM_MODELS_DICT`
- MLM models: Use `load_mlm_model()` - models defined in `MLM_MODELS_DICT`
- Always call `model.eval()` before inference

## Testing

This package does not currently include a formal test suite. When making changes, verify functionality by:
- Running example encoding pipelines on small datasets
- Checking output shapes match expected dimensions
- Verifying brain visualizations render correctly
