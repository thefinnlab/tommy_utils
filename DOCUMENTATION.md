# Tommy Utils Documentation

**Version:** 0.2.0
**Author:** Tommy Botch (FinnLab, Dartmouth College)
**License:** MIT

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Package Structure](#package-structure)
4. [Module Reference](#module-reference)
   - [Config](#config)
   - [Encoding](#encoding)
   - [fMRI](#fmri)
   - [NLP](#nlp)
   - [Visualization](#visualization)
   - [Stats](#stats)
5. [Quick Start Examples](#quick-start-examples)
6. [Migration Guide](#migration-guide)
7. [Development](#development)
8. [FAQ](#faq)

---

## Overview

`tommy_utils` is a Python package providing utilities for neuroscience research, specifically focused on:

- **Encoding models** for fMRI analysis
- **Feature extraction** from vision, audio, and language stimuli
- **Brain visualization** on cortical surfaces
- **Statistical testing** with permutation methods
- **fMRI preprocessing** utilities
- **NLP and transformer models** for language analysis

### Key Features

✨ **Modular Design** - Organized into domain-specific subpackages
⚡ **Lazy Loading** - Fast imports with deferred dependency loading
🔄 **Backward Compatible** - All existing code continues to work
📊 **Rich Visualization** - Brain surface plotting and statistical charts
🧪 **Statistical Testing** - Permutation tests and p-value calculations
🤖 **Modern ML** - Integration with PyTorch, Transformers, and Himalaya

---

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd tommy_utils

# Create conda environment
conda env create -f environment.yml
conda activate tommy_utils

# Install in development mode
pip install -e .
```

### Dependencies

Key dependencies (see `environment.yml` for complete list):
- Python 3.9+
- PyTorch (deep learning)
- Nibabel, Nilearn (neuroimaging)
- Himalaya (ridge regression)
- Transformers, Gensim (NLP)
- Seaborn, Matplotlib (visualization)
- SurfPlot, Neuromaps (surface plotting)

---

## Package Structure

```
tommy_utils/
├── config/              # Centralized configuration
│   ├── models.py        # Model dictionaries and feature definitions
│   └── paths.py         # Path utilities for data files
│
├── encoding/            # Encoding model infrastructure
│   ├── features/        # Feature extraction (lazy loading)
│   │   ├── language.py  # Phoneme, word, transformer features
│   │   ├── vision.py    # CNN and CLIP vision features
│   │   └── audio.py     # Spectral and mel-spectrogram features
│   ├── solvers/         # Custom Himalaya solvers
│   │   └── custom_solvers.py
│   ├── utils/           # Encoding utilities
│   │   ├── helpers.py   # General utility functions
│   │   ├── validation.py # Cross-validation strategies
│   │   ├── evaluation.py # Model metrics
│   │   └── io.py        # Model serialization
│   ├── delayer.py       # HRF delay modeling
│   └── pipeline.py      # Model building pipelines
│
├── fmri/                # fMRI-specific tools
│   ├── atlas.py         # Brain atlas loading and manipulation
│   ├── afni.py          # AFNI regressor creation
│   └── fmriprep.py      # fMRIPrep confound extraction
│
├── nlp/                 # Natural language processing
│   └── nlp_legacy.py    # Word embeddings and transformers
│
├── visualization/       # Brain and statistical visualization
│   ├── brain.py         # Brain surface and volume plotting
│   ├── stats.py         # Statistical plots (boxplots, barplots)
│   └── style.py         # Figure styling utilities
│
├── stats.py             # Statistical testing for neuroimaging
├── plotting.py          # Backward compatibility (deprecated)
└── misc.py              # Miscellaneous utilities
```

---

## Module Reference

### Config

Centralized configuration for models and data paths.

#### config.models

```python
from tommy_utils.config.models import ENCODING_FEATURES, CLM_MODELS_DICT

# Available feature extractors by modality
print(ENCODING_FEATURES['visual'])  # ['alexnet', 'clip']
print(ENCODING_FEATURES['audio'])   # ['spectral', 'phoneme', ...]
print(ENCODING_FEATURES['language']) # ['phoneme', 'word2vec', 'gpt2', ...]

# Causal language model configurations
print(CLM_MODELS_DICT.keys())  # GPT-2, GPT-J, LLaMA, etc.
```

#### config.paths

```python
from tommy_utils.config.paths import get_data_dir, get_phonemes_path

data_dir = get_data_dir()  # Path to package data directory
phonemes = get_phonemes_path()  # Path to CMU phoneme dictionary
```

---

### Encoding

Encoding model infrastructure for relating stimulus features to brain activity.

#### Feature Extraction

**Vision Features:**
```python
from tommy_utils.encoding.features import create_vision_features

# Extract features from images using AlexNet or CLIP
features = create_vision_features(
    images,  # PIL images or tensors
    model_name='alexnet',  # or 'clip'
    batch_size=8
)
```

**Audio Features:**
```python
from tommy_utils.encoding.features import create_spectral_features

# Create spectrograms from audio
features = create_spectral_features(
    audio_data,
    sample_rate=16000,
    n_fft=1024
)
```

**Language Features:**
```python
from tommy_utils.encoding.features import (
    create_phoneme_features,
    create_word_features,
    create_transformer_features
)

# Phoneme features
phoneme_features = create_phoneme_features(transcript_words)

# Word embeddings (GloVe, Word2Vec, etc.)
word_features = create_word_features(
    transcript_words,
    model_name='word2vec',
    embedding_size=300
)

# Transformer features (GPT-2, BERT, etc.)
transformer_features = create_transformer_features(
    sentences,
    model_name='gpt2',
    layer=-1  # Last layer
)
```

#### Model Building

```python
from tommy_utils.encoding import build_encoding_pipeline

# Build complete encoding pipeline
pipeline = build_encoding_pipeline(
    features,
    Y_train,
    run_onsets,
    alphas=np.logspace(-3, 10, 20),
    n_delays=4,
    solver='group_level_random_search'
)

# Fit the model
pipeline.fit(features, Y_train)

# Make predictions
predictions = pipeline.predict(features_test)
```

#### Cross-Validation

```python
from tommy_utils.encoding import generate_leave_one_run_out

# Generate leave-one-run-out splits
for train_idx, val_idx in generate_leave_one_run_out(n_samples, run_onsets):
    X_train, X_val = features[train_idx], features[val_idx]
    Y_train, Y_val = responses[train_idx], responses[val_idx]
    # Train and evaluate model
```

#### Model Evaluation

```python
from tommy_utils.encoding import get_all_banded_metrics

# Compute comprehensive metrics
metrics = get_all_banded_metrics(
    pipeline,
    X_test,
    Y_test,
    use_split=False
)

print(metrics['correlation'])  # Pearson correlations
print(metrics['r2'])            # R² scores
print(metrics['predictions'])   # Model predictions
```

#### Model I/O

```python
from tommy_utils.encoding import save_model_parameters, load_model_from_parameters

# Save trained model
model_dict = save_model_parameters(pipeline)
np.save('model_params.npy', model_dict)

# Load model
model_dict = np.load('model_params.npy', allow_pickle=True).item()
new_pipeline = load_model_from_parameters(model_dict)
```

---

### fMRI

Tools for working with fMRI data and preprocessing.

#### Atlas Loading

```python
from tommy_utils.fmri.atlas import (
    load_fedorenko_atlas,
    load_glasser_atlas,
    load_visual_rois,
    load_combined_atlas,
    data_to_parcel
)

# Load Fedorenko language network
lang_atlas = load_fedorenko_atlas(
    network='language',
    hemi='both',
    space='MNI152'
)

# Load Glasser HCP parcellation
glasser = load_glasser_atlas(hemi='L')

# Load visual stream ROIs
visual_atlas = load_visual_rois(
    atlas_type='nsd_streams',
    hemi='both'
)

# Combine multiple atlases with priority
combined = load_combined_atlas(
    atlases=['fedorenko', 'glasser', 'nsd_streams'],
    priority_order=['fedorenko', 'glasser', 'nsd_streams']
)

# Convert voxel data to parcel averages
parcel_data = data_to_parcel(voxel_data, atlas_img)
```

#### fMRIPrep Confounds

```python
from tommy_utils.fmri.fmriprep import get_fmriprep_confounds

# Extract confounds from fMRIPrep output
confounds = get_fmriprep_confounds(
    confounds_tsv,
    n_compcor=5,
    include_motion=True,
    include_outliers=True
)
```

#### AFNI Regressors

```python
from tommy_utils.fmri.afni import (
    create_amplitude_modulated_regressor,
    create_duration_modulated_regressor
)

# Create amplitude-modulated regressor
regressor = create_amplitude_modulated_regressor(
    onsets,
    amplitudes,
    tr=2.0,
    duration=300.0
)
```

---

### NLP

Natural language processing and transformer models.

```python
from tommy_utils import nlp

# Load word embeddings
word2vec = nlp.load_word_model('word2vec')
embedding = word2vec['king']  # Get word vector

# Load transformer models
model, tokenizer = nlp.load_clm_model('gpt2')
model.eval()

# Get contextualized embeddings
embeddings = nlp.get_contextualized_embeddings(
    sentences,
    model,
    tokenizer,
    layer=-1
)

# Calculate word probabilities
probs = nlp.get_word_probabilities(
    sentences,
    model,
    tokenizer
)

# Semantic similarity
similarity = nlp.calculate_semantic_similarity(
    word1_embedding,
    word2_embedding,
    metric='cosine'
)
```

---

### Visualization

Brain visualization and statistical plotting.

#### Figure Styling

```python
from tommy_utils.visualization import figure_style

# Set consistent plotting style
figure_style(
    font_size=7,
    scatter_size=10,
    axes_color='black',
    font='Liberation Sans',
    fig_size=(4, 4)
)
```

#### Brain Visualization

```python
from tommy_utils.visualization import (
    plot_surf_data,
    vol_to_surf,
    make_layers_dict,
    create_depth_map
)

# Convert volume to surface
surf_data = vol_to_surf(
    nifti_img,
    surf_type='fsaverage',
    target_density='41k'
)

# Create visualization layers
layers = make_layers_dict(
    surf_data,
    cmap='RdBu_r',
    alpha=0.75,
    color_range=(-3, 3)
)

# Create depth map for shading
depth_map = create_depth_map(
    surf_type='fsaverage',
    target_density='41k'
)

# Plot on brain surface
fig = plot_surf_data(
    surfs={'left': surf_data['left'], 'right': surf_data['right']},
    layers_info={'data': layers},
    surf_type='fsaverage',
    views=['lateral', 'medial'],
    depth_map=depth_map
)
```

#### Statistical Plots

```python
from tommy_utils.visualization import (
    scatter_boxplot,
    scatter_barplot,
    kde_boxplot
)

# Scatter + boxplot
import pandas as pd
fig, ax = scatter_boxplot(
    df,
    x='condition',
    y='accuracy',
    group='subject',
    palette='RdBu_r'
)

# Scatter + barplot with error bars
fig, ax = scatter_barplot(
    df,
    x='condition',
    y='accuracy',
    group='subject',
    ci=95,
    plot_points=True
)

# KDE + boxplot
fig, ax = kde_boxplot(
    df,
    x='condition',
    y='accuracy',
    palette='viridis'
)
```

---

### Stats

Statistical testing for neuroimaging analysis.

```python
from tommy_utils.stats import (
    block_permutation_test,
    timeshift_permutation_test,
    p_from_null,
    array_correlation
)

# Block permutation test
null_dist = block_permutation_test(
    Y_true,
    Y_pred,
    metric=lambda y, p: np.corrcoef(y.T, p.T)[0, 1],
    n_permutations=1000,
    n_blocks=10
)

# Calculate p-values
p_values = p_from_null(
    observed_correlations,
    null_dist,
    side='two-sided'
)

# Array-wise correlation (fast)
correlations = array_correlation(
    Y_true,
    Y_pred,
    axis=0
)

# Fisher z-transformation for averaging correlations
from tommy_utils.stats import ztransform_mean
avg_correlation = ztransform_mean([corr1, corr2, corr3])
```

---

## Quick Start Examples

### Example 1: Build an Encoding Model

```python
import numpy as np
from tommy_utils.encoding import (
    create_transformer_features,
    build_encoding_pipeline,
    generate_leave_one_run_out,
    get_all_banded_metrics
)

# 1. Extract features from stimuli
sentences = ["The cat sat on the mat", "The dog ran in the park"]
features, times = create_transformer_features(
    sentences,
    model_name='gpt2',
    layer=-1
)

# 2. Build encoding pipeline
pipeline = build_encoding_pipeline(
    features,
    fmri_responses,
    run_onsets=[0, 100, 200],
    alphas=np.logspace(-3, 10, 20),
    n_delays=4
)

# 3. Cross-validation
for train_idx, val_idx in generate_leave_one_run_out(n_samples, run_onsets):
    pipeline.fit(features[train_idx], fmri_responses[train_idx])
    metrics = get_all_banded_metrics(
        pipeline,
        features[val_idx],
        fmri_responses[val_idx]
    )
    print(f"Validation correlation: {metrics['correlation'].mean():.3f}")
```

### Example 2: Visualize Brain Maps

```python
import nibabel as nib
from tommy_utils.visualization import (
    figure_style,
    vol_to_surf,
    plot_surf_data,
    make_layers_dict
)

# Set plotting style
figure_style(font_size=8, fig_size=(10, 5))

# Load and convert data
nifti_img = nib.load('correlation_map.nii.gz')
surf_data = vol_to_surf(nifti_img, surf_type='fsaverage', target_density='41k')

# Create layers
layers = make_layers_dict(
    surf_data,
    cmap='hot',
    color_range=(0, 0.5)
)

# Plot
fig = plot_surf_data(
    surf_data,
    {'correlations': layers},
    views=['lateral', 'medial', 'ventral'],
    size=(1200, 400)
)
fig.savefig('brain_map.png', dpi=300)
```

### Example 3: Statistical Testing

```python
from tommy_utils.stats import block_permutation_test, p_from_null
import numpy as np

# Observed correlations
observed = np.corrcoef(Y_true.T, Y_pred.T).diagonal(Y_true.shape[1])

# Generate null distribution
null_dist = block_permutation_test(
    Y_true,
    Y_pred,
    metric=lambda y, p: np.corrcoef(y.T, p.T).diagonal(y.shape[1]),
    n_permutations=1000,
    n_blocks=10
)

# Calculate p-values
p_values = p_from_null(observed, null_dist, side='right')

# Apply FDR correction
from statsmodels.stats.multitest import multipletests
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

print(f"Significant voxels: {reject.sum()}/{len(reject)}")
```

---

## Migration Guide

### From v0.1.0 to v0.2.0

**Good news:** All old code continues to work! But we recommend migrating to the new structure.

#### Module Imports

**Old (still works):**
```python
from tommy_utils.plotting import figure_style, scatter_boxplot
from tommy_utils.atlas import load_fedorenko_atlas
from tommy_utils.statistics import p_from_null
```

**New (recommended):**
```python
from tommy_utils.visualization import figure_style, scatter_boxplot
from tommy_utils.fmri.atlas import load_fedorenko_atlas
from tommy_utils.stats import p_from_null
```

#### Key Changes

1. **plotting → visualization**
   - Old: `from tommy_utils.plotting import *`
   - New: `from tommy_utils.visualization import *`

2. **statistics → stats**
   - Old: `from tommy_utils.statistics import *`
   - New: `from tommy_utils.stats import *`
   - Note: `tommy_utils.statistics` is now an alias to `stats`

3. **fMRI modules grouped**
   - Old: `from tommy_utils.atlas import *`
   - New: `from tommy_utils.fmri.atlas import *`
   - Same for `afni` and `fmriprep`

4. **Encoding utils organized**
   - Functions still accessible at top level: `from tommy_utils.encoding import get_modality_features`
   - Also available in submodules: `from tommy_utils.encoding.utils import get_modality_features`

---

## Development

### Running Tests

```bash
# Import tests
python -c "import tommy_utils; print('✓ Package imports successfully')"

# Run all module imports
python -c "
from tommy_utils import config, encoding, fmri, nlp, visualization, stats
print('✓ All modules imported')
"
```

### Code Style

This project follows PEP 8 conventions:

```bash
# Format code
black tommy_utils/

# Sort imports
isort tommy_utils/

# Lint
flake8 tommy_utils/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## FAQ

### Q: Why was `statistics` renamed to `stats`?

**A:** Python has a built-in module called `statistics` that is used by many packages (including seaborn). Having a subpackage with the same name caused import conflicts. The new name `stats` avoids this issue entirely.

### Q: Will my old code break?

**A:** No! We maintain backward compatibility. `tommy_utils.statistics` is now an alias for `tommy_utils.stats`, and the old `plotting` module still works by re-exporting from `visualization`.

### Q: How do I know which encoding model to use?

**A:** Use `BandedRidgeCV` (default) when n_samples > n_features. It automatically switches to `MultipleKernelRidgeCV` when n_samples < n_features (high-dimensional data).

### Q: What's the difference between banded ridge and kernel ridge?

**A:**
- **Banded Ridge**: Solves in primal space, faster for n_samples > n_features
- **Kernel Ridge**: Solves in dual space, more efficient for n_features > n_samples

### Q: Which surface space should I use for visualization?

**A:**
- `fsaverage`: Most common, FreeSurfer standard
- `fslr`: HCP standard, better for subcortical structures
- `civet`: CIVET pipeline output

### Q: How do I cite this package?

```bibtex
@software{tommy_utils,
  author = {Botch, Tommy},
  title = {tommy_utils: Utilities for Neuroscience Research},
  year = {2025},
  version = {0.2.0},
  institution = {FinnLab, Dartmouth College}
}
```

---

## Contact

**Author:** Tommy Botch
**Lab:** FinnLab, Dartmouth College
**Email:** [contact info]
**Issues:** [GitHub Issues URL]

---

## License

MIT License - see LICENSE file for details.

---

**Last Updated:** 2025-01-06
**Documentation Version:** 0.2.0
