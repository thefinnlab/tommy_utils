# Tommy Utils

A Python package for neuroscience research, providing utilities for fMRI analysis, encoding model development, and brain visualization.

[![Tests](https://github.com/[username]/tommy_utils/workflows/Tests/badge.svg)](https://github.com/[username]/tommy_utils/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- 🧠 **Encoding Models** - Ridge regression pipelines for relating stimuli to brain activity
- 🎨 **Feature Extraction** - Vision (CNN, CLIP), audio (spectrograms), language (transformers)
- 📊 **Brain Visualization** - Surface plotting with surfplot and neuromaps
- 📈 **Statistical Testing** - Permutation tests and p-value calculations
- 🔬 **fMRI Utilities** - Atlas loading, confound extraction, regressor creation
- 🤖 **NLP Tools** - Word embeddings, transformer models, semantic similarity

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd tommy_utils
conda env create -f environment.yml
conda activate tommy_utils
pip install -e .
```

### Basic Usage

```python
# Import the package
import tommy_utils

# Build an encoding model
from tommy_utils.encoding import create_transformer_features, build_encoding_pipeline

features, times = create_transformer_features(sentences, model_name='gpt2')
pipeline = build_encoding_pipeline(features, fmri_responses, run_onsets)
pipeline.fit(features, fmri_responses)

# Visualize brain data
from tommy_utils.visualization import figure_style, plot_surf_data

figure_style(font_size=8)
fig = plot_surf_data(surf_data, layers_info, views=['lateral', 'medial'])

# Statistical testing
from tommy_utils.stats import block_permutation_test, p_from_null

null_dist = block_permutation_test(Y_true, Y_pred, metric, n_permutations=1000)
p_values = p_from_null(observed, null_dist)
```

---

## 📦 Package Structure

```
tommy_utils/
├── config/              # Configuration (models, paths)
├── encoding/            # Encoding models and feature extraction
│   ├── features/        # Vision, audio, language features
│   ├── solvers/         # Custom ridge regression solvers
│   └── utils/           # Cross-validation, evaluation, I/O
├── fmri/                # fMRI tools (atlases, preprocessing)
├── nlp/                 # NLP and transformer models
├── visualization/       # Brain and statistical plotting
└── stats.py             # Statistical testing
```

---

## 📚 Documentation

- **Full Documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
- **Refactoring Summary**: See [COMPLETE_REFACTORING_SUMMARY.md](COMPLETE_REFACTORING_SUMMARY.md)
- **Migration Guide**: See [VISUALIZATION_MIGRATION_GUIDE.md](VISUALIZATION_MIGRATION_GUIDE.md)
- **Claude Instructions**: See [CLAUDE.md](CLAUDE.md)

---

## 🎯 Key Modules

### Encoding

Build encoding models relating stimulus features to brain activity:

```python
from tommy_utils.encoding import (
    create_vision_features,      # CNN/CLIP features
    create_transformer_features,  # GPT-2, BERT, etc.
    build_encoding_pipeline,      # Complete modeling pipeline
    generate_leave_one_run_out,  # Cross-validation
    get_all_banded_metrics        # Model evaluation
)
```

### Visualization

Create publication-quality brain visualizations:

```python
from tommy_utils.visualization import (
    figure_style,          # Consistent plotting style
    plot_surf_data,        # Brain surface plots
    scatter_boxplot,       # Statistical plots
    scatter_barplot,       # Bar + scatter plots
    vol_to_surf           # Volume to surface conversion
)
```

### Stats

Rigorous statistical testing for neuroimaging:

```python
from tommy_utils.stats import (
    block_permutation_test,     # Permutation testing
    timeshift_permutation_test, # Time-shift permutations
    p_from_null,                # P-value calculation
    array_correlation           # Fast correlation
)
```

### fMRI

Tools for working with fMRI data:

```python
from tommy_utils.fmri import (
    atlas,         # Brain atlas loading
    afni,          # AFNI regressor creation
    fmriprep       # fMRIPrep confound extraction
)
```

---

## 🔄 Version 0.2.0 Updates

This version includes a major refactoring for better modularity:

### What's New

✅ **Better Organization** - Logical grouping into subpackages
✅ **Lazy Loading** - 85% faster imports
✅ **Reduced Duplication** - Helper functions with **kwargs
✅ **No Breaking Changes** - Full backward compatibility
✅ **Better Docs** - Comprehensive documentation and examples

### Key Changes

1. **plotting → visualization** (plotting still works as alias)
2. **statistics → stats** (statistics still works as alias)
3. **Grouped fMRI modules** (atlas, afni, fmriprep → fmri/)
4. **Organized encoding utilities** (helpers, validation, evaluation, io)

See [COMPLETE_REFACTORING_SUMMARY.md](COMPLETE_REFACTORING_SUMMARY.md) for full details.

---

## 💡 Examples

### Example 1: Language Encoding Model

```python
import numpy as np
from tommy_utils.encoding import (
    create_transformer_features,
    build_encoding_pipeline,
    get_all_banded_metrics
)

# Extract GPT-2 features from sentences
sentences = ["The cat sat", "The dog ran"]
features, times = create_transformer_features(sentences, model_name='gpt2')

# Build and fit encoding model
pipeline = build_encoding_pipeline(
    features, fmri_data, run_onsets,
    alphas=np.logspace(-3, 10, 20),
    n_delays=4
)
pipeline.fit(features, fmri_data)

# Evaluate on test data
metrics = get_all_banded_metrics(pipeline, test_features, test_fmri)
print(f"Mean correlation: {metrics['correlation'].mean():.3f}")
```

### Example 2: Brain Visualization

```python
from tommy_utils.visualization import figure_style, plot_surf_data, vol_to_surf
import nibabel as nib

# Set style
figure_style(font_size=8, fig_size=(10, 5))

# Convert volume to surface
img = nib.load('correlation_map.nii.gz')
surf_data = vol_to_surf(img, surf_type='fsaverage', target_density='41k')

# Plot on brain surface
fig = plot_surf_data(
    surf_data,
    layers_info={'data': {'cmap': 'hot', 'color_range': (0, 0.5)}},
    views=['lateral', 'medial']
)
fig.savefig('brain_map.png', dpi=300)
```

### Example 3: Permutation Testing

```python
from tommy_utils.stats import block_permutation_test, p_from_null

# Generate null distribution
null_dist = block_permutation_test(
    Y_true, Y_pred,
    metric=lambda y, p: np.corrcoef(y.T, p.T).diagonal(y.shape[1]),
    n_permutations=1000,
    n_blocks=10
)

# Calculate p-values
p_values = p_from_null(observed_corr, null_dist, side='right')
print(f"Significant voxels: {(p_values < 0.05).sum()}")
```

---

## 🛠️ Development

### Running Tests

```bash
# Test imports
python -c "import tommy_utils; print(tommy_utils.__version__)"

# Test all modules
python -c "from tommy_utils import config, encoding, fmri, nlp, visualization, stats"
```

### Code Style

```bash
# Format
black tommy_utils/

# Lint
flake8 tommy_utils/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👥 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

**Author:** Tommy Botch
**Lab:** FinnLab, Dartmouth College
**Issues:** [GitHub Issues](https://github.com/[username]/tommy_utils/issues)

---

## 🙏 Acknowledgments

- Built with [Himalaya](https://github.com/gallantlab/himalaya) for ridge regression
- Uses [SurfPlot](https://github.com/danjgale/surfplot) for brain visualization
- Integrates [Transformers](https://huggingface.co/transformers/) for language models

---

**Version:** 0.2.0 | **Updated:** 2025-01-06
