import os
import sys
from setuptools import setup, find_packages

def read(rel_path):
    """Read a file relative to setup.py directory."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    """Extract version string from __init__.py file."""
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# Get version
version = get_version("tommy_utils/__init__.py")
print(f"Loaded version {version}")

# Check Python version
if sys.version_info[:2] < (3, 10):
    raise RuntimeError("Python version >=3.10 required.")

# Read README
readme = read("README.md")

# Core dependencies from environment.yml
# Note: Full dependency list is managed in environment.yml
# These are the essential packages required for installation
core_dependencies = [
    'numpy<2.0.0',
    'pandas',
    'matplotlib',
    'seaborn',
    'scipy',
    'scikit-learn',
    'nibabel',
    'nilearn',
    'torch',
    'transformers>=4.51.0',
    'huggingface_hub',
    'gensim',
    'nltk',
    'himalaya',
    'neuromaps',
    'tqdm',
    'regex',
    'einops',
]

setup(
    name="tommy_utils",
    version=version,
    description="Utilities for neuroscience research: fMRI analysis, encoding models, and brain visualization",
    author="Tommy Botch, FinnLab, Dartmouth College",
    author_email="tlb.gr@dartmouth.edu",
    url="https://github.com/thefinnlab/tommy_utils",
    license="GNU General Public License Version 2",
    long_description=readme,
    long_description_content_type="text/markdown",

    # Package configuration
    packages=find_packages(),
    package_data={
        'tommy_utils': [
            'data/**/*',  # Include all data files
            'data/nlp/*.txt',
            'data/atlases/**/*',
        ],
    },
    include_package_data=True,

    # Dependencies
    install_requires=core_dependencies + [
        # Git dependencies for custom forks
        'brainspace @ git+https://github.com/thefinnlab/BrainSpace.git',
        'surfplot @ git+https://github.com/thefinnlab/surfplot.git',
    ],

    # Python version requirement
    python_requires=">=3.10",

    # Optional dependencies for development
    extras_require={
        'dev': [
            'flake8>=6.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'pylint>=2.17.0',
            'pytest>=7.3.0',
            'pytest-cov>=4.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.2.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
        ],
        'all': [
            'flake8>=6.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'pylint>=2.17.0',
            'pytest>=7.3.0',
            'pytest-cov>=4.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.2.0',
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],

    # Keywords for discoverability
    keywords=[
        "neuroscience",
        "fMRI",
        "encoding models",
        "brain visualization",
        "neuroimaging",
        "machine learning",
    ],
)

# Note: For full environment setup including optional dependencies,
# use: conda env create -f environment.yml
print(f"\nInstallation complete for tommy_utils v{version}")
print("For full environment setup with all dependencies:")
print("  conda env create -f environment.yml")
