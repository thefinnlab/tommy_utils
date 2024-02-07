import os
import sys
from setuptools import setup, find_packages

# install_requires = [
#     "pytorch",
#     "nibabel",
#     "nilearn",
#     "seaborn>=0.13.0",
#     "gensim",
#     "nltk",
#     "numpy>=1.23.0",
#     "librosa",
#     "pandas>=1.2.2",
#     "matplotlib",
#     "regex",
#     "moviepy",
#     "mesalib",
#     "libiconv",
#     "vtk",
#     "joypy",
#     "statsmodels",
#     "transformers>=4.35.0",
#     "huggingface_hub",
#     "scikit-learn",
#     "scipy",
#     "tqdm",
#     "einops",
#     "praatio",
#     "torchaudio",
#     "torchvision",
#     "av",
#     "torchlens",
#     "neuromaps",
#     "brainspace",
#     "mergedeep",
#     "himalaya",
#     "pydub",
#     "tensorflow",
#     "natsort",
#     "sentencepiece",
#     "transformers_stream_generator",
#     "fasttext-wheel"
# ]

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

version = get_version("tommy_utils/__init__.py")
print("Loaded version ",version)

if sys.version_info[:2] < (3, 9):
    raise RuntimeError("Python version >=3.9 required.")

readme = open("README.md").read()
setup(
    name="tommy_utils",
    version=version,
    description="tommy_utils",
    author="Tommy Botch, FinnLab, Dartmouth College",
    author_email="tlb.gr@dartmouth.edu",
    packages=find_packages(),
    package_data={
      'tommy_utils': ['./tommy_utils/data/*.txt'],
      },
    python_requires=">=3.9",
    license="GNU General Public License Version 2",
    # install_requires=install_requires,
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/thefinnlab/tommy_utils",
)

# # get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))