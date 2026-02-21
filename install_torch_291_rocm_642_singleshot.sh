# Complete, reproducible script to build and prepare environment
# Exit immediately if a command exits with a non-zero status
set -e

REPO=$(pwd)

# modify the installation path and env name if you want. Assumes $WRKSPC is set.
INSTALLDIR=${WRKSPC}
ENV_NAME="torch_291_rocm_642_singleshot"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.
# Steps below assume that conda command is available and a base environment is active.

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.12 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# HPC cluster specific, remove if not relevant.
# Conda packages:
conda install -c conda-forge libstdcxx-ng --yes

# Load modules
rocm_version=6.4.2

module load rocm/$rocm_version

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
cd ${INSTALLDIR}
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install torch==2.9.1+rocm6.4 torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/rocm6.4
pip install ninja packaging numpy

cd "${REPO}"
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install -e '.[all]'
cd ${INSTALLDIR}

# extras
pip install transformers hf_transfer litdata datasets matplotlib torchdata tenacity ipykernel jupyterlab
pip install wandb==0.23.0 # 0.23.1 seemingly broke the tables feat we rely on

# attn gym
cd ${INSTALLDIR}
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym
pip install .
cd ${INSTALLDIR}

# lm-evaluation-harness
cd ${INSTALLDIR}
git clone git@github.com:jwkirchenbauer/lm-evaluation-harness-mtp-lm.git
cd lm-evaluation-harness-mtp-lm
pip install -e .
cd ${INSTALLDIR}

# amdsmi
cd ${INSTALLDIR}
cp -R /opt/rocm-${rocm_version}/share/amd_smi/ $WRKSPC/amd_smi_${rocm_version}
cd $WRKSPC/amd_smi_${rocm_version}
pip install .
cd ${INSTALLDIR}
