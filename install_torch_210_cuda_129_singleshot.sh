# Complete, reproducible script to build and prepare environment
# Exit immediately if a command exits with a non-zero status
set -e

# specific to the HPC cluster, rm or modify as needed
#module load aws-ofi-nccl cuda nccl libfabric

REPO=$(pwd)

# modify the installation path and env name if you want. Assumes $WRKSPC is set.
WRKSPC="/data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/third_party/mtp-lm"
INSTALLDIR=${WRKSPC}
ENV_NAME="torch_210_cuda_129_singleshot"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.
# Steps below assume that conda command is available and a base environment is active.

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
#conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.13.5 --yes -c defaults
#source activate ${INSTALLDIR}/$ENV_NAME
source /data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/miniforge3/etc/profile.d/conda.sh
conda activate /data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/.venv
cd /data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/third_party/mtp-lm
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Also HPC cluster specific, remove if not relevant.
# Conda packages:
#conda install -c conda-forge conda-pack libstdcxx-ng  --yes

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
cd ${INSTALLDIR}
pip install --pre torch==2.10.0.dev20251213+cu129 torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/nightly/cu129
pip install ninja packaging numpy

cd "${REPO}"
pip install -e '.[all]'
cd ${INSTALLDIR}

# extras
pip install transformers hf_transfer litdata datasets matplotlib torchdata tenacity ipykernel jupyterlab langdetect immutabledict unitxt
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
