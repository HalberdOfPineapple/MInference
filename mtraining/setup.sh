#!/usr/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda create -n mtrain -y python=3.10
conda activate mtrain

BASE_DIR="$(cd "$(dirname "$0")" && pwd)" # path/to/MInference/mtraining
PROJECT_ROOT="$(cd "${BASE_DIR}/.." && pwd)" # path/to/MInference
PIP="$(which pip)"

$PIP install -U pip setuptools wheel
$PIP install ninja cmake pybind11 packaging psutil pytest
$PIP install -r "${BASE_DIR}/requirements.txt"
$PIP install git+https://github.com/microsoft/nnscaler.git@2368540417bc3b77b7e714d3f1a0de8a51bb66e8 
$PIP install "rotary-emb @ git+https://github.com/Dao-AILab/flash-attention.git@9356a1c0389660d7e231ff3163c1ac17d9e3824a#subdirectory=csrc/rotary" --no-build-isolation
$PIP install "block_sparse_attn @ git+https://github.com/HalberdOfPineapple/flash-attention.git@block-sparse" --no-build-isolation
$PIP install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1 --no-build-isolation
$PIP install torch==2.3.1 torchvision==0.18.1
$PIP install triton==3.0.0

# Get the path to nnscaler and write its path to PYTHONPATH in ~/.profile
NNSCALER_HOME=$(python -c "import nnscaler; print(nnscaler.__path__[0])")
echo "export NNSCALER_HOME=${NNSCALER_HOME}" >> ~/.profile
echo "export PYTHONPATH=${NNSCALER_HOME}:${PROJECT_ROOT}:\${PYTHONPATH}" >> ~/.profile
source ~/.profile

cd $PROJECT_ROOT
MINFERENCE_FORCE_BUILD=TRUE $PIP install -e . --no-build-isolation

cd $BASE_DIR
$PIP install -e $BASE_DIR

