#!/usr/bin/env bash

set -Eeuo pipefail

ENV_NAME="${1:-navila-eval}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${EXTERNAL_DIR:-"$ROOT_DIR/external"}"
HABITAT_SIM_DIR="${HABITAT_SIM_DIR:-"$EXTERNAL_DIR/habitat-sim"}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-"$EXTERNAL_DIR/habitat-lab"}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/navila-pip-cache}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.14}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
INSTALL_APT="${INSTALL_APT:-0}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-0}"
BUILD_HEADLESS="${BUILD_HEADLESS:-1}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

pip_install() {
    python -m pip --cache-dir "$PIP_CACHE_DIR" install --timeout 180 --retries 10 "$@"
}

clone_or_update() {
    local repo_url="$1"
    local ref="$2"
    local dst="$3"

    if [ -d "$dst/.git" ]; then
        log "Using existing source: $dst"
        git -C "$dst" fetch --depth 1 origin "$ref"
        git -C "$dst" checkout FETCH_HEAD
    else
        log "Cloning $repo_url ($ref) -> $dst"
        git clone --branch "$ref" --depth 1 "$repo_url" "$dst"
    fi
}

require_cmd conda
require_cmd git

if [ "$INSTALL_APT" = "1" ]; then
    log "Installing Ubuntu system packages for Habitat build"
    sudo apt-get update || true
    sudo apt-get install -y --no-install-recommends \
        libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev \
        mesa-utils xorg-dev freeglut3-dev
else
    log "Skipping apt packages. Set INSTALL_APT=1 to install Habitat system deps."
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$FORCE_RECREATE" = "1" ]; then
        log "Removing existing conda env: $ENV_NAME"
        conda env remove -n "$ENV_NAME" -y
    else
        log "Using existing conda env: $ENV_NAME"
    fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating conda env $ENV_NAME with Python $PYTHON_VERSION"
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

log "Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"

log "Pinning legacy-friendly Python packaging/build tools"
conda install -n "$ENV_NAME" \
    "python=$PYTHON_VERSION" \
    "pip=22.3.1" \
    "setuptools=65.6.3" \
    "wheel=0.38.4" \
    "cmake=3.14.0" \
    -y

conda activate "$ENV_NAME"
mkdir -p "$EXTERNAL_DIR" "$PIP_CACHE_DIR"

if [ "$INSTALL_CUDA_TOOLKIT" = "1" ]; then
    log "Installing conda CUDA toolkit"
    conda install -n "$ENV_NAME" -c nvidia cuda-toolkit -y
else
    log "Skipping conda cuda-toolkit. Set INSTALL_CUDA_TOOLKIT=1 if local nvcc is required."
fi

log "Installing Habitat-Sim Python dependencies"
pip_install ninja numpy==1.26.0
pip_install --only-binary=:all: \
    attrs==23.2.0 \
    numba==0.59.1 \
    numpy-quaternion==2023.0.4 \
    pillow==10.3.0 \
    scipy==1.11.4 \
    tqdm==4.66.4 \
    matplotlib==3.8.4 \
    gitpython==3.1.43 \
    imageio==2.34.1 \
    imageio-ffmpeg==0.4.9

clone_or_update "https://github.com/facebookresearch/habitat-sim.git" "v0.1.7" "$HABITAT_SIM_DIR"

log "Building and installing Habitat-Sim v0.1.7"
pushd "$HABITAT_SIM_DIR" >/dev/null
if [ "$BUILD_HEADLESS" = "1" ]; then
    python setup.py install --headless
else
    python setup.py install
fi
popd >/dev/null

clone_or_update "https://github.com/facebookresearch/habitat-lab.git" "v0.1.7" "$HABITAT_LAB_DIR"

log "Installing Habitat-Lab and VLN-CE runtime dependencies"
pip_install pytest-runner==6.0.1
pip_install \
    Cython==0.29.37 \
    gym==0.17.3 \
    yacs==0.1.8 \
    opencv-python==4.8.0.74 \
    moviepy==1.0.3 \
    ifcfg==0.24 \
    lmdb==1.4.1 \
    webdataset==0.1.103 \
    dtw==1.4.0 \
    fastdtw==0.3.4 \
    gdown==5.2.0 \
    jsonlines==4.0.0 \
    msgpack_numpy==0.4.8 \
    networkx==3.2.1 \
    tensorboard==2.15.2 \
    tensorflow-cpu==2.15.1

log "Installing Habitat-Lab v0.1.7 in editable mode"
pushd "$HABITAT_LAB_DIR" >/dev/null
python setup.py develop --all --no-deps
popd >/dev/null

log "Installing VILA / NaVILA package and extras"
pushd "$ROOT_DIR" >/dev/null
pip_install -e .
pip_install -e ".[train]"
pip_install -e ".[eval]"

log "Installing FlashAttention2 prebuilt wheel"
pip_install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

log "Installing Transformers from Hugging Face v4.37.2 tag"
pip_install --force-reinstall --no-deps git+https://github.com/huggingface/transformers@v4.37.2

log "Applying NaVILA Transformers and DeepSpeed replacements"
SITE_PKG_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -rv "$ROOT_DIR"/llava/train/transformers_replace/* "$SITE_PKG_PATH"/transformers/
cp -rv "$ROOT_DIR"/llava/train/deepspeed_replace/* "$SITE_PKG_PATH"/deepspeed/

log "Pinning WebDataset back to VLN-CE-compatible version"
pip_install webdataset==0.1.103

log "Applying Habitat-Sim NumPy compatibility hotfix"
python "$ROOT_DIR"/evaluation/scripts/habitat_sim_autofix.py
popd >/dev/null

if [ "$SKIP_SMOKE_TEST" != "1" ]; then
    log "Running import smoke tests"
    pushd "$ROOT_DIR/evaluation" >/dev/null
    python - <<'PY'
import habitat
import habitat_baselines
import habitat_sim
import torch
import torchvision
import transformers
import deepspeed
import flash_attn
import webdataset
from llava.model.builder import load_pretrained_model
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
import vlnce_baselines.navila_trainer

cfg = get_config(
    "vlnce_baselines/config/r2r_baselines/navila.yaml",
    ["EVAL_CKPT_PATH_DIR", "../checkpoints/navila-llama3-8b-8f"],
)

print("habitat_sim:", habitat_sim.__file__)
print("habitat:", habitat.__file__)
print("habitat_baselines:", habitat_baselines.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("deepspeed:", deepspeed.__version__)
print("flash_attn:", flash_attn.__version__)
print("webdataset:", getattr(webdataset, "__version__", "unknown"))
print("llava builder:", load_pretrained_model.__name__)
print("trainer:", baseline_registry.get_trainer(cfg.TRAINER_NAME).__name__)
PY
    popd >/dev/null
fi

log "NaVILA eval environment is ready: $ENV_NAME"
cat <<EOF

Usage:
  conda activate $ENV_NAME
  cd "$ROOT_DIR/evaluation"
  bash scripts/eval/r2r.sh ../checkpoints/navila-llama3-8b-8f 1 0 "0"

Useful options:
  INSTALL_APT=1              install Ubuntu Habitat build packages with sudo apt
  FORCE_RECREATE=1           recreate the conda env from scratch
  INSTALL_CUDA_TOOLKIT=1     install conda cuda-toolkit if local nvcc is needed
  SKIP_SMOKE_TEST=1          skip final import/config smoke test
  EXTERNAL_DIR=/path         choose where Habitat sources are cloned

EOF
