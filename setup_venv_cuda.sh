#!/usr/bin/env bash
# WeSpeaker dev env: Python 3.9 + PyTorch CUDA 12.4 wheels (matches recent NVIDIA drivers).
# System: driver CUDA 13.x / nvcc 12.x → use cu124 index (bundles CUDA 12.4 libs with PyTorch).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-${HOME}/.local/bin/python3.9}"
if [[ ! -x "$PY" ]]; then
  echo "Need Python 3.9 (README). Set PYTHON= to your python3.9 path. Not found: $PY" >&2
  exit 1
fi

"$PY" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# torchnet -> visdom build needs pkg_resources during metadata; setuptools 82+ can break isolated builds
pip install 'setuptools<81'
pip install -r requirements.txt --no-build-isolation
pip install -e .
# Optional but required for `import wespeaker` (frontend imports whisper + peft)
pip install openai-whisper peft

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import wespeaker; print('wespeaker OK')"

echo "Activate with: source ${ROOT}/.venv/bin/activate"
