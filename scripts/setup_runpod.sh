#!/usr/bin/env bash
# One-time setup on a RunPod pod (PyTorch template). Run from the repo root:
#   bash scripts/setup_runpod.sh
# Safe to re-run. Everything heavy lives under /workspace, which survives pod restarts.
set -euo pipefail

UPSTREAM_COMMIT=3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9
START=$(date +%s)

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

echo "== upstream code (nicklashansen/mmbench2 @ ${UPSTREAM_COMMIT:0:7})"
if [ ! -d third_party/mmbench2/.git ]; then
  git clone https://github.com/nicklashansen/mmbench2 third_party/mmbench2
fi
git -C third_party/mmbench2 fetch --quiet origin || true
git -C third_party/mmbench2 checkout --quiet "$UPSTREAM_COMMIT"

echo "== python packages"
python - <<'EOF'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if v < (2, 4):
    raise SystemExit("torch is older than 2.4: pick a newer RunPod PyTorch template")
EOF
pip install -q -r requirements.txt

echo "== caches"
mkdir -p /workspace/hf_cache
grep -q "HF_HOME=/workspace/hf_cache" ~/.bashrc 2>/dev/null || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
export HF_HOME=/workspace/hf_cache

echo "== pipeline test (CPU-sized, about a minute)"
python tests/test_pipeline.py

echo "setup done in $(( $(date +%s) - START ))s"
