#!/usr/bin/env bash
# One-shot setup for a fresh vast.ai GPU instance: link nsys, install uv,
# clone/update the repo, and sync the environment. Safe to re-run.
set -euo pipefail

REPO_URL="https://github.com/Stav42/assignment2-systems.git"
REPO_DIR="$HOME/assignment2-systems"

SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"

echo "==> GPU check"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "==> Ensuring git/curl are present"
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq git curl
fi

echo "==> Linking nsys onto PATH"
if ! command -v nsys >/dev/null; then
    NSYS_BIN=$(find /opt/nvidia -iname "nsys" -type f 2>/dev/null | head -n1 || true)
    if [ -n "$NSYS_BIN" ]; then
        $SUDO ln -sf "$NSYS_BIN" /usr/local/bin/nsys
        echo "Linked nsys from $NSYS_BIN"
    else
        echo "No bundled nsys found under /opt/nvidia, trying apt..."
        $SUDO apt-get install -y -qq nsight-systems-cli || echo "WARNING: could not auto-install nsys; install manually."
    fi
fi
nsys --version || echo "WARNING: nsys still not available."

echo "==> Installing uv"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> Cloning or updating repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "==> Syncing Python environment"
uv sync

echo "==> Verifying CUDA torch"
uv run python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
"

echo ""
echo "==> Setup complete. Try:"
echo "    cd $REPO_DIR && ./cs336_systems/run_profile.sh small 256 forward_only"
