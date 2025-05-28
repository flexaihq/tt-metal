#!/bin/bash
set -eo pipefail

# Allow overriding Python command via environment variable
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
else
    echo "Using user-specified Python: $PYTHON_CMD"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR=$(pwd)/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

UV_VERSION=0.7.8
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# Create and activate virtual environment
uv venv -p $PYTHON_CMD $PYTHON_ENV_DIR
source $PYTHON_ENV_DIR/bin/activate

#echo "Forcefully using a version of pip that will work with our view of editable installs"
#uv pip install --force-reinstall pip==21.2.4

echo "Setting up virtual env"
uv pip install setuptools wheel

echo "Installing dev dependencies"
uv pip install -r $(pwd)/tt_metal/python_env/requirements-dev.txt --index=https://download.pytorch.org/whl/cpu --index-strategy=unsafe-best-match

echo "Installing tt-metal"
uv pip install -e . --index=https://download.pytorch.org/whl/cpu --index-strategy=unsafe-best-match

# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
