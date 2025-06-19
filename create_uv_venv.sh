#!/bin/bash
set -eo pipefail

HERE=$(dirname ${BASH_SOURCE[0]})

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
    PYTHON_ENV_DIR=${HERE}/python_env
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

UV_VERSION=0.7.8
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh

# Create and activate virtual environment
uv venv -p $PYTHON_CMD $PYTHON_ENV_DIR
source $PYTHON_ENV_DIR/bin/activate

echo "Setting up virtual env"
uv pip install ${UV_PIP_EXTRA_OPTS} setuptools wheel pip uv

echo "Installing dev dependencies"
uv pip install ${UV_PIP_EXTRA_OPTS} -r ${HERE}/tt_metal/python_env/requirements-dev.txt

echo "Installing tt-metal"
pushd ${HERE}
uv pip install ${UV_PIP_EXTRA_OPTS} -e .

# Do not install hooks when this is a worktree
if [ $(git rev-parse --git-dir) = $(git rev-parse --git-common-dir) ]; then
    echo "Generating git hooks"
    pre-commit install
    pre-commit install --hook-type commit-msg
else
    echo "In worktree: not generating git hooks"
fi

popd
echo "If you want stubs, run ./scripts/build_scripts/create_stubs.sh"
