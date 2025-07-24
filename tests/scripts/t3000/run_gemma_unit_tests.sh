#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

function run_gemma-3-4b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_gemma-3-4b_tests"
  HF_MODEL='google/gemma-3-4b-it'
  MESH_DEVICE=N150
  # 10 seconds should be more than enough
  TIMEOUT=10

  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_attention.py --timeout=$TIMEOUT ; fail+=$?
  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_attention_prefill.py --timeout=$TIMEOUT; fail+=$?
  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_embedding.py --timeout=$TIMEOUT; fail+=$?
  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_mlp.py --timeout=$TIMEOUT; fail+=$?
  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_rms_norm.py --timeout=$TIMEOUT; fail+=$?
#  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_decoder.py --timeout=$TIMEOUT; fail+=$?
#  MESH_DEVICE=$MESH_DEVICE HF_MODEL=$HF_MODEL pytest -n auto models/tt_transformers/tests/test_decoder_prefill.py --timeout=$TIMEOUT; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_gemma-3-4b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_gemma-3-4b_tests
