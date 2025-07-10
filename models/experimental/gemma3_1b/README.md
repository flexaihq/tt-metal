# Gemma-3-1b-it Models

This folder includes the google/gemma-3-1b-it model.

The supported gemma model is compatible and tested on the following Tenstorrent hardware:
- N150 (1-chip)

## How to Run

### Setup TT environment

1. Set up environment variables:
```
export HF_MODEL="google/gemma-3-1b-it"
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

- `$HF_MODEL` sets the path for the Gemma model weights and caches.

- `$WH_ARCH_YAML` sets the dispatch over ethernet cores. This is optional for N150 and required for N300 and T3000, enabling a full core grid utilization (8x8), allowing for maximum performance of models.

On the first execution of each model, weights will get downloaded into your HuggingFace cache directory and will be getting reused.
```
model_cache/google/gemma-3-1b-it/N150  # For N150
```

### Run the demo

The current demo loads a prompt file, prefills the encoded prompt and then runs decode for specified number of iterations.

The demo is also parametrized to run for 1 or 32 continuous batch of users, i.e. to simulate multiple users generating text one after another.

The input prompts are based on the general weights. The prompts are included in the demo folder `models/tt_transformers/demo/sample_prompts/`.

When running the demo, do not forget to setup the `$HF_MODEL` environment variable to the corresponding gemma-3-1b-it model weights.

```
# Examples of how to run the demo

Currently, this model is only supported on N150 (single-device). If you are running this demo on a multi-chip device, please make sure MESH_DEVICE is set to N150.

# Run a single continuous batch
MESH_DEVICE=N150 pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# Run 2 for 32 continuous batches with weights
MESH_DEVICE=N150 pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"
```

## Known Issues

### 1. Variation in the PCC scores.

PCC (Pearson Correlation Coefficient) is used to measure the inference differences between the TT model and the reference model.

Specifically, we measure the PCC score for each token used during inference. While some tokens achieve high PCC scores, indicating close alignment with the reference implementation, others do not perform as well.

At present, we set the PCC score of 0.86 as the pass threshold in our tests.

Use Bfloat16 for better accuracy and better relevant output.
