#!/usr/bin/env python3
"""
Comprehensive RMSNorm Test for Mistral24B Model

Single test function that compares PyTorch reference vs TTNN implementations
on different mesh devices (N300 and T3K) with detailed statistics and metrics.

This test captures:
- Real input tensors captured via PyTorch hooks from the full multimodal model
- Real image and text inputs processed through HuggingFace AutoProcessor
- Real weights from the model state dict
- PyTorch reference output
- TTNN N300 output
- TTNN T3K output
- Comprehensive comparison metrics between all three implementations

Tests the ln_pre RMSNorm layer: PixtralRMSNorm((1024,), eps=1e-05)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.experimental.mistral_24b.tt.rmsnorm import RMSNorm
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from ttnn import ConcatMeshToTensor


@dataclass
class TensorStats:
    """Statistics for a tensor."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    sparsity: float  # Percentage of zeros
    inf_count: int
    nan_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "sparsity": self.sparsity,
            "inf_count": self.inf_count,
            "nan_count": self.nan_count,
        }


@dataclass
class ComparisonMetrics:
    """Comparison metrics between two tensors."""

    pcc: float
    atol: float
    rtol: float
    mse: float
    mae: float
    max_abs_error: float
    max_rel_error: float
    cosine_similarity: float
    passing_elements_ratio: float  # Ratio of elements within tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pcc": self.pcc,
            "atol": self.atol,
            "rtol": self.rtol,
            "mse": self.mse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "cosine_similarity": self.cosine_similarity,
            "passing_elements_ratio": self.passing_elements_ratio,
        }


@dataclass
class MeshDeviceResult:
    """Results for a specific mesh device configuration."""

    device_name: str
    device_config: Tuple[int, int]
    output_stats: TensorStats
    vs_torch_metrics: ComparisonMetrics
    test_passed: bool


@dataclass
class ComprehensiveRMSNormTestResult:
    """Complete test result with all statistics across mesh devices."""

    test_passed: bool
    pcc_threshold: float

    # Input and weight statistics
    input_stats: TensorStats
    weight_stats: TensorStats

    # Reference output statistics
    torch_output_stats: TensorStats

    # TTNN results for different mesh devices
    mesh_results: Dict[str, MeshDeviceResult]

    # Cross-device comparison metrics
    n300_vs_t3k_metrics: Optional[ComparisonMetrics]

    # Test configuration
    test_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_passed": self.test_passed,
            "pcc_threshold": self.pcc_threshold,
            "input_stats": self.input_stats.to_dict(),
            "weight_stats": self.weight_stats.to_dict(),
            "torch_output_stats": self.torch_output_stats.to_dict(),
            "mesh_results": {
                k: {
                    "device_name": v.device_name,
                    "device_config": v.device_config,
                    "output_stats": v.output_stats.to_dict(),
                    "vs_torch_metrics": v.vs_torch_metrics.to_dict(),
                    "test_passed": v.test_passed,
                }
                for k, v in self.mesh_results.items()
            },
            "n300_vs_t3k_metrics": self.n300_vs_t3k_metrics.to_dict() if self.n300_vs_t3k_metrics else None,
            "test_config": self.test_config,
        }


def compute_tensor_stats(tensor: torch.Tensor, name: str) -> TensorStats:
    """Compute comprehensive statistics for a tensor."""
    tensor_flat = tensor.flatten().float()

    return TensorStats(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        mean=tensor_flat.mean().item(),
        std=tensor_flat.std().item(),
        min_val=tensor_flat.min().item(),
        max_val=tensor_flat.max().item(),
        median=tensor_flat.median().item(),
        q25=torch.quantile(tensor_flat, 0.25).item(),
        q75=torch.quantile(tensor_flat, 0.75).item(),
        sparsity=(tensor_flat == 0).float().mean().item() * 100,
        inf_count=torch.isinf(tensor_flat).sum().item(),
        nan_count=torch.isnan(tensor_flat).sum().item(),
    )


def compute_comparison_metrics(
    torch_tensor: torch.Tensor, ttnn_tensor: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5
) -> ComparisonMetrics:
    """Compute comprehensive comparison metrics between two tensors."""

    # Ensure same shape and dtype
    assert torch_tensor.shape == ttnn_tensor.shape, f"Shape mismatch: {torch_tensor.shape} vs {ttnn_tensor.shape}"

    torch_flat = torch_tensor.flatten().float()
    ttnn_flat = ttnn_tensor.flatten().float()

    # Basic metrics
    diff = torch_flat - ttnn_flat
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(torch_flat) + 1e-8)

    # PCC (Pearson Correlation Coefficient)
    pcc = torch.corrcoef(torch.stack([torch_flat, ttnn_flat]))[0, 1].item()

    # Tolerance-based metrics
    atol_achieved = abs_diff.max().item()
    rtol_achieved = rel_diff.max().item()

    # MSE and MAE
    mse = torch.mean(diff**2).item()
    mae = torch.mean(abs_diff).item()

    # Max errors
    max_abs_error = abs_diff.max().item()
    max_rel_error = rel_diff.max().item()

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(torch_flat.unsqueeze(0), ttnn_flat.unsqueeze(0)).item()

    # Passing elements ratio
    within_tolerance = torch.isclose(torch_flat, ttnn_flat, atol=atol, rtol=rtol)
    passing_ratio = within_tolerance.float().mean().item()

    return ComparisonMetrics(
        pcc=pcc,
        atol=atol_achieved,
        rtol=rtol_achieved,
        mse=mse,
        mae=mae,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        cosine_similarity=cosine_sim,
        passing_elements_ratio=passing_ratio,
    )


class InputCaptureHook:
    """Hook to capture inputs to the ln_pre RMSNorm layer."""

    def __init__(self):
        self.captured_inputs = []
        self.hook_handle = None

    def __call__(self, module, input, output):
        """Hook function that captures the input tensor."""
        if len(input) > 0:
            # Capture the first input tensor
            captured_input = input[0].clone().detach()
            self.captured_inputs.append(captured_input)
            logger.info(f"Captured RMSNorm input tensor with shape: {captured_input.shape}")

    def attach_to_module(self, module):
        """Attach this hook to a specific module."""
        self.hook_handle = module.register_forward_hook(self)
        return self

    def remove(self):
        """Remove the hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_latest_input(self):
        """Get the most recently captured input."""
        if self.captured_inputs:
            return self.captured_inputs[-1]
        return None


def capture_real_rmsnorm_inputs(model_args) -> torch.Tensor:
    """
    Capture real input tensor to the ln_pre RMSNorm layer by running the full multimodal model
    with real dog image and haiku text request using PyTorch hooks.

    Args:
        model_args: Model configuration

    Returns:
        Captured input tensor from the ln_pre RMSNorm layer
    """
    logger.info("Capturing real RMSNorm input from full multimodal model...")

    # Import required modules for multimodal processing
    from transformers import AutoProcessor
    from transformers import Mistral3ForConditionalGeneration
    from PIL import Image
    import os
    from pkg_resources import resource_filename

    # Load the full HuggingFace Mistral3 multimodal model for reference
    model_id = os.getenv("HF_MODEL", "mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    logger.info(f"Loading HuggingFace Mistral3ForConditionalGeneration model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    reference_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",  # Keep on CPU to avoid GPU memory issues
        torch_dtype=torch.bfloat16,
    )
    reference_model.eval()

    # Create hook to capture inputs to ln_pre RMSNorm
    input_hook = InputCaptureHook()

    # Find the ln_pre RMSNorm module in the vision tower and attach hook
    ln_pre_module = None
    for name, module in reference_model.named_modules():
        if "ln_pre" in name.lower() and hasattr(module, "weight"):
            ln_pre_module = module
            logger.info(f"Found ln_pre RMSNorm module at: {name}")
            break
        # Also check for RMSNorm layers in vision tower before transformer
        elif "vision" in name.lower() and "ln_pre" in name.lower():
            ln_pre_module = module
            logger.info(f"Found ln_pre module in vision tower at: {name}")
            break

    if ln_pre_module is None:
        raise RuntimeError("Could not find ln_pre RMSNorm module in the multimodal model")

    # Attach hook
    input_hook.attach_to_module(ln_pre_module)

    # Load only the dog image for simplicity (Dog Image + Haiku Request)
    IMG_PATH = resource_filename("llama_models", "scripts/resources/")

    # Load only the dog image from llama_models resources
    with open(os.path.join(IMG_PATH, "dog.jpg"), "rb") as f:
        dog_image = Image.open(f).convert("RGB")
        logger.info(f"Loaded dog image: {dog_image.size}")

    # Use only the haiku prompt for simplicity
    text_prompt = "Write a haiku for this image."

    try:
        with torch.no_grad():
            logger.info("Processing Dog Image + Haiku Request")
            logger.info(f"Image size: {dog_image.size}, Text prompt: '{text_prompt}'")

            # Create multimodal message with dog image and haiku request
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": dog_image}, {"type": "text", "text": text_prompt}],
                }
            ]

            # Apply chat template and process inputs
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process with the multimodal processor
            inputs = processor(
                text=[prompt_text],
                images=[dog_image],
                return_tensors="pt",
            ).to("cpu", dtype=torch.bfloat16)

            logger.info("Running multimodal forward pass")
            logger.info(
                f"Input pixel values shape: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'None'}"
            )
            logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")

            # Run forward pass through the multimodal model to trigger hook
            try:
                # Just run a forward pass to capture the RMSNorm input
                _ = reference_model.generate(
                    **inputs,
                    max_new_tokens=1,  # Minimal generation to trigger vision processing
                    do_sample=False,
                    pad_token_id=reference_model.config.pad_token_id,
                )
            except Exception as e:
                # Even if generation fails, the vision processing should have occurred
                logger.warning(f"Generation failed but vision processing likely completed: {e}")

            # Get the captured input
            captured_input = input_hook.get_latest_input()
            if captured_input is not None:
                logger.info(f"Successfully captured input with shape: {captured_input.shape}")
                logger.info(
                    f"Input stats: min={captured_input.min():.6f}, max={captured_input.max():.6f}, "
                    f"mean={captured_input.mean():.6f}, std={captured_input.std():.6f}"
                )
            else:
                raise RuntimeError("No input was captured by the hook")

    finally:
        # Clean up hook
        input_hook.remove()
        # Clean up model to free memory
        del reference_model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("Successfully captured real input tensor from multimodal model")
    return captured_input


def run_ttnn_rmsnorm(mesh_device, device_name: str, model_args, state_dict, input_tensor):
    """Run TTNN RMSNorm implementation for a specific mesh device."""
    logger.info(f"Running TTNN RMSNorm on {device_name} device...")

    # RMSNorm parameters
    dim = model_args.vision_dim  # 1024
    eps = 1e-05

    first_layer_prefix = "vision_tower."
    dtype = ttnn.bfloat16

    # Create TTNN RMSNorm model
    tt_model = RMSNorm(
        device=mesh_device,
        dim=dim,
        state_dict=state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_key="ln_pre",
        weight_dtype=dtype,
        is_distributed=False,
        eps=eps,
    )

    # Convert input to TTNN tensor
    tt_input = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TTNN implementation
    mode = "prefill"  # Use prefill mode
    tt_output = tt_model(tt_input, mode=mode)

    # Convert TTNN output to torch
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # Select appropriate slice based on the original tensor dimensions
    tt_output_torch = tt_output_torch[:, :, : tt_output.shape[-1]]

    logger.info(f"{device_name} output shape: {tt_output_torch.shape}")
    return tt_output_torch


def comprehensive_rmsnorm_test(
    mesh_devices: Dict[str, Any],
    pcc_threshold: float = 0.99,
    atol_threshold: float = 1e-2,
    rtol_threshold: float = 1e-2,
    save_results: bool = True,
    results_dir: str = "comprehensive_rmsnorm_test_results",
) -> ComprehensiveRMSNormTestResult:
    """
    Comprehensive test function comparing PyTorch reference vs TTNN implementations
    across different mesh devices using real captured inputs and real weights.

    Args:
        mesh_devices: Dictionary of mesh device configurations {"N300": device, "T3K": device}
        pcc_threshold: Minimum PCC required for pass
        atol_threshold: Absolute tolerance threshold
        rtol_threshold: Relative tolerance threshold
        save_results: Whether to save detailed results to file
        results_dir: Directory to save results

    Returns:
        ComprehensiveRMSNormTestResult with all statistics and metrics
    """

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MISTRAL24B RMSNORM TEST WITH MULTIMODAL REAL INPUTS")
    logger.info("=" * 80)

    # Test configuration
    test_config = {
        "pcc_threshold": pcc_threshold,
        "atol_threshold": atol_threshold,
        "rtol_threshold": rtol_threshold,
        "scenario": "Dog Image + Haiku Request",
        "layer": "ln_pre RMSNorm",
        "mesh_devices": {name: str(device) for name, device in mesh_devices.items()},
        "device_counts": {name: device.get_num_devices() for name, device in mesh_devices.items()},
    }

    logger.info(f"Test Configuration: {test_config}")

    # Use first device for model initialization (doesn't matter which for state dict loading)
    first_device = list(mesh_devices.values())[0]
    model_args = ModelArgs(first_device)
    state_dict = model_args.load_state_dict()

    # RMSNorm parameters
    dim = model_args.vision_dim  # 1024
    eps = 1e-05

    logger.info(f"RMSNorm params: dim={dim}, eps={eps}")

    # ==================== CAPTURE REAL INPUT FROM MULTIMODAL MODEL ====================
    logger.info("\n" + "-" * 50)
    logger.info("CAPTURING REAL INPUT FROM FULL MULTIMODAL MODEL")
    logger.info("-" * 50)

    # Capture real input tensor using hooks
    input_tensor = capture_real_rmsnorm_inputs(model_args)

    logger.info(f"Using captured input tensor with shape: {input_tensor.shape}")
    logger.info(
        f"Input tensor stats: min={input_tensor.min():.6f}, max={input_tensor.max():.6f}, "
        f"mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}"
    )

    # ==================== PYTORCH REFERENCE ====================
    logger.info("\n" + "-" * 50)
    logger.info("PYTORCH REFERENCE")
    logger.info("-" * 50)

    # Get reference model and weights
    reference_model = model_args.reference_vision_rms()
    first_layer_prefix = "vision_tower.ln_pre."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }
    reference_model.load_state_dict(partial_state_dict)

    # Run PyTorch reference
    with torch.no_grad():
        torch_output = reference_model(input_tensor)

    logger.info(f"PyTorch output shape: {torch_output.shape}")

    # ==================== TTNN IMPLEMENTATIONS ====================
    logger.info("\n" + "-" * 50)
    logger.info("TTNN IMPLEMENTATIONS")
    logger.info("-" * 50)

    mesh_results = {}
    ttnn_outputs = {}

    for device_name, mesh_device in mesh_devices.items():
        logger.info(f"\n--- Running on {device_name} ---")

        try:
            tt_output_torch = run_ttnn_rmsnorm(mesh_device, device_name, model_args, state_dict, input_tensor)
            ttnn_outputs[device_name] = tt_output_torch

            # Compute output statistics
            output_stats = compute_tensor_stats(tt_output_torch, f"{device_name}_output")

            # Compute comparison metrics vs torch
            vs_torch_metrics = compute_comparison_metrics(
                torch_output, tt_output_torch, atol=atol_threshold, rtol=rtol_threshold
            )

            # Test pass/fail for this device
            device_test_passed = (
                vs_torch_metrics.pcc >= pcc_threshold
                and vs_torch_metrics.max_abs_error <= atol_threshold
                and vs_torch_metrics.max_rel_error <= rtol_threshold
                and output_stats.nan_count == 0
                and output_stats.inf_count == 0
            )

            device_config = (1, 2) if device_name == "N300" else (1, 8) if device_name == "T3K" else (0, 0)

            mesh_results[device_name] = MeshDeviceResult(
                device_name=device_name,
                device_config=device_config,
                output_stats=output_stats,
                vs_torch_metrics=vs_torch_metrics,
                test_passed=device_test_passed,
            )

            logger.info(f"{device_name} test: {'✅ PASSED' if device_test_passed else '❌ FAILED'}")
            logger.info(f"{device_name} PCC vs PyTorch: {vs_torch_metrics.pcc:.8f}")

        except Exception as e:
            logger.error(f"Error running {device_name}: {e}")
            # Create failed result
            mesh_results[device_name] = MeshDeviceResult(
                device_name=device_name,
                device_config=(0, 0),
                output_stats=TensorStats("error", (0,), "error", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                vs_torch_metrics=ComparisonMetrics(
                    0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 0, 0
                ),
                test_passed=False,
            )

    # ==================== STATISTICS COMPUTATION ====================
    logger.info("\n" + "-" * 50)
    logger.info("COMPUTING STATISTICS")
    logger.info("-" * 50)

    # Input statistics
    input_stats = compute_tensor_stats(input_tensor, "input")
    logger.info(
        f"Input stats computed: shape={input_stats.shape}, range=[{input_stats.min_val:.6f}, {input_stats.max_val:.6f}]"
    )

    # Weight statistics
    weight_tensor = state_dict[f"{first_layer_prefix}weight"]
    weight_stats = compute_tensor_stats(weight_tensor, "rmsnorm_weight")
    logger.info(
        f"Weight stats computed: shape={weight_stats.shape}, range=[{weight_stats.min_val:.6f}, {weight_stats.max_val:.6f}]"
    )

    # Output statistics
    torch_output_stats = compute_tensor_stats(torch_output, "torch_output")
    logger.info(f"PyTorch output stats: range=[{torch_output_stats.min_val:.6f}, {torch_output_stats.max_val:.6f}]")

    # Cross-device comparison (N300 vs T3K)
    n300_vs_t3k_metrics = None
    if "N300" in ttnn_outputs and "T3K" in ttnn_outputs:
        logger.info("\n--- Cross-device comparison (N300 vs T3K) ---")
        n300_vs_t3k_metrics = compute_comparison_metrics(
            ttnn_outputs["N300"], ttnn_outputs["T3K"], atol=atol_threshold, rtol=rtol_threshold
        )
        logger.info(f"N300 vs T3K PCC: {n300_vs_t3k_metrics.pcc:.8f}")

    # Overall test pass/fail
    overall_test_passed = all(result.test_passed for result in mesh_results.values())

    # ==================== RESULTS SUMMARY ====================
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info(f"OVERALL TEST STATUS: {'✅ PASSED' if overall_test_passed else '❌ FAILED'}")
    logger.info(f"")

    # Per-device results
    for device_name, result in mesh_results.items():
        logger.info(f"{device_name.upper()} RESULTS:")
        logger.info(f"  Status:             {'✅ PASSED' if result.test_passed else '❌ FAILED'}")
        logger.info(f"  PCC vs PyTorch:     {result.vs_torch_metrics.pcc:.8f} (threshold: {pcc_threshold})")
        logger.info(f"  ATol (achieved):    {result.vs_torch_metrics.atol:.2e} (threshold: {atol_threshold:.2e})")
        logger.info(f"  RTol (achieved):    {result.vs_torch_metrics.rtol:.2e} (threshold: {rtol_threshold:.2e})")
        logger.info(f"  MSE:               {result.vs_torch_metrics.mse:.2e}")
        logger.info(f"  MAE:               {result.vs_torch_metrics.mae:.2e}")
        logger.info(f"  Cosine Similarity: {result.vs_torch_metrics.cosine_similarity:.8f}")
        logger.info(f"  Output Range:      [{result.output_stats.min_val:.6f}, {result.output_stats.max_val:.6f}]")
        logger.info(f"")

    # Cross-device comparison
    if n300_vs_t3k_metrics:
        logger.info(f"CROSS-DEVICE COMPARISON (N300 vs T3K):")
        logger.info(f"  PCC:               {n300_vs_t3k_metrics.pcc:.8f}")
        logger.info(f"  MSE:               {n300_vs_t3k_metrics.mse:.2e}")
        logger.info(f"  MAE:               {n300_vs_t3k_metrics.mae:.2e}")
        logger.info(f"  Cosine Similarity: {n300_vs_t3k_metrics.cosine_similarity:.8f}")
        logger.info(f"")

    logger.info(f"INPUT STATISTICS:")
    logger.info(f"  Shape:              {input_stats.shape}")
    logger.info(f"  Mean ± Std:         {input_stats.mean:.6f} ± {input_stats.std:.6f}")
    logger.info(f"  Range:              [{input_stats.min_val:.6f}, {input_stats.max_val:.6f}]")
    logger.info(f"  Sparsity:           {input_stats.sparsity:.2f}%")

    logger.info(f"")
    logger.info(f"WEIGHT STATISTICS:")
    logger.info(f"  Shape:              {weight_stats.shape}")
    logger.info(f"  Mean ± Std:         {weight_stats.mean:.6f} ± {weight_stats.std:.6f}")
    logger.info(f"  Range:              [{weight_stats.min_val:.6f}, {weight_stats.max_val:.6f}]")
    logger.info(f"  Sparsity:           {weight_stats.sparsity:.2f}%")

    logger.info(f"")
    logger.info(f"PYTORCH OUTPUT STATISTICS:")
    logger.info(f"  Shape:              {torch_output_stats.shape}")
    logger.info(f"  Mean ± Std:         {torch_output_stats.mean:.6f} ± {torch_output_stats.std:.6f}")
    logger.info(f"  Range:              [{torch_output_stats.min_val:.6f}, {torch_output_stats.max_val:.6f}]")
    logger.info(f"  Sparsity:           {torch_output_stats.sparsity:.2f}%")
    logger.info(f"  NaN Count:          {torch_output_stats.nan_count}")
    logger.info(f"  Inf Count:          {torch_output_stats.inf_count}")

    # Create comprehensive result
    result = ComprehensiveRMSNormTestResult(
        test_passed=overall_test_passed,
        pcc_threshold=pcc_threshold,
        input_stats=input_stats,
        weight_stats=weight_stats,
        torch_output_stats=torch_output_stats,
        mesh_results=mesh_results,
        n300_vs_t3k_metrics=n300_vs_t3k_metrics,
        test_config=test_config,
    )

    # Save results if requested
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)

        # Save JSON report
        json_path = results_path / "comprehensive_rmsnorm_test_results.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save tensors
        tensors_to_save = {"input": input_tensor, "torch_output": torch_output, "weight": weight_tensor}

        # Save the captured input
        tensors_to_save["captured_input"] = input_tensor

        # Add TTNN outputs
        for device_name, output in ttnn_outputs.items():
            tensors_to_save[f"{device_name.lower()}_output"] = output

        torch.save(tensors_to_save, results_path / "test_tensors.pt")

        logger.info(f"")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"  JSON report: {json_path}")
        logger.info(f"  Tensors: {results_path / 'test_tensors.pt'}")

    logger.info("=" * 80)

    return result


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device_configs",
    [
        # Test on both N300 and T3K if available
        {"N300": (1, 2), "T3K": (1, 8)},
    ],
    indirect=True,
)
def test_comprehensive_rmsnorm_cross_mesh(mesh_device_configs):
    """Comprehensive RMSNorm test across multiple mesh device configurations."""
    logger.info("Running comprehensive RMSNorm test across mesh devices...")

    # Filter to only available devices based on environment
    available_devices = {}
    for device_name, device in mesh_device_configs.items():
        if device is not None:
            available_devices[device_name] = device

    if not available_devices:
        pytest.skip("No mesh devices available for testing")

    logger.info(f"Testing on devices: {list(available_devices.keys())}")

    try:
        result = comprehensive_rmsnorm_test(mesh_devices=available_devices, pcc_threshold=0.99, save_results=True)

        # Assert overall test passed
        assert result.test_passed, f"RMSNorm test failed. Check detailed results in logs."

        # Assert each device passed
        for device_name, mesh_result in result.mesh_results.items():
            assert mesh_result.test_passed, f"RMSNorm test failed on {device_name} device"

        # If we have both N300 and T3K, check cross-device consistency
        if "N300" in result.mesh_results and "T3K" in result.mesh_results:
            if result.n300_vs_t3k_metrics:
                cross_device_pcc = result.n300_vs_t3k_metrics.pcc
                assert (
                    cross_device_pcc >= 0.995
                ), f"Cross-device consistency too low: N300 vs T3K PCC = {cross_device_pcc:.6f}"
                logger.info(f"✅ Cross-device consistency check passed: N300 vs T3K PCC = {cross_device_pcc:.6f}")

        return result

    finally:
        # Clean up mesh devices
        for device in available_devices.values():
            try:
                ttnn.close_mesh_device(device)
            except:
                pass


# Additional parametrized test for specific mesh configurations
@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, 2)  # Default to N300 config
        )
    ],
    indirect=True,
)
def test_single_device_rmsnorm(mesh_device):
    """Test RMSNorm on a single mesh device configuration."""
    logger.info("Running single device RMSNorm test...")

    # Determine device name based on config
    num_devices = mesh_device.get_num_devices()
    if num_devices == 2:
        device_name = "N300"
    elif num_devices == 8:
        device_name = "T3K"
    else:
        device_name = f"CUSTOM_{num_devices}"

    try:
        result = comprehensive_rmsnorm_test(
            mesh_devices={device_name: mesh_device},
            pcc_threshold=0.99,
            save_results=True,
            results_dir=f"rmsnorm_test_results_{device_name.lower()}",
        )

        assert result.test_passed, f"RMSNorm test failed on {device_name}"

        return result

    finally:
        ttnn.close_mesh_device(mesh_device)
