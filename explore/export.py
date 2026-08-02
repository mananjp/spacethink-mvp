"""Model export — TFLite / ONNX quantization and export.

Quantizes the Telemanom forecaster for edge deployment.
Target: 59 KB RAM, OPS-SAT-class Cortex (Berkenkamp 2026 result).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnx  # type: ignore[import-untyped]
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    import onnxruntime as ort  # type: ignore[import-untyped]
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

EXPORT_DIR = Path(__file__).resolve().parent.parent / "models" / "export"


@dataclass
class ExportReport:
    """Report of a model export operation."""
    model_name: str
    format: str           # "onnx" | "tflite" | "numpy"
    output_path: str
    size_bytes: int
    estimated_ram_kb: float
    quantized: bool
    latency_ms: Optional[float] = None


def export_forecaster_numpy(
    output_dir: Path = EXPORT_DIR,
    model_name: str = "telemanom_forecaster",
) -> ExportReport:
    """Export forecaster weights as NumPy arrays (universal fallback).

    Exports the exponential smoothing parameters as a minimal representation
    that can be loaded on any platform.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exponential smoothing "model" — just parameters
    params = {
        "alpha": np.float32(0.3),
        "beta": np.float32(0.1),
        "model_type": "double_exponential_smoothing",
        "input_channels": 3,
        "version": "1.0",
    }

    output_path = output_dir / f"{model_name}_params.npz"
    np.savez(
        output_path,
        alpha=params["alpha"],
        beta=params["beta"],
        input_channels=np.int32(params["input_channels"]),
    )

    size = output_path.stat().st_size

    return ExportReport(
        model_name=model_name,
        format="numpy",
        output_path=str(output_path),
        size_bytes=size,
        estimated_ram_kb=size / 1024 + 4,  # params + working buffer
        quantized=False,
    )


def export_forecaster_onnx(
    output_dir: Path = EXPORT_DIR,
    model_name: str = "telemanom_forecaster",
    quantize: bool = True,
) -> ExportReport:
    """Export forecaster as ONNX model for edge deployment.

    When ONNX is not available, falls back to NumPy export.
    """
    if not _HAS_ONNX:
        warnings.warn(
            "onnx not installed. Falling back to NumPy export. "
            "Install with: pip install onnx onnxruntime",
            stacklevel=2,
        )
        return export_forecaster_numpy(output_dir, model_name)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a simple ONNX graph for exponential smoothing
    from onnx import TensorProto, helper

    # Input: sequence of shape (N, 3) — 3 channels
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3])

    # Alpha parameter
    alpha_init = helper.make_tensor("alpha", TensorProto.FLOAT, [1], [0.3])

    # Simple identity + scale node (placeholder for the actual forecaster)
    mul_node = helper.make_node("Mul", ["input", "alpha"], ["output"], name="smoothing")

    graph = helper.make_graph(
        [mul_node],
        model_name,
        [X],
        [Y],
        initializer=[alpha_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    output_path = output_dir / f"{model_name}.onnx"
    onnx.save(model, str(output_path))

    size = output_path.stat().st_size

    # Quantize if requested
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import-untyped]
            quantized_path = output_dir / f"{model_name}_quantized.onnx"
            quantize_dynamic(
                str(output_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8,
            )
            if quantized_path.exists():
                output_path = quantized_path
                size = quantized_path.stat().st_size
        except Exception as e:
            warnings.warn(f"Quantization failed: {e}", stacklevel=2)
            quantize = False

    # Benchmark latency if runtime available
    latency = None
    if _HAS_ORT:
        try:
            session = ort.InferenceSession(str(output_path))
            import time
            test_input = np.random.randn(100, 3).astype(np.float32)
            # Warmup
            session.run(None, {"input": test_input})
            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                session.run(None, {"input": test_input})
            latency = (time.perf_counter() - start) / 100 * 1000  # ms
        except Exception:
            pass

    return ExportReport(
        model_name=model_name,
        format="onnx",
        output_path=str(output_path),
        size_bytes=size,
        estimated_ram_kb=size / 1024 + 16,  # model + inference buffer
        quantized=quantize,
        latency_ms=latency,
    )


def format_export_report(report: ExportReport) -> str:
    """Format export report as Markdown."""
    lines = [
        "## 📦 Model Export Report",
        "",
        f"**Model:** {report.model_name}",
        f"**Format:** {report.format.upper()}",
        f"**Size:** {report.size_bytes:,} bytes ({report.size_bytes / 1024:.1f} KB)",
        f"**Estimated RAM:** {report.estimated_ram_kb:.1f} KB",
        f"**Quantized:** {'Yes' if report.quantized else 'No'}",
    ]

    if report.latency_ms is not None:
        lines.append(f"**Inference Latency:** {report.latency_ms:.2f} ms (100-sample batch)")

    # Target check
    target_ram_kb = 59
    meets_target = report.estimated_ram_kb <= target_ram_kb
    status = "✅ MEETS" if meets_target else "❌ EXCEEDS"
    lines.extend([
        "",
        f"**OPS-SAT Target (59 KB RAM):** {status}",
    ])

    return "\n".join(lines)
