"""
src/deployment module — Model export and Processor-in-the-Loop (PIL) validation.
"""
from src.deployment.exporter import export_to_onnx, export_to_torchscript, count_parameters
from src.deployment.pil_runner import (
    EmbeddedGNCInferenceEngine,
    profile_inference_latency,
    run_closed_loop_pil_simulation,
)

__all__ = [
    "export_to_onnx",
    "export_to_torchscript",
    "count_parameters",
    "EmbeddedGNCInferenceEngine",
    "profile_inference_latency",
    "run_closed_loop_pil_simulation",
]
