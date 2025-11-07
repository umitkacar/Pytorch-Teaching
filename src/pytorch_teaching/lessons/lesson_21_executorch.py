"""
Lesson 21: Mobile and Edge Deployment with ExecuTorch.

ExecuTorch is PyTorch's solution for deploying AI models on mobile and edge devices
with exceptional performance and minimal footprint (50KB runtime).

Learning Objectives:
    - Understand ExecuTorch architecture and benefits
    - Export PyTorch models for edge deployment
    - Optimize models for mobile constraints
    - Deploy on various hardware backends (Qualcomm, Apple, ARM, MediaTek)
    - Run LLMs on edge devices (Llama 3.2)
    - Implement on-device privacy-preserving AI

Official Documentation: https://pytorch.org/executorch/
"""

import torch
import torch.nn as nn
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def explain_executorch():
    """Explain what ExecuTorch is and its benefits."""
    console.print("\n[bold cyan]═══ What is ExecuTorch? ═══[/bold cyan]\n")

    console.print("[yellow]ExecuTorch is PyTorch's official solution for edge and mobile AI deployment.[/yellow]\n")

    benefits_table = Table(show_header=True, header_style="bold magenta")
    benefits_table.add_column("Feature", style="cyan", width=25)
    benefits_table.add_column("Benefit", style="green", width=50)

    benefits_table.add_row("Runtime Size", "🔥 Only 50KB - smallest in industry")
    benefits_table.add_row("Performance", "⚡ Hardware-accelerated via delegates")
    benefits_table.add_row("Deployment", "📱 Direct export without intermediate formats")
    benefits_table.add_row("Hardware Support", "🎯 Qualcomm, Apple, ARM, MediaTek, XNNPACK")
    benefits_table.add_row("Model Support", "🧠 CNNs, Transformers, LLMs (Llama 3.2)")
    benefits_table.add_row("Privacy", "🔒 100% on-device inference")
    benefits_table.add_row("Latency", "⏱️ <100ms for most models")
    benefits_table.add_row("PyTorch Native", "✅ No conversion required")

    console.print(benefits_table)

    console.print("\n[bold yellow]💡 Use Cases:[/bold yellow]")
    console.print("  • Mobile AI apps (iOS, Android)")
    console.print("  • Edge devices (IoT, embedded systems)")
    console.print("  • Real-time inference on resource-constrained hardware")
    console.print("  • Privacy-sensitive applications")
    console.print("  • On-device LLMs and GenAI\n")


def demonstrate_architecture():
    """Demonstrate ExecuTorch architecture."""
    console.print("\n[bold cyan]═══ ExecuTorch Architecture ═══[/bold cyan]\n")

    architecture_diagram = """
    ┌─────────────────────────────────────────────────────────┐
    │                   PyTorch Model (Eager)                 │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              torch.export() - AOT Export                │
    │           (Captures computation graph)                  │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │           Edge Dialect (ExecuTorch IR)                  │
    │       (Optimized for edge deployment)                   │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │           Quantization & Optimization                   │
    │      (INT8, INT4, pruning, fusion)                     │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │           Delegate Lowering (Optional)                  │
    │    Hardware-specific optimizations                      │
    │  (Qualcomm HTP, Apple Neural Engine, etc.)             │
    └───────────────────────┬─────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │           .pte File (ExecuTorch Program)                │
    │               (50KB Runtime)                            │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │      Mobile/Edge Device Inference                       │
    │    (iOS, Android, Embedded Linux, etc.)                │
    └─────────────────────────────────────────────────────────┘
    """

    console.print(architecture_diagram, style="green")


def demonstrate_model_export():
    """Demonstrate how to export a model for ExecuTorch."""
    console.print("\n[bold cyan]═══ Exporting Models for ExecuTorch ═══[/bold cyan]\n")

    console.print("[yellow]Step 1: Define your PyTorch model[/yellow]")

    model_code = '''
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
model.eval()
'''

    syntax = Syntax(model_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n[yellow]Step 2: Export using torch.export()[/yellow]")

    export_code = '''
from torch.export import export
from executorch.exir import to_edge

# Create example input
example_input = (torch.randn(1, 3, 224, 224),)

# Export the model
exported_program = export(model, example_input)

# Convert to ExecuTorch Edge Dialect
edge_program = to_edge(exported_program)

# Save the .pte file
edge_program.to_executorch().save("model.pte")
'''

    syntax = Syntax(export_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n[green]✓ Model exported to 'model.pte'![/green]\n")


def demonstrate_quantization():
    """Demonstrate quantization for ExecuTorch."""
    console.print("\n[bold cyan]═══ Quantization for Mobile Deployment ═══[/bold cyan]\n")

    console.print(
        "[yellow]Quantization reduces model size and improves inference speed on mobile devices.[/yellow]\n"
    )

    quantization_code = '''
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

# Create quantizer
quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config())

# Prepare for quantization
prepared_model = prepare_pt2e(exported_program, quantizer)

# Calibrate with sample data
for _ in range(100):
    sample_input = torch.randn(1, 3, 224, 224)
    prepared_model(sample_input)

# Convert to quantized model
quantized_model = convert_pt2e(prepared_model)

# Export quantized model
edge_program = to_edge(export(quantized_model, example_input))
edge_program.to_executorch().save("model_quantized.pte")
'''

    syntax = Syntax(quantization_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n[bold yellow]Quantization Benefits:[/bold yellow]")
    console.print("  • 4x smaller model size (FP32 → INT8)")
    console.print("  • 2-4x faster inference")
    console.print("  • Lower memory bandwidth")
    console.print("  • Longer battery life\n")


def demonstrate_delegates():
    """Demonstrate hardware delegates."""
    console.print("\n[bold cyan]═══ Hardware Delegates ═══[/bold cyan]\n")

    console.print(
        "[yellow]Delegates enable hardware-specific optimizations for maximum performance.[/yellow]\n"
    )

    delegates_table = Table(show_header=True, header_style="bold magenta")
    delegates_table.add_column("Delegate", style="cyan", width=20)
    delegates_table.add_column("Hardware", style="green", width=25)
    delegates_table.add_column("Performance", style="yellow", width=30)

    delegates_table.add_row(
        "Qualcomm HTP", "Snapdragon devices", "Up to 10x faster on Hexagon DSP"
    )
    delegates_table.add_row("Apple CoreML", "iPhone/iPad (A/M chips)", "Neural Engine acceleration")
    delegates_table.add_row("ARM Ethos-U", "ARM Cortex-M devices", "Optimized for microcontrollers")
    delegates_table.add_row("MediaTek NeuroPilot", "MediaTek SoCs", "APU acceleration")
    delegates_table.add_row("XNNPACK", "All platforms (CPU)", "Optimized CPU kernels")

    console.print(delegates_table)

    console.print("\n[yellow]Example: Using Qualcomm Delegate[/yellow]")

    delegate_code = '''
from executorch.backends.qualcomm import QnnBackend

# Configure Qualcomm HTP backend
backend_config = {
    "precision": "int8",  # Use INT8 quantization
    "use_htp": True,      # Use Hexagon Tensor Processor
}

# Lower to Qualcomm backend
edge_program_with_delegate = edge_program.to_backend(
    QnnBackend,
    compile_specs=[backend_config]
)

# Save with delegate
edge_program_with_delegate.save("model_qnn.pte")
'''

    syntax = Syntax(delegate_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)


def demonstrate_llm_deployment():
    """Demonstrate deploying LLMs with ExecuTorch."""
    console.print("\n[bold cyan]═══ Deploying LLMs on Edge (Llama 3.2) ═══[/bold cyan]\n")

    console.print(
        "[yellow]ExecuTorch enables running large language models directly on mobile devices![/yellow]\n"
    )

    console.print("[bold green]Llama 3.2 Models for Edge:[/bold green]")
    console.print("  • Llama 3.2 1B: Fits on most smartphones")
    console.print("  • Llama 3.2 3B: High quality on modern devices")
    console.print("  • INT4 quantization: Reduces size by 8x\n")

    llm_code = '''
# Download and prepare Llama 3.2
from executorch.examples.models.llama2 import export_llama

# Export Llama 3.2 1B for mobile
export_llama(
    model_name="llama-3.2-1b",
    quantization="int4",      # 4-bit quantization
    use_kv_cache=True,        # Enable KV cache for efficiency
    max_seq_length=512,
    output_path="llama_mobile.pte"
)

# Optimizations applied:
# - INT4 weight quantization
# - KV cache for autoregressive generation
# - Fused operators
# - Hardware-specific kernels
'''

    syntax = Syntax(llm_code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n[bold yellow]Real-world Performance:[/bold yellow]")
    console.print("  • iPhone 15 Pro: ~30 tokens/sec (Llama 3.2 1B)")
    console.print("  • Snapdragon 8 Gen 3: ~25 tokens/sec")
    console.print("  • Model size: ~500MB (INT4 quantized)")
    console.print("  • Memory usage: <2GB RAM\n")


def demonstrate_android_ios_deployment():
    """Demonstrate deployment on Android and iOS."""
    console.print("\n[bold cyan]═══ Android & iOS Deployment ═══[/bold cyan]\n")

    console.print("[bold yellow]Android Deployment:[/bold yellow]")

    android_code = '''
// Android (Java/Kotlin)
import org.pytorch.executorch.Module;

// Load the ExecuTorch model
Module module = Module.load("model.pte");

// Prepare input tensor
float[] input = new float[1 * 3 * 224 * 224];
Tensor inputTensor = Tensor.fromBlob(input, new long[]{1, 3, 224, 224});

// Run inference
Tensor output = module.forward(inputTensor);

// Get results
float[] scores = output.getDataAsFloatArray();
'''

    syntax = Syntax(android_code, "java", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("\n[bold yellow]iOS Deployment:[/bold yellow]")

    ios_code = '''
// iOS (Swift)
import ExecuTorch

// Load the model
let module = try! ExecuTorchModule(modelPath: "model.pte")

// Prepare input
let input = Tensor(shape: [1, 3, 224, 224], data: imageData)

// Run inference
let output = try! module.forward([input])

// Process results
let scores = output[0].floatData
'''

    syntax = Syntax(ios_code, "swift", theme="monokai", line_numbers=True)
    console.print(syntax)


def demonstrate_best_practices():
    """Demonstrate best practices for ExecuTorch deployment."""
    console.print("\n[bold cyan]═══ Best Practices ═══[/bold cyan]\n")

    practices = [
        ("Model Optimization", "Always quantize (INT8 or INT4) before deployment"),
        ("Delegate Selection", "Use hardware delegates when available (10x+ speedup)"),
        ("Memory Management", "Enable KV cache for LLMs, use in-place operations"),
        ("Input Preprocessing", "Move preprocessing to device when possible"),
        ("Batch Size", "Use batch_size=1 for lowest latency on mobile"),
        ("Testing", "Profile on target device, not desktop"),
        ("Model Size", "Keep under 100MB for easy app distribution"),
        ("Fallback", "Implement CPU fallback if hardware delegate unavailable"),
    ]

    for practice, description in practices:
        console.print(f"[green]✓ {practice}:[/green] {description}")

    console.print("\n[bold yellow]Performance Checklist:[/bold yellow]")
    console.print("  ☐ Model quantized (INT8/INT4)")
    console.print("  ☐ Hardware delegate configured")
    console.print("  ☐ Operators fused where possible")
    console.print("  ☐ Tested on target device")
    console.print("  ☐ Memory usage < 2GB")
    console.print("  ☐ Latency < 100ms for real-time apps\n")


def create_comparison_table():
    """Create comparison table of deployment options."""
    console.print("\n[bold cyan]═══ ExecuTorch vs Other Solutions ═══[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("ExecuTorch", style="green", width=20)
    table.add_column("TorchScript", style="yellow", width=20)
    table.add_column("ONNX Runtime", style="red", width=20)

    table.add_row("Runtime Size", "50KB 🔥", "~10MB", "~20MB")
    table.add_row("Mobile Support", "Native ✅", "Limited ⚠️", "Yes ✅")
    table.add_row("PyTorch Native", "Yes ✅", "Yes ✅", "No ❌")
    table.add_row("Hardware Delegates", "Extensive ✅", "Limited ⚠️", "Some ⚠️")
    table.add_row("LLM Support", "Yes (Llama) ✅", "No ❌", "Limited ⚠️")
    table.add_row("Quantization", "INT8, INT4 ✅", "INT8 ⚠️", "INT8 ✅")
    table.add_row("Export Format", ".pte direct ✅", "TorchScript ⚠️", "ONNX conversion ⚠️")

    console.print(table)


def run(interactive: bool = True, verbose: bool = False):
    """
    Run Lesson 21: Mobile and Edge Deployment with ExecuTorch.

    Args:
        interactive: If True, wait for user input between sections
        verbose: If True, show additional details
    """
    console.print(
        Panel.fit(
            "[bold cyan]Lesson 21: Mobile & Edge Deployment with ExecuTorch[/bold cyan]\n\n"
            "Deploy PyTorch models on mobile and edge devices with 50KB runtime.\n"
            "Learn to run LLMs on smartphones with maximum performance!",
            border_style="cyan",
        )
    )

    # Introduction
    explain_executorch()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Architecture
    demonstrate_architecture()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Model export
    demonstrate_model_export()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Quantization
    demonstrate_quantization()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Delegates
    demonstrate_delegates()
    if interactive:
        input("\n[Press Enter to continue...]")

    # LLM deployment
    demonstrate_llm_deployment()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Platform deployment
    demonstrate_android_ios_deployment()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Best practices
    demonstrate_best_practices()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Comparison
    create_comparison_table()

    console.print("\n[bold green]✓ Lesson 21 Complete![/bold green]")
    console.print(
        "\n[bold yellow]🚀 Next Steps:[/bold yellow]"
        "\n  • Try exporting your own models"
        "\n  • Experiment with different quantization schemes"
        "\n  • Profile on real mobile devices"
        "\n  • Explore hardware delegates for your target platform\n"
    )

    console.print("[cyan]📚 Resources:[/cyan]")
    console.print("  • Official Docs: https://pytorch.org/executorch/")
    console.print("  • Examples: https://github.com/pytorch/executorch/tree/main/examples")
    console.print("  • Llama on Mobile: https://github.com/pytorch/executorch/tree/main/examples/models/llama2\n")


if __name__ == "__main__":
    run(interactive=False)
