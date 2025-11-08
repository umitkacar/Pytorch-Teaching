"""
PyTorch Teaching CLI - Interactive command-line interface for PyTorch lessons.

This module provides a modern, user-friendly CLI for accessing and running
PyTorch teaching lessons interactively.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from pytorch_teaching import __version__

# Lessons are imported on-demand to avoid dependency issues at CLI startup

app = typer.Typer(
    name="pytorch-teach",
    help="🔥 Professional PyTorch CLI Teaching Tool - Master Deep Learning",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()


def display_banner():
    """Display the PyTorch Teaching banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🔥 PyTorch Teaching - Professional Learning CLI 🔥    ║
    ║                                                           ║
    ║   Master Deep Learning from Basics to Production         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print(f"[bold green]Version:[/bold green] {__version__}")

    # Try to import torch for version info
    try:
        import torch
        console.print(f"[bold green]PyTorch:[/bold green] {torch.__version__}\n")
    except ImportError:
        console.print(f"[bold yellow]PyTorch:[/bold yellow] Not installed\n")


def check_cuda_availability():
    """Display CUDA availability status."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            console.print(
                f"[bold green]✓ CUDA Available:[/bold green] {cuda_version} "
                f"({device_count} device(s))"
            )
            console.print(f"[bold green]  GPU:[/bold green] {device_name}\n")
        else:
            console.print("[bold yellow]⚠ CUDA Not Available[/bold yellow] - Using CPU\n")
    except ImportError:
        console.print("[bold yellow]⚠ PyTorch Not Installed[/bold yellow]\n")


@app.command()
def info():
    """Display PyTorch Teaching information and system status."""
    display_banner()
    check_cuda_availability()

    # System information
    info_table = Table(title="System Information", show_header=True, header_style="bold magenta")
    info_table.add_column("Component", style="cyan", width=20)
    info_table.add_column("Version/Status", style="green")

    info_table.add_row("Python", f"{sys.version.split()[0]}")

    try:
        import torch
        info_table.add_row("PyTorch", torch.__version__)
        info_table.add_row("CUDA Available", "Yes ✓" if torch.cuda.is_available() else "No ✗")
        info_table.add_row("MPS Available", "Yes ✓" if torch.backends.mps.is_available() else "No ✗")
    except ImportError:
        info_table.add_row("PyTorch", "Not installed")
        info_table.add_row("CUDA Available", "N/A")
        info_table.add_row("MPS Available", "N/A")

    console.print(info_table)


@app.command()
def list_lessons():
    """List all available PyTorch lessons."""
    display_banner()

    lessons_tree = Tree("📚 [bold cyan]Available Lessons[/bold cyan]")

    # Foundation lessons
    foundation = lessons_tree.add("🎓 [bold yellow]Foundation[/bold yellow]")
    foundation.add("✅ Lesson 1: Tensor Fundamentals")
    foundation.add("✅ Lesson 2: Mathematical Operations")
    foundation.add("✅ Lesson 3: Device Management (CPU/CUDA)")
    foundation.add("🚧 Lesson 4: Autograd and Automatic Differentiation")
    foundation.add("🚧 Lesson 5: Building Neural Networks with nn.Module")
    foundation.add("🚧 Lesson 6: DataLoaders and Data Pipelines")
    foundation.add("🚧 Lesson 7: Training Loops and Optimization")

    # Performance optimization
    performance = lessons_tree.add("⚡ [bold yellow]Performance Optimization[/bold yellow]")
    performance.add("🚧 Lesson 8: Automatic Mixed Precision (AMP)")
    performance.add("🚧 Lesson 9: Model Compilation with torch.compile")
    performance.add("🚧 Lesson 10: Profiling and Performance Analysis")

    # Distributed training
    distributed = lessons_tree.add("🌐 [bold yellow]Distributed Training[/bold yellow]")
    distributed.add("🚧 Lesson 11: DistributedDataParallel (DDP)")
    distributed.add("🚧 Lesson 12: Fully Sharded Data Parallel (FSDP)")
    distributed.add("🚧 Lesson 13: Advanced Distributed Strategies")

    # Model optimization
    optimization = lessons_tree.add("🔧 [bold yellow]Model Optimization[/bold yellow]")
    optimization.add("🚧 Lesson 14: Quantization for Production")
    optimization.add("🚧 Lesson 15: Model Pruning and Sparsity")
    optimization.add("🚧 Lesson 16: Knowledge Distillation")

    # Modern architectures
    architectures = lessons_tree.add("🏗️ [bold yellow]Modern Architectures[/bold yellow]")
    architectures.add("🚧 Lesson 17: Transformer Architectures")
    architectures.add("🚧 Lesson 18: CNNs Best Practices")
    architectures.add("🚧 Lesson 19: RNNs and Sequence Modeling")

    # Production deployment
    production = lessons_tree.add("🚀 [bold yellow]Production Deployment[/bold yellow]")
    production.add("🚧 Lesson 20: Model Export and Deployment")
    production.add("✅ Lesson 21: Mobile & Edge with ExecutorTorch 🔥")
    production.add("🚧 Lesson 22: Custom Operators and C++ Extensions")

    # Advanced topics
    advanced = lessons_tree.add("🎯 [bold yellow]Advanced Topics[/bold yellow]")
    advanced.add("🚧 Lesson 23: Memory Optimization Techniques")
    advanced.add("🚧 Lesson 24: Production Best Practices")

    console.print(lessons_tree)
    console.print(
        "\n[bold green]✅ Available[/bold green] | [bold yellow]🚧 Coming Soon[/bold yellow]"
    )


@app.command()
def run(
    lesson: int = typer.Argument(..., help="Lesson number to run (1-24)"),
    interactive: bool = typer.Option(True, "--interactive/--batch", help="Interactive mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run a specific PyTorch lesson."""
    display_banner()

    if lesson < 1 or lesson > 24:
        console.print("[bold red]Error:[/bold red] Lesson number must be between 1 and 24")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading Lesson {lesson}...", total=None)

        # Run the appropriate lesson (import on-demand to avoid dependency issues)
        if lesson == 1:
            from pytorch_teaching.lessons import lesson_01_tensors
            progress.update(task, description="✓ Lesson 1 loaded!")
            progress.stop()
            lesson_01_tensors.run(interactive=interactive, verbose=verbose)
        elif lesson == 2:
            from pytorch_teaching.lessons import lesson_02_math_ops
            progress.update(task, description="✓ Lesson 2 loaded!")
            progress.stop()
            lesson_02_math_ops.run(interactive=interactive, verbose=verbose)
        elif lesson == 3:
            from pytorch_teaching.lessons import lesson_03_device_management
            progress.update(task, description="✓ Lesson 3 loaded!")
            progress.stop()
            lesson_03_device_management.run(interactive=interactive, verbose=verbose)
        elif lesson == 21:
            from pytorch_teaching.lessons import lesson_21_executorch
            progress.update(task, description="✓ Lesson 21 loaded!")
            progress.stop()
            lesson_21_executorch.run(interactive=interactive, verbose=verbose)
        else:
            progress.stop()
            console.print(
                f"[bold yellow]Lesson {lesson} is coming soon![/bold yellow] "
                "Stay tuned for updates."
            )
            raise typer.Exit(code=0)


@app.command()
def version():
    """Display version information."""
    console.print(f"[bold cyan]PyTorch Teaching[/bold cyan] version [bold green]{__version__}[/bold green]")

    try:
        import torch
        console.print(f"PyTorch version: [bold green]{torch.__version__}[/bold green]")
    except ImportError:
        console.print("PyTorch version: [bold yellow]Not installed[/bold yellow]")


@app.command()
def doctor():
    """Run a health check on the PyTorch installation."""
    display_banner()

    console.print("[bold cyan]Running PyTorch Health Check...[/bold cyan]\n")

    checks = Table(title="Health Check Results", show_header=True, header_style="bold magenta")
    checks.add_column("Check", style="cyan", width=30)
    checks.add_column("Status", style="green", width=15)
    checks.add_column("Details", style="yellow")

    # Check PyTorch installation
    torch_available = False
    try:
        import torch

        torch_available = True
        checks.add_row("PyTorch Installation", "✓ Pass", f"Version {torch.__version__}")
    except ImportError:
        checks.add_row("PyTorch Installation", "✗ Fail", "Not installed")
        console.print(checks)
        console.print("\n[bold yellow]Install PyTorch to continue: pip install torch[/bold yellow]")
        return

    # Check CUDA
    if torch.cuda.is_available():
        checks.add_row("CUDA Support", "✓ Pass", f"CUDA {torch.version.cuda}")
    else:
        checks.add_row("CUDA Support", "⚠ Warning", "Not available")

    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        checks.add_row("MPS Support", "✓ Pass", "Apple Silicon GPU available")
    else:
        checks.add_row("MPS Support", "⚠ Info", "Not available (non-Apple hardware)")

    # Check cuDNN
    if torch.cuda.is_available():
        cudnn_available = torch.backends.cudnn.is_available()
        checks.add_row(
            "cuDNN", "✓ Pass" if cudnn_available else "⚠ Warning", "Available" if cudnn_available else "Not available"
        )

    # Check basic tensor operations
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        checks.add_row("Basic Operations", "✓ Pass", "CPU tensor operations working")
    except Exception as e:
        checks.add_row("Basic Operations", "✗ Fail", str(e))

    # Check CUDA operations if available
    if torch.cuda.is_available():
        try:
            x = torch.randn(3, 3, device="cuda")
            y = torch.randn(3, 3, device="cuda")
            z = x + y
            checks.add_row("CUDA Operations", "✓ Pass", "GPU tensor operations working")
        except Exception as e:
            checks.add_row("CUDA Operations", "✗ Fail", str(e))

    console.print(checks)


@app.callback()
def callback():
    """PyTorch Teaching CLI - Professional learning tool for PyTorch."""
    pass


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
