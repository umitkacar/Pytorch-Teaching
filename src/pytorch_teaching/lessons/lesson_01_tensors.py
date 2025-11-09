"""
Lesson 1: Tensor Fundamentals.

This lesson covers the foundational concepts of tensors in PyTorch,
including scalars, vectors, matrices, and multidimensional tensors.

Learning Objectives:
    - Understand what tensors are and their mathematical foundations
    - Compare Python lists, NumPy arrays, and PyTorch tensors
    - Create and manipulate tensors
    - Understand tensor shapes and dimensions
    - Work with different data types in tensors
"""

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def explain_scalars():
    """Demonstrate scalar values in Python, NumPy, and PyTorch."""
    console.print("\n[bold cyan]═══ Scalars ═══[/bold cyan]")
    console.print("[yellow]A scalar is a single number (zero-dimensional).[/yellow]\n")

    # Python scalar
    python_scalar = 42
    console.print(
        f"[green]Python scalar:[/green] {python_scalar} (type: {type(python_scalar).__name__})",
    )

    # NumPy scalar
    numpy_scalar = np.array(42)
    console.print(
        f"[green]NumPy scalar:[/green] {numpy_scalar} "
        f"(shape: {numpy_scalar.shape}, dtype: {numpy_scalar.dtype})",
    )

    # PyTorch scalar
    torch_scalar = torch.tensor(42)
    console.print(
        f"[green]PyTorch scalar:[/green] {torch_scalar} "
        f"(shape: {torch_scalar.shape}, dtype: {torch_scalar.dtype})",
    )

    return python_scalar, numpy_scalar, torch_scalar


def explain_vectors():
    """Demonstrate vector values in Python, NumPy, and PyTorch."""
    console.print("\n[bold cyan]═══ Vectors ═══[/bold cyan]")
    console.print("[yellow]A vector is a one-dimensional array (first-order tensor).[/yellow]\n")

    # Python vector (list)
    python_vector = [1, 2, 3, 4, 5]
    console.print(f"[green]Python list:[/green] {python_vector}")

    # NumPy vector
    numpy_vector = np.array([1, 2, 3, 4, 5])
    console.print(
        f"[green]NumPy vector:[/green] {numpy_vector} "
        f"(shape: {numpy_vector.shape}, dtype: {numpy_vector.dtype})",
    )

    # PyTorch vector
    torch_vector = torch.tensor([1, 2, 3, 4, 5])
    console.print(
        f"[green]PyTorch vector:[/green] {torch_vector} "
        f"(shape: {torch_vector.shape}, dtype: {torch_vector.dtype})",
    )

    return python_vector, numpy_vector, torch_vector


def explain_matrices():
    """Demonstrate matrix values in Python, NumPy, and PyTorch."""
    console.print("\n[bold cyan]═══ Matrices ═══[/bold cyan]")
    console.print("[yellow]A matrix is a two-dimensional array (second-order tensor).[/yellow]\n")

    # Python matrix (nested lists)
    python_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    console.print("[green]Python nested list:[/green]")
    for row in python_matrix:
        console.print(f"  {row}")

    # NumPy matrix
    numpy_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    console.print(
        f"\n[green]NumPy matrix:[/green]\n{numpy_matrix}"
        f"\n(shape: {numpy_matrix.shape}, dtype: {numpy_matrix.dtype})",
    )

    # PyTorch matrix
    torch_matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    console.print(
        f"\n[green]PyTorch matrix:[/green]\n{torch_matrix}"
        f"\n(shape: {torch_matrix.shape}, dtype: {torch_matrix.dtype})",
    )

    return python_matrix, numpy_matrix, torch_matrix


def explain_tensors():
    """Demonstrate high-dimensional tensors."""
    console.print("\n[bold cyan]═══ Tensors (3D and Higher) ═══[/bold cyan]")
    console.print(
        "[yellow]A tensor is a generalization of matrices to multiple dimensions.[/yellow]\n",
    )

    # 3D tensor
    tensor_3d = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ],
    )

    console.print("[green]3D Tensor:[/green]")
    console.print(f"Shape: {tensor_3d.shape}")
    console.print(f"Dimensions: {tensor_3d.dim()}")
    console.print(f"Size: {tensor_3d.size()}")
    console.print(f"Total elements: {tensor_3d.numel()}")
    console.print(f"\n{tensor_3d}")

    # Image tensor example
    console.print("\n[bold yellow]💡 Real-world Example: Image Batches[/bold yellow]")
    batch_size, channels, height, width = 8, 3, 224, 224
    image_batch = torch.randn(batch_size, channels, height, width)

    console.print(f"[green]Image batch tensor shape:[/green] {image_batch.shape}")
    console.print(f"  - Batch size: {batch_size} images")
    console.print(f"  - Channels: {channels} (RGB)")
    console.print(f"  - Height: {height} pixels")
    console.print(f"  - Width: {width} pixels")
    console.print(f"  - Total elements: {image_batch.numel():,}")

    return tensor_3d, image_batch


def create_comparison_table():
    """Create a comparison table of Python, NumPy, and PyTorch."""
    table = Table(title="Python vs NumPy vs PyTorch", show_header=True, header_style="bold magenta")

    table.add_column("Feature", style="cyan", width=20)
    table.add_column("Python", style="yellow", width=20)
    table.add_column("NumPy", style="green", width=20)
    table.add_column("PyTorch", style="red", width=20)

    table.add_row("Data Structure", "list, int, float", "ndarray", "Tensor")
    table.add_row("GPU Support", "❌ No", "❌ No", "✅ Yes (CUDA/ROCm)")
    table.add_row("Autograd", "❌ No", "❌ No", "✅ Yes")
    table.add_row("Broadcasting", "❌ No", "✅ Yes", "✅ Yes")
    table.add_row("Performance", "⚠️ Slow", "✅ Fast", "✅ Fast")
    table.add_row("ML/DL Support", "❌ No", "⚠️ Limited", "✅ Extensive")
    table.add_row("Optimization", "❌ No", "❌ No", "✅ Yes (JIT, quantization)")

    console.print("\n")
    console.print(table)


def demonstrate_tensor_properties():
    """Demonstrate important tensor properties and operations."""
    console.print("\n[bold cyan]═══ Tensor Properties ═══[/bold cyan]\n")

    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    properties_table = Table(show_header=True, header_style="bold magenta")
    properties_table.add_column("Property", style="cyan", width=20)
    properties_table.add_column("Value", style="green", width=40)

    properties_table.add_row("Tensor", str(tensor.tolist()))
    properties_table.add_row("Shape", str(tensor.shape))
    properties_table.add_row("Size", str(tensor.size()))
    properties_table.add_row("Dimension", str(tensor.dim()))
    properties_table.add_row("Dtype", str(tensor.dtype))
    properties_table.add_row("Device", str(tensor.device))
    properties_table.add_row("Requires Grad", str(tensor.requires_grad))
    properties_table.add_row("Is Contiguous", str(tensor.is_contiguous()))
    properties_table.add_row("Number of Elements", str(tensor.numel()))

    console.print(properties_table)


def demonstrate_indexing():
    """Demonstrate tensor indexing and slicing."""
    console.print("\n[bold cyan]═══ Tensor Indexing & Slicing ═══[/bold cyan]\n")

    tensor = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
    )

    console.print(f"[green]Original tensor shape:[/green] {tensor.shape}")
    console.print(f"{tensor}\n")

    console.print(f"[yellow]tensor[0]:[/yellow] {tensor[0]}")  # First 2D slice
    console.print(f"[yellow]tensor[0, 0]:[/yellow] {tensor[0, 0]}")  # First row
    console.print(f"[yellow]tensor[0, 0, 0]:[/yellow] {tensor[0, 0, 0]}")  # Single element
    console.print(f"[yellow]tensor[:, :, 0]:[/yellow] {tensor[:, :, 0]}")  # First column of all


def run(interactive: bool = True, verbose: bool = False):
    """
    Run Lesson 1: Tensor Fundamentals.

    Args:
        interactive: If True, wait for user input between sections
        verbose: If True, show additional details
    """
    console.print(
        Panel.fit(
            "[bold cyan]Lesson 1: Tensor Fundamentals[/bold cyan]\n\n"
            "Understanding scalars, vectors, matrices, and tensors.\n"
            "Comparing Python, NumPy, and PyTorch implementations.",
            border_style="cyan",
        ),
    )

    # Scalars
    explain_scalars()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Vectors
    explain_vectors()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Matrices
    explain_matrices()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Tensors
    explain_tensors()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Comparison
    create_comparison_table()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Properties
    demonstrate_tensor_properties()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Indexing
    demonstrate_indexing()

    console.print("\n[bold green]✓ Lesson 1 Complete![/bold green]")
    console.print(
        "[yellow]Next:[/yellow] Run [cyan]pytorch-teach run 2[/cyan] for Lesson 2: Mathematical Operations\n",
    )


if __name__ == "__main__":
    run(interactive=False)
