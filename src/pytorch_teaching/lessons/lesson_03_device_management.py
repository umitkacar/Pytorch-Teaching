"""
Lesson 3: Device Management and Data Type Conversions.

This lesson covers data conversions between Python, NumPy, and PyTorch,
as well as moving tensors between CPU and CUDA (GPU) devices.

Learning Objectives:
    - Convert between Python lists, NumPy arrays, and PyTorch tensors
    - Understand memory sharing vs copying
    - Manage tensors on different devices (CPU, CUDA, MPS)
    - Optimize data transfers for performance
    - Handle device compatibility and errors
"""

import sys

import numpy as np
import torch
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_device_availability():
    """Check and display available compute devices."""
    console.print("\n[bold cyan]═══ Available Devices ═══[/bold cyan]\n")

    devices_table = Table(show_header=True, header_style="bold magenta")
    devices_table.add_column("Device", style="cyan", width=20)
    devices_table.add_column("Available", style="green", width=15)
    devices_table.add_column("Details", style="yellow")

    # CPU (always available)
    devices_table.add_row("CPU", "✓ Yes", f"Default device")

    # CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        devices_table.add_row(
            "CUDA", "✓ Yes", f"{device_count} GPU(s): {device_name}\nCUDA {cuda_version}"
        )
    else:
        devices_table.add_row("CUDA", "✗ No", "Install CUDA-enabled PyTorch")

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        devices_table.add_row("MPS", "✓ Yes", "Apple Silicon GPU acceleration")
    elif sys.platform == "darwin":
        devices_table.add_row("MPS", "✗ No", "Requires macOS 12.3+ and Apple Silicon")
    else:
        devices_table.add_row("MPS", "— N/A", "Not on macOS")

    console.print(devices_table)
    console.print()

    # Recommended device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[bold green]✓ Recommended device:[/bold green] {device}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print(f"[bold green]✓ Recommended device:[/bold green] {device}")
    else:
        device = torch.device("cpu")
        console.print(f"[bold yellow]⚠ Using device:[/bold yellow] {device}")

    return device


def demonstrate_numpy_to_pytorch():
    """Demonstrate converting NumPy arrays to PyTorch tensors."""
    console.print("\n[bold cyan]═══ NumPy → PyTorch Conversion ═══[/bold cyan]\n")

    # Create NumPy array
    numpy_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    console.print(f"[green]NumPy array:[/green]\n{numpy_array}")
    console.print(f"Type: {type(numpy_array)}, dtype: {numpy_array.dtype}\n")

    # Method 1: torch.from_numpy() - SHARES MEMORY
    console.print("[yellow]Method 1: torch.from_numpy() - Shares memory[/yellow]")
    tensor_shared = torch.from_numpy(numpy_array)
    console.print(f"Tensor: {tensor_shared}")
    console.print(f"Type: {type(tensor_shared)}, dtype: {tensor_shared.dtype}\n")

    # Demonstrate memory sharing
    console.print("[bold yellow]Testing memory sharing:[/bold yellow]")
    console.print(f"Before: numpy_array[0,0] = {numpy_array[0,0]}")
    numpy_array[0, 0] = 99
    console.print(f"After modifying numpy_array[0,0] = 99:")
    console.print(f"numpy_array[0,0] = {numpy_array[0,0]}")
    console.print(f"tensor_shared[0,0] = {tensor_shared[0,0]}")
    console.print("[dim]Both changed! Memory is shared.[/dim]\n")

    # Method 2: torch.from_numpy().clone() - COPIES MEMORY
    numpy_array[0, 0] = 1  # Reset
    console.print("[yellow]Method 2: torch.from_numpy().clone() - Copies memory[/yellow]")
    tensor_copied = torch.from_numpy(numpy_array).clone()
    console.print(f"Tensor: {tensor_copied}\n")

    console.print("[bold yellow]Testing memory copying:[/bold yellow]")
    console.print(f"Before: numpy_array[0,0] = {numpy_array[0,0]}")
    numpy_array[0, 0] = 77
    console.print(f"After modifying numpy_array[0,0] = 77:")
    console.print(f"numpy_array[0,0] = {numpy_array[0,0]}")
    console.print(f"tensor_copied[0,0] = {tensor_copied[0,0]}")
    console.print("[dim]Only numpy_array changed! Memory is copied.[/dim]\n")


def demonstrate_pytorch_to_numpy():
    """Demonstrate converting PyTorch tensors to NumPy arrays."""
    console.print("\n[bold cyan]═══ PyTorch → NumPy Conversion ═══[/bold cyan]\n")

    # Create PyTorch tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    console.print(f"[green]PyTorch tensor:[/green]\n{tensor}\n")

    # Method 1: .numpy() - SHARES MEMORY (CPU only)
    console.print("[yellow]Method 1: tensor.numpy() - Shares memory (CPU only)[/yellow]")
    numpy_shared = tensor.numpy()
    console.print(f"NumPy array: {numpy_shared}\n")

    # Demonstrate memory sharing
    console.print("[bold yellow]Testing memory sharing:[/bold yellow]")
    tensor[0, 0] = 99
    console.print(f"After modifying tensor[0,0] = 99:")
    console.print(f"tensor[0,0] = {tensor[0,0]}")
    console.print(f"numpy_shared[0,0] = {numpy_shared[0,0]}")
    console.print("[dim]Both changed! Memory is shared.[/dim]\n")

    # Method 2: .numpy().copy() - COPIES MEMORY
    tensor[0, 0] = 1.0  # Reset
    console.print("[yellow]Method 2: tensor.numpy().copy() - Copies memory[/yellow]")
    numpy_copied = tensor.numpy().copy()

    tensor[0, 0] = 88
    console.print(f"After modifying tensor[0,0] = 88:")
    console.print(f"tensor[0,0] = {tensor[0,0]}")
    console.print(f"numpy_copied[0,0] = {numpy_copied[0,0]}")
    console.print("[dim]Only tensor changed! Memory is copied.[/dim]\n")


def demonstrate_python_list_conversions():
    """Demonstrate conversions with Python lists."""
    console.print("\n[bold cyan]═══ Python Lists ↔ PyTorch Tensors ═══[/bold cyan]\n")

    # Python list to tensor
    console.print("[yellow]Python list → Tensor:[/yellow]")
    python_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    console.print(f"Python list: {python_list}")

    tensor = torch.tensor(python_list)
    console.print(f"torch.tensor(): {tensor}\n")

    # Tensor to Python list
    console.print("[yellow]Tensor → Python list:[/yellow]")
    result_list = tensor.tolist()
    console.print(f"tensor.tolist(): {result_list}\n")

    # Extract single value
    console.print("[yellow]Extract single value:[/yellow]")
    value = tensor[0, 0].item()
    console.print(f"tensor[0, 0].item(): {value} (type: {type(value)})\n")

    console.print("[dim]Note: .tolist() always creates a copy[/dim]\n")


def demonstrate_device_transfer():
    """Demonstrate moving tensors between devices."""
    console.print("\n[bold cyan]═══ Device Transfer (CPU ↔ GPU) ═══[/bold cyan]\n")

    # Create tensor on CPU
    cpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    console.print(f"[green]CPU Tensor:[/green] device={cpu_tensor.device}")
    console.print(f"{cpu_tensor}\n")

    if torch.cuda.is_available():
        # Move to CUDA
        console.print("[yellow]Moving to CUDA:[/yellow]")
        cuda_tensor = cpu_tensor.to("cuda")
        console.print(f"cuda_tensor = cpu_tensor.to('cuda')")
        console.print(f"Device: {cuda_tensor.device}\n")

        # Alternative methods
        console.print("[yellow]Alternative methods:[/yellow]")
        console.print("1. cpu_tensor.cuda()")
        console.print("2. cpu_tensor.to(device='cuda:0')")
        console.print("3. cpu_tensor.to(torch.device('cuda'))\n")

        # Move back to CPU
        console.print("[yellow]Moving back to CPU:[/yellow]")
        back_to_cpu = cuda_tensor.to("cpu")
        console.print(f"back_to_cpu = cuda_tensor.to('cpu')")
        console.print(f"Device: {back_to_cpu.device}\n")

        # Alternative method
        console.print("[yellow]Alternative: cuda_tensor.cpu()[/yellow]\n")

        # Create directly on GPU
        console.print("[yellow]Create directly on GPU:[/yellow]")
        direct_cuda = torch.randn(2, 2, device="cuda")
        console.print(f"torch.randn(2, 2, device='cuda')")
        console.print(f"Device: {direct_cuda.device}\n")

    else:
        console.print("[bold yellow]⚠ CUDA not available[/bold yellow]")
        console.print("Example code that would run with CUDA:")
        console.print("  cuda_tensor = cpu_tensor.to('cuda')")
        console.print("  back_to_cpu = cuda_tensor.to('cpu')\n")


def demonstrate_dtype_conversions():
    """Demonstrate data type conversions."""
    console.print("\n[bold cyan]═══ Data Type Conversions ═══[/bold cyan]\n")

    # Create tensor with default dtype
    tensor = torch.tensor([1.0, 2.0, 3.0])
    console.print(f"[green]Original tensor:[/green] dtype={tensor.dtype}")
    console.print(f"{tensor}\n")

    # Convert to different dtypes
    conversions = {
        "int32": tensor.to(torch.int32),
        "int64": tensor.to(torch.int64),
        "float16": tensor.to(torch.float16),
        "float32": tensor.to(torch.float32),
        "float64": tensor.to(torch.float64),
    }

    dtype_table = Table(show_header=True, header_style="bold magenta")
    dtype_table.add_column("Method", style="cyan", width=30)
    dtype_table.add_column("Result dtype", style="green", width=20)
    dtype_table.add_column("Values", style="yellow")

    for dtype_name, converted in conversions.items():
        method = f"tensor.to(torch.{dtype_name})"
        dtype_table.add_row(method, str(converted.dtype), str(converted.tolist()))

    console.print(dtype_table)
    console.print()

    # Alternative methods
    console.print("[yellow]Alternative conversion methods:[/yellow]")
    console.print("  tensor.int()    → torch.int32")
    console.print("  tensor.long()   → torch.int64")
    console.print("  tensor.float()  → torch.float32")
    console.print("  tensor.double() → torch.float64")
    console.print("  tensor.half()   → torch.float16\n")


def demonstrate_best_practices():
    """Demonstrate best practices for device management."""
    console.print("\n[bold cyan]═══ Best Practices ═══[/bold cyan]\n")

    console.print("[bold yellow]1. Device-agnostic code:[/bold yellow]")
    console.print(
        """
[green]device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randn(3, 3, device=device)
model.to(device)[/green]
"""
    )

    console.print("[bold yellow]2. Avoid unnecessary transfers:[/bold yellow]")
    console.print(
        """
[red]# Bad: Multiple transfers[/red]
[dim]for i in range(1000):
    x = x.to('cuda')
    y = process(x)
    z = y.to('cpu')[/dim]

[green]# Good: Transfer once[/green]
x = x.to('cuda')
for i in range(1000):
    y = process(x)
z = y.to('cpu')
"""
    )

    console.print("[bold yellow]3. Check device compatibility:[/bold yellow]")
    console.print(
        """
[green]# Ensure tensors are on same device
if x.device != y.device:
    y = y.to(x.device)
result = x + y[/green]
"""
    )

    console.print("[bold yellow]4. Memory management:[/bold yellow]")
    console.print(
        """
[green]# Clear GPU cache
torch.cuda.empty_cache()

# Pin memory for faster transfers
tensor = torch.randn(1000, 1000, pin_memory=True)
tensor = tensor.to('cuda', non_blocking=True)[/green]
"""
    )


def create_conversion_reference():
    """Create a reference table for conversions."""
    table = Table(
        title="Conversion Reference Guide", show_header=True, header_style="bold magenta"
    )

    table.add_column("From → To", style="cyan", width=25)
    table.add_column("Method", style="green", width=35)
    table.add_column("Memory", style="yellow", width=15)

    table.add_row("NumPy → PyTorch", "torch.from_numpy(arr)", "Shared")
    table.add_row("NumPy → PyTorch", "torch.from_numpy(arr).clone()", "Copied")
    table.add_row("PyTorch → NumPy", "tensor.numpy()", "Shared (CPU)")
    table.add_row("PyTorch → NumPy", "tensor.numpy().copy()", "Copied")
    table.add_row("List → PyTorch", "torch.tensor(list)", "Copied")
    table.add_row("PyTorch → List", "tensor.tolist()", "Copied")
    table.add_row("Tensor → Scalar", "tensor.item()", "Copied")
    table.add_row("CPU → CUDA", "tensor.to('cuda')", "Copied")
    table.add_row("CUDA → CPU", "tensor.to('cpu')", "Copied")

    console.print("\n")
    console.print(table)


def run(interactive: bool = True, verbose: bool = False):
    """
    Run Lesson 3: Device Management and Data Type Conversions.

    Args:
        interactive: If True, wait for user input between sections
        verbose: If True, show additional details
    """
    console.print(
        Panel.fit(
            "[bold cyan]Lesson 3: Device Management & Data Type Conversions[/bold cyan]\n\n"
            "Master data conversions and device management for optimal performance.",
            border_style="cyan",
        )
    )

    # Check devices
    device = check_device_availability()
    if interactive:
        input("\n[Press Enter to continue...]")

    # NumPy to PyTorch
    demonstrate_numpy_to_pytorch()
    if interactive:
        input("\n[Press Enter to continue...]")

    # PyTorch to NumPy
    demonstrate_pytorch_to_numpy()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Python lists
    demonstrate_python_list_conversions()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Device transfer
    demonstrate_device_transfer()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Dtype conversions
    demonstrate_dtype_conversions()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Best practices
    demonstrate_best_practices()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Reference table
    create_conversion_reference()

    console.print("\n[bold green]✓ Lesson 3 Complete![/bold green]")
    console.print(
        "[yellow]Next:[/yellow] More lessons coming soon! Check [cyan]pytorch-teach list[/cyan]\n"
    )


if __name__ == "__main__":
    run(interactive=False)
