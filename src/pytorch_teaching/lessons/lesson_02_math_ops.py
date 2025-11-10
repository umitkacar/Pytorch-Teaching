"""
Lesson 2: Mathematical Operations with Tensors.

This lesson covers essential mathematical operations and functions
available in PyTorch for tensor manipulation and computation.

Learning Objectives:
    - Master tensor creation functions (rand, randn, zeros, ones, etc.)
    - Perform element-wise and matrix operations
    - Understand tensor reshaping and views
    - Learn about in-place vs standard operations
    - Compute statistical operations (mean, std, cumsum, etc.)
"""

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


def demonstrate_tensor_creation():
    """Demonstrate various tensor creation functions."""
    console.print("\n[bold cyan]═══ Tensor Creation Functions ═══[/bold cyan]\n")

    # torch.rand - Uniform distribution [0, 1)
    rand_tensor = torch.rand(3, 3)
    console.print("[green]torch.rand(3, 3):[/green] Uniform distribution [0, 1)")
    console.print(f"{rand_tensor}\n")

    # torch.randn - Normal distribution (mean=0, std=1)
    randn_tensor = torch.randn(3, 3)
    console.print("[green]torch.randn(3, 3):[/green] Normal distribution (μ=0, σ=1)")
    console.print(f"{randn_tensor}\n")

    # torch.randint - Random integers
    randint_tensor = torch.randint(low=0, high=10, size=(3, 3))
    console.print("[green]torch.randint(0, 10, (3, 3)):[/green] Random integers [0, 10)")
    console.print(f"{randint_tensor}\n")

    # torch.randperm - Random permutation
    randperm_tensor = torch.randperm(10)
    console.print("[green]torch.randperm(10):[/green] Random permutation of [0, 10)")
    console.print(f"{randperm_tensor}\n")

    # torch.zeros - All zeros
    zeros_tensor = torch.zeros(3, 3)
    console.print("[green]torch.zeros(3, 3):[/green]")
    console.print(f"{zeros_tensor}\n")

    # torch.ones - All ones
    ones_tensor = torch.ones(3, 3)
    console.print("[green]torch.ones(3, 3):[/green]")
    console.print(f"{ones_tensor}\n")

    # torch.eye - Identity matrix
    eye_tensor = torch.eye(3)
    console.print("[green]torch.eye(3):[/green] Identity matrix")
    console.print(f"{eye_tensor}\n")

    # torch.arange - Range of values
    arange_tensor = torch.arange(0, 10, 2)
    console.print("[green]torch.arange(0, 10, 2):[/green] Range with step")
    console.print(f"{arange_tensor}\n")

    # torch.linspace - Linearly spaced values
    linspace_tensor = torch.linspace(0, 1, 5)
    console.print("[green]torch.linspace(0, 1, 5):[/green] 5 values from 0 to 1")
    console.print(f"{linspace_tensor}\n")


def demonstrate_element_wise_operations():
    """Demonstrate element-wise arithmetic operations."""
    console.print("\n[bold cyan]═══ Element-wise Operations ═══[/bold cyan]\n")

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    console.print(f"[green]Tensor X:[/green]\n{x}")
    console.print(f"[green]Tensor Y:[/green]\n{y}\n")

    # Addition
    console.print("[yellow]Addition (x + y):[/yellow]")
    console.print(f"{x + y}")
    console.print("[dim]Also: torch.add(x, y)[/dim]\n")

    # Subtraction
    console.print("[yellow]Subtraction (x - y):[/yellow]")
    console.print(f"{x - y}")
    console.print("[dim]Also: torch.sub(x, y)[/dim]\n")

    # Multiplication
    console.print("[yellow]Multiplication (x * y):[/yellow]")
    console.print(f"{x * y}")
    console.print("[dim]Also: torch.mul(x, y)[/dim]\n")

    # Division
    console.print("[yellow]Division (x / y):[/yellow]")
    console.print(f"{x / y}")
    console.print("[dim]Also: torch.div(x, y)[/dim]\n")

    # Power
    console.print("[yellow]Power (x ** 2):[/yellow]")
    console.print(f"{x ** 2}")
    console.print("[dim]Also: torch.pow(x, 2)[/dim]\n")

    # Square root
    console.print("[yellow]Square Root (torch.sqrt(x)):[/yellow]")
    console.print(f"{torch.sqrt(x)}\n")

    # Exponential
    console.print("[yellow]Exponential (torch.exp(x)):[/yellow]")
    console.print(f"{torch.exp(x)}\n")

    # Logarithm
    console.print("[yellow]Natural Log (torch.log(x)):[/yellow]")
    console.print(f"{torch.log(x)}\n")


def demonstrate_matrix_operations():
    """Demonstrate matrix operations."""
    console.print("\n[bold cyan]═══ Matrix Operations ═══[/bold cyan]\n")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    console.print(f"[green]Matrix A:[/green]\n{a}")
    console.print(f"[green]Matrix B:[/green]\n{b}\n")

    # Matrix multiplication
    console.print("[yellow]Matrix Multiplication (A @ B):[/yellow]")
    console.print(f"{a @ b}")
    console.print("[dim]Also: torch.matmul(a, b) or torch.mm(a, b)[/dim]\n")

    # Matrix transpose
    console.print("[yellow]Transpose (A.T):[/yellow]")
    console.print(f"{a.T}")
    console.print("[dim]Also: torch.transpose(a, 0, 1)[/dim]\n")

    # Matrix determinant
    console.print("[yellow]Determinant (torch.linalg.det(A)):[/yellow]")
    console.print(f"{torch.linalg.det(a)}\n")

    # Matrix inverse
    console.print("[yellow]Inverse (torch.linalg.inv(A)):[/yellow]")
    console.print(f"{torch.linalg.inv(a)}\n")

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(a)
    console.print("[yellow]Eigenvalues:[/yellow]")
    console.print(f"{eigenvalues}\n")


def demonstrate_reshaping():
    """Demonstrate tensor reshaping operations."""
    console.print("\n[bold cyan]═══ Tensor Reshaping ═══[/bold cyan]\n")

    x = torch.arange(12)
    console.print(f"[green]Original tensor:[/green] shape {x.shape}")
    console.print(f"{x}\n")

    # View - creates a view (shares memory)
    console.print("[yellow]x.view(3, 4):[/yellow] Reshape to 3x4")
    view_tensor = x.view(3, 4)
    console.print(f"{view_tensor}")
    console.print("[dim]Note: Shares memory with original[/dim]\n")

    # Reshape - similar to view but may copy
    console.print("[yellow]x.reshape(2, 6):[/yellow] Reshape to 2x6")
    reshape_tensor = x.reshape(2, 6)
    console.print(f"{reshape_tensor}\n")

    # Flatten
    console.print("[yellow]x.view(-1):[/yellow] Flatten to 1D")
    flat_tensor = x.view(-1)
    console.print(f"{flat_tensor}\n")

    # Unsqueeze - add dimension
    console.print("[yellow]x.unsqueeze(0):[/yellow] Add dimension at position 0")
    unsqueezed = x.unsqueeze(0)
    console.print(f"Shape: {unsqueezed.shape}")
    console.print(f"{unsqueezed}\n")

    # Squeeze - remove dimensions of size 1
    console.print("[yellow]unsqueezed.squeeze():[/yellow] Remove size-1 dimensions")
    squeezed = unsqueezed.squeeze()
    console.print(f"Shape: {squeezed.shape}")
    console.print(f"{squeezed}\n")


def demonstrate_statistical_operations():
    """Demonstrate statistical operations."""
    console.print("\n[bold cyan]═══ Statistical Operations ═══[/bold cyan]\n")

    x = torch.randn(3, 4)
    console.print(f"[green]Tensor X:[/green]\n{x}\n")

    # Mean
    console.print(f"[yellow]Mean (overall):[/yellow] {torch.mean(x)}")
    console.print(f"[yellow]Mean (dim=0):[/yellow] {torch.mean(x, dim=0)}")
    console.print(f"[yellow]Mean (dim=1):[/yellow] {torch.mean(x, dim=1)}\n")

    # Standard deviation
    console.print(f"[yellow]Std (overall):[/yellow] {torch.std(x)}")
    console.print(f"[yellow]Std (dim=0):[/yellow] {torch.std(x, dim=0)}\n")

    # Sum
    console.print(f"[yellow]Sum (overall):[/yellow] {torch.sum(x)}")
    console.print(f"[yellow]Sum (dim=1):[/yellow] {torch.sum(x, dim=1)}\n")

    # Min and Max
    console.print(f"[yellow]Min:[/yellow] {torch.min(x)}")
    console.print(f"[yellow]Max:[/yellow] {torch.max(x)}")
    console.print(f"[yellow]Argmin:[/yellow] {torch.argmin(x)}")
    console.print(f"[yellow]Argmax:[/yellow] {torch.argmax(x)}\n")

    # Cumulative sum
    console.print(f"[yellow]Cumsum (dim=0):[/yellow]\n{torch.cumsum(x, dim=0)}\n")

    # Sorting
    sorted_tensor, indices = torch.sort(x, dim=1)
    console.print(f"[yellow]Sorted (dim=1):[/yellow]\n{sorted_tensor}\n")


def demonstrate_inplace_operations():
    """Demonstrate in-place vs standard operations."""
    console.print("\n[bold cyan]═══ In-place vs Standard Operations ═══[/bold cyan]\n")

    console.print("[yellow]Standard operations (create new tensor):[/yellow]")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    console.print(f"[green]Before:[/green]\nx = {x}")
    result = x.add(y)
    console.print(f"[green]After x.add(y):[/green]\nx = {x}")
    console.print(f"result = {result}")
    console.print("[dim]Original x unchanged[/dim]\n")

    console.print("[yellow]In-place operations (modify original tensor):[/yellow]")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    console.print(f"[green]Before:[/green]\nx = {x}")
    x.add_(y)
    console.print(f"[green]After x.add_(y):[/green]\nx = {x}")
    console.print("[dim]Original x modified (note the underscore)[/dim]\n")

    console.print(
        "[bold red]⚠️ Warning:[/bold red] In-place operations can cause issues with autograd!",
    )
    console.print("Use them carefully, especially during training.\n")


def create_operations_reference():
    """Create a reference table of common operations."""
    table = Table(
        title="Common PyTorch Operations Reference", show_header=True, header_style="bold magenta",
    )

    table.add_column("Operation", style="cyan", width=25)
    table.add_column("Function", style="green", width=30)
    table.add_column("In-place", style="yellow", width=20)

    table.add_row("Addition", "torch.add(x, y) or x + y", "x.add_(y)")
    table.add_row("Subtraction", "torch.sub(x, y) or x - y", "x.sub_(y)")
    table.add_row("Multiplication", "torch.mul(x, y) or x * y", "x.mul_(y)")
    table.add_row("Division", "torch.div(x, y) or x / y", "x.div_(y)")
    table.add_row("Power", "torch.pow(x, n) or x ** n", "x.pow_(n)")
    table.add_row("Matrix Multiplication", "torch.matmul(x, y) or x @ y", "—")
    table.add_row("Transpose", "torch.transpose(x, 0, 1) or x.T", "x.t_()")
    table.add_row("Fill with value", "torch.full((3,3), value)", "x.fill_(value)")
    table.add_row("Zero out", "torch.zeros_like(x)", "x.zero_()")

    console.print("\n")
    console.print(table)


def run(interactive: bool = True, verbose: bool = False):
    """
    Run Lesson 2: Mathematical Operations with Tensors.

    Args:
        interactive: If True, wait for user input between sections
        verbose: If True, show additional details
    """
    console.print(
        Panel.fit(
            "[bold cyan]Lesson 2: Mathematical Operations with Tensors[/bold cyan]\n\n"
            "Master tensor creation, arithmetic, and statistical operations.",
            border_style="cyan",
        ),
    )

    # Tensor creation
    demonstrate_tensor_creation()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Element-wise operations
    demonstrate_element_wise_operations()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Matrix operations
    demonstrate_matrix_operations()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Reshaping
    demonstrate_reshaping()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Statistical operations
    demonstrate_statistical_operations()
    if interactive:
        input("\n[Press Enter to continue...]")

    # In-place operations
    demonstrate_inplace_operations()
    if interactive:
        input("\n[Press Enter to continue...]")

    # Reference table
    create_operations_reference()

    console.print("\n[bold green]✓ Lesson 2 Complete![/bold green]")
    console.print(
        "[yellow]Next:[/yellow] Run [cyan]pytorch-teach run 3[/cyan] for Lesson 3: Device Management\n",
    )


if __name__ == "__main__":
    run(interactive=False)
