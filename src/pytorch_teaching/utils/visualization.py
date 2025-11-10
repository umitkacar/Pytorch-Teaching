"""Visualization utilities for PyTorch Teaching lessons."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_tensor_2d(
    tensor: torch.Tensor, title: str = "Tensor Visualization", cmap: str = "viridis",
):
    """
    Visualize a 2D tensor as a heatmap.

    Args:
        tensor: 2D PyTorch tensor
        title: Plot title
        cmap: Matplotlib colormap name
    """
    if tensor.dim() != 2:
        msg = f"Expected 2D tensor, got {tensor.dim()}D"
        raise ValueError(msg)

    plt.figure(figsize=(8, 6))
    sns.heatmap(tensor.detach().cpu().numpy(), annot=True, fmt=".2f", cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: list[float],
    val_losses: Optional[list[float]] = None,
    train_accs: Optional[list[float]] = None,
    val_accs: Optional[list[float]] = None,
):
    """
    Plot training history (loss and accuracy).

    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        train_accs: Optional list of training accuracies
        val_accs: Optional list of validation accuracies
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot losses
    axes[0].plot(train_losses, label="Train Loss", marker="o")
    if val_losses:
        axes[0].plot(val_losses, label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies
    if train_accs:
        axes[1].plot(train_accs, label="Train Acc", marker="o")
    if val_accs:
        axes[1].plot(val_accs, label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_model_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[list[str]] = None,
    num_images: int = 16,
):
    """
    Visualize model predictions on a grid of images.

    Args:
        images: Batch of images [B, C, H, W]
        predictions: Model predictions [B, num_classes]
        labels: Ground truth labels [B]
        class_names: Optional list of class names
        num_images: Number of images to display
    """
    images = images[:num_images].cpu()
    predictions = predictions[:num_images].cpu()
    labels = labels[:num_images].cpu()

    pred_classes = torch.argmax(predictions, dim=1)

    rows = int(np.sqrt(num_images))
    cols = (num_images + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for idx in range(num_images):
        ax = axes[idx]

        # Convert image to displayable format
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize

        ax.imshow(img)
        ax.axis("off")

        pred_class = pred_classes[idx].item()
        true_class = labels[idx].item()

        if class_names:
            pred_label = class_names[pred_class]
            true_label = class_names[true_class]
        else:
            pred_label = str(pred_class)
            true_label = str(true_class)

        color = "green" if pred_class == true_class else "red"
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_gradient_flow(named_parameters):
    """
    Plot gradient flow through the network during training.

    Args:
        named_parameters: Model's named_parameters()
    """
    avg_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            avg_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, label="Max gradient")
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.5, label="Avg gradient")
    plt.hlines(0, 0, len(avg_grads) + 1, linewidth=2, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001)
    plt.xlabel("Layers")
    plt.ylabel("Gradient magnitude")
    plt.title("Gradient Flow")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, title: str = "Attention Weights"):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Attention weight tensor [seq_len, seq_len]
        title: Plot title
    """
    attention_weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=False, cmap="Blues", square=True, cbar=True)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(title)
    plt.tight_layout()
    plt.show()
