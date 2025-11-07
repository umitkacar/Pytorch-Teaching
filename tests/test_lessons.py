"""Tests for PyTorch Teaching lessons."""

import pytest
import torch

from pytorch_teaching.lessons import (
    lesson_01_tensors,
    lesson_02_math_ops,
    lesson_03_device_management,
)


class TestLesson01:
    """Tests for Lesson 1: Tensor Fundamentals."""

    def test_lesson_runs(self):
        """Test that lesson 1 runs without errors."""
        # This should not raise any exceptions
        lesson_01_tensors.run(interactive=False, verbose=False)

    def test_scalar_creation(self):
        """Test scalar tensor creation."""
        scalar = torch.tensor(42)
        assert scalar.dim() == 0
        assert scalar.item() == 42

    def test_vector_creation(self):
        """Test vector tensor creation."""
        vector = torch.tensor([1, 2, 3, 4, 5])
        assert vector.dim() == 1
        assert vector.shape == (5,)

    def test_matrix_creation(self):
        """Test matrix tensor creation."""
        matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert matrix.dim() == 2
        assert matrix.shape == (2, 3)

    def test_3d_tensor_creation(self):
        """Test 3D tensor creation."""
        tensor_3d = torch.randn(2, 3, 4)
        assert tensor_3d.dim() == 3
        assert tensor_3d.shape == (2, 3, 4)


class TestLesson02:
    """Tests for Lesson 2: Mathematical Operations."""

    def test_lesson_runs(self):
        """Test that lesson 2 runs without errors."""
        lesson_02_math_ops.run(interactive=False, verbose=False)

    def test_element_wise_addition(self):
        """Test element-wise addition."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        result = x + y
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected)

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = a @ b
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert torch.allclose(result, expected)

    def test_tensor_reshaping(self):
        """Test tensor reshaping."""
        x = torch.arange(12)
        reshaped = x.view(3, 4)
        assert reshaped.shape == (3, 4)
        assert reshaped[0, 0] == 0
        assert reshaped[2, 3] == 11

    def test_statistical_operations(self):
        """Test statistical operations."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.mean(x).item() == 3.0
        assert torch.sum(x).item() == 15.0
        assert torch.min(x).item() == 1.0
        assert torch.max(x).item() == 5.0


class TestLesson03:
    """Tests for Lesson 3: Device Management."""

    def test_lesson_runs(self):
        """Test that lesson 3 runs without errors."""
        lesson_03_device_management.run(interactive=False, verbose=False)

    def test_cpu_device(self):
        """Test CPU device."""
        tensor = torch.randn(3, 3)
        assert tensor.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device."""
        tensor = torch.randn(3, 3, device="cuda")
        assert tensor.device.type == "cuda"

    def test_numpy_conversion(self):
        """Test NumPy conversion."""
        import numpy as np

        tensor = torch.tensor([1.0, 2.0, 3.0])
        numpy_array = tensor.numpy()
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (3,)

    def test_python_list_conversion(self):
        """Test Python list conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        python_list = tensor.tolist()
        assert isinstance(python_list, list)
        assert python_list == [[1.0, 2.0], [3.0, 4.0]]

    def test_dtype_conversion(self):
        """Test dtype conversion."""
        tensor_float = torch.tensor([1.0, 2.0, 3.0])
        tensor_int = tensor_float.to(torch.int32)
        assert tensor_int.dtype == torch.int32
        assert tensor_int.tolist() == [1, 2, 3]


class TestUtils:
    """Tests for utility functions."""

    def test_helpers_import(self):
        """Test that helpers can be imported."""
        from pytorch_teaching.utils import helpers

        assert hasattr(helpers, "set_seed")
        assert hasattr(helpers, "get_device")

    def test_visualization_import(self):
        """Test that visualization can be imported."""
        from pytorch_teaching.utils import visualization

        assert hasattr(visualization, "plot_tensor_2d")

    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        from pytorch_teaching.utils.helpers import set_seed

        set_seed(42)
        x1 = torch.randn(5)

        set_seed(42)
        x2 = torch.randn(5)

        assert torch.allclose(x1, x2)

    def test_get_device(self):
        """Test device selection."""
        from pytorch_teaching.utils.helpers import get_device

        device = get_device()
        assert isinstance(device, torch.device)
