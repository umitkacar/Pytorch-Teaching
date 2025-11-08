"""
PyTorch Teaching - Professional CLI Tool for Learning PyTorch.

A comprehensive, production-ready PyTorch learning resource covering everything
from basic tensor operations to advanced distributed training, quantization,
ExecutorTorch deployment, and modern architectures.

Author: PyTorch Teaching Team
License: MIT
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "PyTorch Teaching Team"
__license__ = "MIT"

# Lazy imports to avoid dependency issues at import time
# Import submodules on demand to prevent circular imports and missing dependencies
__all__ = ["__version__", "__author__", "__license__"]
