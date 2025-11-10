# 🚀 Installation Guide

Complete installation instructions for PyTorch Teaching CLI.

## ⚡ Quick Start (Recommended)

### Option 1: Full Installation with PyTorch

```bash
# Clone the repository
git clone https://github.com/umitkacar/Pytorch-Teaching.git
cd Pytorch-Teaching

# Install with all dependencies (includes PyTorch, numpy, etc.)
pip install -e .
```

**Note:** PyTorch download is ~900MB, installation may take 5-10 minutes.

### Option 2: CLI First, Dependencies Later

```bash
# Install just the CLI tools (fast, ~5 seconds)
pip install -e . --no-deps
pip install typer rich

# Test the CLI immediately
pytorch-teach --help
pytorch-teach list-lessons

# Install full dependencies when ready
pip install torch torchvision torchaudio numpy matplotlib pandas seaborn pillow tqdm
```

## 📋 Prerequisites

- Python 3.9, 3.10, 3.11, or 3.12
- pip >= 23.0
- Git

## 🔧 Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/umitkacar/Pytorch-Teaching.git
cd Pytorch-Teaching
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Package

#### Production Installation

```bash
pip install -e .
```

#### Development Installation

```bash
pip install -e ".[dev]"
```

This installs:
- All production dependencies
- Development tools (pytest, black, ruff, mypy, etc.)
- Pre-commit hooks
- Testing frameworks

## ✅ Verify Installation

Test that the CLI is working correctly:

```bash
# Show help
pytorch-teach --help

# Show version
pytorch-teach version

# List all lessons
pytorch-teach list-lessons

# Show system info
pytorch-teach info

# Run health check
pytorch-teach doctor
```

### Expected Output Examples

**Without PyTorch installed:**
```bash
$ pytorch-teach version
PyTorch Teaching version 2.0.0
PyTorch version: Not installed
```

**With PyTorch installed:**
```bash
$ pytorch-teach version
PyTorch Teaching version 2.0.0
PyTorch version: 2.9.0
```

## 🏃 Running Lessons

```bash
# Run Lesson 1: Tensor Fundamentals
pytorch-teach run 1

# Run Lesson 2: Mathematical Operations
pytorch-teach run 2

# Run Lesson 3: Device Management
pytorch-teach run 3

# Run Lesson 21: ExecutorTorch (Mobile/Edge AI)
pytorch-teach run 21

# Run in batch mode (non-interactive)
pytorch-teach run 1 --batch

# Run with verbose output
pytorch-teach run 1 --verbose
```

## 🎯 Available Lessons

### Foundation (Lessons 1-7)
- ✅ **Lesson 1:** Tensor Fundamentals
- ✅ **Lesson 2:** Mathematical Operations
- ✅ **Lesson 3:** Device Management (CPU/CUDA)
- 🚧 **Lesson 4:** Autograd and Automatic Differentiation
- 🚧 **Lesson 5:** Building Neural Networks with nn.Module
- 🚧 **Lesson 6:** DataLoaders and Data Pipelines
- 🚧 **Lesson 7:** Training Loops and Optimization

### Production Deployment (Lessons 20-22)
- 🚧 **Lesson 20:** Model Export and Deployment
- ✅ **Lesson 21:** Mobile & Edge with ExecutorTorch 🔥
- 🚧 **Lesson 22:** Custom Operators and C++ Extensions

_(And 17 more advanced lessons - see full list with `pytorch-teach list-lessons`)_

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch

```bash
pip install torch torchvision torchaudio
```

Or use CPU-only version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "CUDA not available"

**Solution:** This is normal if you don't have an NVIDIA GPU. The CLI will use CPU automatically.

To install CUDA-enabled PyTorch:
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "command not found: pytorch-teach"

**Solutions:**
1. Ensure you activated your virtual environment
2. Try running with Python module: `python -m pytorch_teaching.cli --help`
3. Reinstall: `pip install -e .`

### Issue: Slow installation

**Cause:** PyTorch is a large package (~900MB)

**Solutions:**
1. Use Option 2 (CLI First, Dependencies Later) from Quick Start
2. Install CPU-only PyTorch (smaller): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Be patient - it's a one-time download

## 🧪 Testing Installation

```bash
# Run health check
pytorch-teach doctor

# Expected output:
# ✓ PyTorch Installation: Pass
# ✓ CUDA Support: Pass (or Warning if no GPU)
# ✓ Basic Operations: Pass
```

## 📦 Dependencies

### Required
- `typer[all]>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Vision utilities
- `torchaudio>=2.0.0` - Audio utilities
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Plotting
- `pandas>=2.0.0` - Data manipulation
- `seaborn>=0.12.0` - Statistical visualization
- `pillow>=10.0.0` - Image processing
- `tqdm>=4.65.0` - Progress bars

### Optional (Development)
- `pytest>=7.4.0` - Testing
- `black>=23.7.0` - Code formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.5.0` - Type checking
- `bandit>=1.7.5` - Security scanning

## 🎓 Next Steps

1. ✅ Install the package
2. ✅ Run `pytorch-teach --help` to see available commands
3. 📚 Start with `pytorch-teach run 1` for Tensor Fundamentals
4. 📖 Check out [DEVELOPMENT.md](DEVELOPMENT.md) to contribute
5. 🌟 Star the repository on GitHub!

## 💡 Tips

- **Interactive Mode (default):** Pauses between sections for learning
- **Batch Mode:** `--batch` flag runs lessons without pauses
- **Verbose Mode:** `--verbose` flag shows detailed output
- **GPU Support:** Automatically detected and used when available
- **Lesson Progress:** Track your learning with the lesson numbering system

## 🐛 Getting Help

- 📖 [Read the Documentation](https://github.com/umitkacar/Pytorch-Teaching)
- 💬 [GitHub Discussions](https://github.com/umitkacar/Pytorch-Teaching/discussions)
- 🐛 [Report Issues](https://github.com/umitkacar/Pytorch-Teaching/issues)
- 📧 Contact the maintainers

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Happy Learning! 🚀**
