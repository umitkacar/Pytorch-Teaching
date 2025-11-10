# 🛠️ Development Guide

Comprehensive guide for contributing to PyTorch Teaching.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/umitkacar/Pytorch-Teaching.git
cd Pytorch-Teaching

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
make install-dev
# OR
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📦 Project Structure

```
pytorch-teaching/
├── src/pytorch_teaching/          # Source code
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point
│   ├── lessons/                   # Lesson modules
│   │   ├── lesson_01_tensors.py
│   │   ├── lesson_02_math_ops.py
│   │   ├── lesson_03_device_management.py
│   │   ├── lesson_21_executorch.py
│   │   └── ...
│   └── utils/                     # Utility modules
│       ├── helpers.py
│       └── visualization.py
├── tests/                         # Test suite
│   ├── __init__.py
│   └── test_lessons.py
├── docs/                          # Documentation
├── pyproject.toml                 # Project configuration
├── .pre-commit-config.yaml        # Pre-commit hooks
├── Makefile                       # Development commands
├── tox.ini                        # Tox configuration
└── .editorconfig                  # Editor configuration
```

## 🔧 Development Tools

### Using Make

```bash
make help              # Show all available commands
make install-dev       # Install with dev dependencies
make test              # Run tests
make test-cov          # Run tests with coverage
make test-parallel     # Run tests in parallel
make lint              # Run linters
make lint-fix          # Auto-fix linting issues
make format            # Format code
make format-check      # Check formatting
make type-check        # Run type checking
make security          # Run security checks
make check             # Run all checks
make clean             # Clean build artifacts
```

### Using Hatch

```bash
hatch run test         # Run tests
hatch run test-cov     # Run tests with coverage
hatch run lint         # Run linting
hatch run format       # Format code
hatch run type-check   # Type checking
hatch run security     # Security checks
hatch run check        # All checks
hatch run all          # Format + lint + type + test
```

### Using Tox

```bash
tox -e py311           # Test on Python 3.11
tox -e lint            # Linting
tox -e type            # Type checking
tox -e security        # Security checks
tox                    # Run all environments
```

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=pytorch_teaching --cov-report=html

# Parallel execution
pytest -n auto

# Specific test file
pytest tests/test_lessons.py

# Specific test
pytest tests/test_lessons.py::TestLesson01::test_scalar_creation

# Verbose output
pytest -vv

# Stop on first failure
pytest -x

# Show locals on failure
pytest -l
```

### Test Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run only integration tests
pytest -m integration
```

### Coverage Reports

```bash
# HTML report
pytest --cov=pytorch_teaching --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=pytorch_teaching --cov-report=term-missing

# XML report (for CI)
pytest --cov=pytorch_teaching --cov-report=xml
```

## 🎨 Code Quality

### Formatting

```bash
# Format code
black src tests

# Check formatting
black --check src tests

# Format specific file
black src/pytorch_teaching/cli.py
```

### Linting

```bash
# Lint all code
ruff check src tests

# Auto-fix issues
ruff check --fix src tests

# Specific file
ruff check src/pytorch_teaching/cli.py

# Show rule descriptions
ruff check --show-source src tests
```

### Type Checking

```bash
# Check all code
mypy src

# Specific file
mypy src/pytorch_teaching/cli.py

# With coverage report
mypy --html-report mypy-report src
```

### Security Scanning

```bash
# Scan for security issues
bandit -r src

# With configuration
bandit -r src -ll

# Generate baseline
bandit -r src -f json -o .bandit.json
```

## 🔄 Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Update hooks
pre-commit autoupdate

# Bypass hooks (not recommended)
git commit --no-verify
```

## 📝 Adding a New Lesson

1. **Create lesson file:**
   ```bash
   touch src/pytorch_teaching/lessons/lesson_XX_name.py
   ```

2. **Implement lesson structure:**
   ```python
   """
   Lesson XX: Title.

   Description of what this lesson covers.

   Learning Objectives:
       - Objective 1
       - Objective 2
   """

   from rich.console import Console

   console = Console()

   def run(interactive: bool = True, verbose: bool = False):
       """Run Lesson XX."""
       console.print("[bold cyan]Lesson XX: Title[/bold cyan]")
       # Implementation
   ```

3. **Update CLI:**
   - Add import in `src/pytorch_teaching/lessons/__init__.py`
   - Add case in `src/pytorch_teaching/cli.py`

4. **Add tests:**
   ```python
   class TestLessonXX:
       def test_lesson_runs(self):
           lesson_XX.run(interactive=False, verbose=False)
   ```

5. **Update documentation:**
   - Update README.md
   - Update CHANGELOG.md

## 🏗️ Building & Publishing

### Build Package

```bash
# Clean previous builds
make clean

# Build distribution
python -m build

# Check dist files
ls -lh dist/
```

### Check Package

```bash
# Install twine
pip install twine

# Check package
twine check dist/*

# Test install locally
pip install dist/*.whl
```

### Publish

```bash
# Test PyPI
make publish-test

# Production PyPI
make publish
```

## 🐛 Debugging

### CLI Debugging

```bash
# Run with verbose output
pytorch-teach run 1 --verbose

# Python debugger
python -m pdb -m pytorch_teaching.cli run 1

# IPython debugger
ipython --pdb -m pytorch_teaching.cli run 1
```

### Test Debugging

```bash
# Run tests with pdb on failure
pytest --pdb

# Drop into pdb on first failure
pytest -x --pdb

# Verbose test output
pytest -vv --tb=long
```

## 📊 Performance Profiling

### Benchmarking

```bash
# Run benchmarks
pytest tests/ --benchmark-only

# Compare benchmarks
pytest tests/ --benchmark-compare
```

### Memory Profiling

```python
# Using memory_profiler
from memory_profiler import profile

@profile
def my_function():
    # Code to profile
    pass
```

## 🔐 Security Best Practices

1. **Never commit secrets:**
   - Use `.env` files (gitignored)
   - Use environment variables
   - Use secret management tools

2. **Dependency security:**
   ```bash
   # Check for vulnerabilities
   pip-audit

   # Update dependencies
   pip list --outdated
   ```

3. **Code scanning:**
   ```bash
   # Bandit
   bandit -r src

   # Safety
   safety check
   ```

## 📚 Documentation

### Building Docs

```bash
# Build documentation
cd docs
mkdocs build

# Serve locally
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Writing Docstrings

Follow Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """
    Short description.

    Longer description with more details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Example:
        >>> function("test", 42)
        True
    """
    pass
```

## 🤝 Pull Request Process

1. **Create feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make changes:**
   - Write code
   - Add tests
   - Update documentation

3. **Run checks:**
   ```bash
   make check
   make test-cov
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/amazing-feature
   ```

## 🎯 Code Review Checklist

- [ ] Tests pass locally
- [ ] Code coverage maintained/improved
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No linting errors
- [ ] Security checks pass
- [ ] Commit messages follow convention
- [ ] PR description is clear

## 🔄 Continuous Integration

Our CI pipeline runs:

1. **Linting:** Black, Ruff, MyPy
2. **Security:** Bandit
3. **Testing:** Pytest on Python 3.9-3.12
4. **Coverage:** Codecov upload
5. **Build:** Package build test

## 📞 Getting Help

- 💬 [GitHub Discussions](https://github.com/umitkacar/Pytorch-Teaching/discussions)
- 🐛 [Issue Tracker](https://github.com/umitkacar/Pytorch-Teaching/issues)
- 📧 Contact maintainers

## 🎓 Learning Resources

- [PyTorch Contributing Guide](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)
- [Python Packaging Guide](https://packaging.python.org/)
- [Hatch Documentation](https://hatch.pypa.io/)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Happy Developing! 🚀**
