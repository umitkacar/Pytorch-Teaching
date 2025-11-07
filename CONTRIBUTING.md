# 🤝 Contributing to PyTorch Teaching

First off, thank you for considering contributing to PyTorch Teaching! 🎉 It's people like you that make this learning resource amazing for everyone.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Development Setup](#-development-setup)
- [Style Guidelines](#-style-guidelines)
- [Commit Guidelines](#-commit-guidelines)
- [Pull Request Process](#-pull-request-process)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When submitting a bug report, include:**
- 📝 Clear and descriptive title
- 🔍 Detailed steps to reproduce
- 💻 Your environment (OS, Python version, PyTorch version)
- 📸 Screenshots if applicable
- 🎯 Expected vs actual behavior

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- 🎨 Clear use case and motivation
- 📊 Examples of how it would work
- 🔗 Related resources or implementations

### 📚 Adding New Lessons

We're always looking for new educational content!

**Lesson Requirements:**
- ✅ Jupyter Notebook format
- 📖 Clear learning objectives
- 💻 Working code examples
- 📝 Detailed explanations
- 🎯 Practical exercises
- 🔗 References to official documentation

**Lesson Structure:**
```markdown
# Lesson X: Title

## 🎯 Learning Objectives
- Objective 1
- Objective 2

## 📖 Theory
[Explanation with visual aids]

## 💻 Code Examples
[Working code with comments]

## 🏋️ Exercises
[Practice problems]

## 📚 References
[Links to resources]
```

### 🔧 Improving Documentation

Documentation improvements are highly valued:
- 📝 Fixing typos or unclear explanations
- 🌍 Adding translations
- 🎨 Improving visual elements
- 🔗 Adding useful resources

---

## 🛠️ Development Setup

### Prerequisites

```bash
# Python 3.8+
python --version

# Git
git --version
```

### Setup Steps

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub first
   git clone https://github.com/YOUR_USERNAME/Pytorch-Teaching.git
   cd Pytorch-Teaching
   ```

2. **Create Virtual Environment**
   ```bash
   # Create venv
   python -m venv venv

   # Activate (Linux/Mac)
   source venv/bin/activate

   # Activate (Windows)
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install jupyter notebook
   pip install matplotlib numpy pandas
   pip install black flake8  # Code formatting
   ```

4. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 🎨 Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# ✅ Good
import torch
import numpy as np

def create_tensor(data: list) -> torch.Tensor:
    """
    Create a PyTorch tensor from a list.

    Args:
        data: Input list of numbers

    Returns:
        PyTorch tensor
    """
    return torch.tensor(data)

# ❌ Bad
import torch,numpy as np
def create_tensor(data):
    return torch.tensor(data)
```

**Key Points:**
- 📏 Line length: max 100 characters
- 🎯 Use type hints where possible
- 📝 Write docstrings for functions
- 💬 Add comments for complex logic
- 🧹 Use meaningful variable names

### Jupyter Notebook Style

```python
# Cell 1: Imports and Setup
import torch
import matplotlib.pyplot as plt

# Cell 2: Explanation (Markdown)
# ## What is a Tensor?
# A tensor is a multi-dimensional array...

# Cell 3: Code Example
tensor = torch.tensor([[1, 2], [3, 4]])
print(f"Tensor shape: {tensor.shape}")

# Cell 4: Visualization
plt.imshow(tensor.numpy())
plt.title("Tensor Visualization")
plt.show()
```

**Best Practices:**
- 🔢 Number your cells logically
- 📝 Add markdown cells for explanations
- 🎨 Use visualizations where helpful
- ⚡ Keep cells focused and short
- 🧪 Ensure all cells run in order

### Markdown Style

```markdown
# ✅ Good - Clear hierarchy
## Section Title
### Subsection

- Use bullet points
- For lists

**Bold** for emphasis
`code` for technical terms

# ❌ Bad - Inconsistent formatting
## SECTION TITLE
- inconsistent
* mixing
+ list styles
```

---

## 📝 Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- ✨ `feat`: New feature or lesson
- 🐛 `fix`: Bug fix
- 📝 `docs`: Documentation changes
- 🎨 `style`: Formatting, missing semicolons, etc.
- ♻️ `refactor`: Code refactoring
- ✅ `test`: Adding tests
- 🔧 `chore`: Maintenance tasks

### Examples

```bash
# ✅ Good commits
git commit -m "feat(lesson4): add neural networks basics"
git commit -m "fix(lesson1): correct tensor dimension example"
git commit -m "docs(readme): add installation instructions"

# ❌ Bad commits
git commit -m "update stuff"
git commit -m "fix bug"
git commit -m "changes"
```

### Commit Best Practices

- 🎯 One logical change per commit
- 📝 Clear and descriptive messages
- 🔍 Reference issues if applicable (#123)
- ✅ Ensure code runs before committing

---

## 🚀 Pull Request Process

### Before Submitting

- [ ] ✅ Code runs without errors
- [ ] 📝 Documentation is updated
- [ ] 🧪 Examples work as expected
- [ ] 🎨 Code follows style guidelines
- [ ] 📚 Commit messages are clear
- [ ] 🔍 No merge conflicts

### PR Template

```markdown
## 📋 Description
Brief description of changes

## 🎯 Type of Change
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📝 Documentation
- [ ] ♻️ Refactoring

## 🧪 Testing
How to test these changes

## 📸 Screenshots (if applicable)
Add screenshots here

## 📚 Related Issues
Closes #123
```

### Review Process

1. 👀 **Automated checks** run on your PR
2. 🔍 **Maintainer review** - typically within 48 hours
3. 💬 **Discussion** - address feedback if needed
4. ✅ **Approval** - PR gets merged!

### After Your PR is Merged

- 🎉 Celebrate! You're now a contributor!
- 🔄 Sync your fork with upstream
- 🌟 Star the repository if you haven't!

```bash
# Sync your fork
git checkout main
git pull upstream main
git push origin main
```

---

## 🌟 Recognition

All contributors will be recognized in:
- 📋 README.md contributors section
- 🏆 GitHub contributors page
- 💝 Special thanks in release notes

---

## 🤔 Questions?

- 💬 Open a [Discussion](https://github.com/umitkacar/Pytorch-Teaching/discussions)
- 📧 Contact maintainers
- 📖 Check [Documentation](README.md)

---

## 📚 Resources for Contributors

### Learning Resources
- 📖 [PyTorch Official Docs](https://pytorch.org/docs/)
- 📘 [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/)
- 📗 [Python PEP 8 Style Guide](https://pep8.org/)
- 📕 [Git Best Practices](https://git-scm.com/doc)

### Tools
- 🛠️ [Black](https://black.readthedocs.io/) - Python formatter
- 🔍 [Flake8](https://flake8.pycqa.org/) - Linter
- 📓 [JupyterLab](https://jupyterlab.readthedocs.io/) - Development environment

---

<div align="center">

## 💖 Thank You!

Your contributions make this project better for everyone.

**Happy Contributing! 🚀**

</div>
