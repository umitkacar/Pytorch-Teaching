<div align="center">

# 🔥 PyTorch Teaching - Ultra Modern Learning Hub 🚀

<img src="images/pytorch.jpg" alt="PyTorch Logo" width="300"/>

### *Master Deep Learning with Style* ✨

[![GitHub stars](https://img.shields.io/github/stars/umitkacar/Pytorch-Teaching?style=for-the-badge&logo=github&color=yellow)](https://github.com/umitkacar/Pytorch-Teaching/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/umitkacar/Pytorch-Teaching?style=for-the-badge&logo=github&color=blue)](https://github.com/umitkacar/Pytorch-Teaching/network)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-lessons">Lessons</a> •
  <a href="#-2024-2025-trending-resources">Trending Resources</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🎯 **Interactive Learning**
- 📓 Jupyter Notebook based tutorials
- 🎨 Visual explanations with code
- 💡 Real-world examples
- ⚡ Hands-on practice

</td>
<td width="50%">

### 🚀 **Modern Approach**
- 🔬 Latest PyTorch features (2024-2025)
- 🧠 AI/ML best practices
- 🏆 Industry-standard techniques
- 📊 Performance optimization tips

</td>
</tr>
</table>

---

## 📚 Lessons

### 🎓 **Core Curriculum**

<details open>
<summary><b>📖 Lesson 1: What is Tensor?</b></summary>
<br>

> **🎯 Learning Objectives:**
> - Understanding scalars, vectors, matrices, and tensors
> - Comparing Python, NumPy, and PyTorch implementations
> - Tensor creation and basic operations

```python
import torch
tensor = torch.tensor([[1, 2], [3, 4]])
print(tensor)
```

**📁 File:** `Pytorch-Lesson-1 (What is tensor?).ipynb`

</details>

<details open>
<summary><b>🧮 Lesson 2: Math Functions with Tensors</b></summary>
<br>

> **🎯 Learning Objectives:**
> - Tensor generation: `rand()`, `randn()`, `zeros()`, `ones()`
> - Mathematical operations: addition, multiplication, division
> - Tensor manipulation: `view()`, `reshape()`, `mean()`, `std()`
> - In-place vs standard operations

```python
# Element-wise operations
a = torch.rand(3, 3)
b = torch.rand(3, 3)
result = a * b  # Element-wise multiplication
```

**📁 File:** `Pytorch-Lesson-2 (Math Function with Tensor).ipynb`

</details>

<details open>
<summary><b>⚙️ Lesson 3: Convert Tensor & CPU-CUDA</b></summary>
<br>

> **🎯 Learning Objectives:**
> - Data type conversions (NumPy ↔ PyTorch ↔ Lists)
> - Memory management and sharing
> - CPU to GPU (CUDA) operations
> - Device management best practices

```python
# Move tensor to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([1, 2, 3]).to(device)
```

**📁 File:** `Pytorch-Lesson-3 (Convert tensor and cpu-cuda).ipynb`

</details>

---

## 🔥 2024-2025 Trending Resources

### 🏆 **Must-Follow Repositories**

<table>
<tr>
<td align="center" width="33%">

#### 🤖 **Large Language Models**
[![LLaMA](https://img.shields.io/badge/Meta_LLaMA_3-★_67k-0467DF?style=flat-square&logo=meta)](https://github.com/meta-llama/llama3)
[![GPT-NeoX](https://img.shields.io/badge/GPT--NeoX-★_6k-FF6B6B?style=flat-square)](https://github.com/EleutherAI/gpt-neox)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-★_135k-FFD21E?style=flat-square)](https://github.com/huggingface/transformers)

</td>
<td align="center" width="33%">

#### 🎨 **Computer Vision**
[![YOLOv10](https://img.shields.io/badge/YOLOv10-★_12k-00DFA2?style=flat-square)](https://github.com/THU-MIG/yolov10)
[![SAM 2](https://img.shields.io/badge/Segment_Anything_2-★_25k-4A90E2?style=flat-square&logo=meta)](https://github.com/facebookresearch/segment-anything-2)
[![GroundingDINO](https://img.shields.io/badge/GroundingDINO-★_8k-F6C358?style=flat-square)](https://github.com/IDEA-Research/GroundingDINO)

</td>
<td align="center" width="33%">

#### 🚀 **Training & Optimization**
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-★_35k-0078D4?style=flat-square&logo=microsoft)](https://github.com/microsoft/DeepSpeed)
[![Flash-Attention](https://img.shields.io/badge/Flash_Attention_3-★_15k-FF9500?style=flat-square)](https://github.com/Dao-AILab/flash-attention)
[![Axolotl](https://img.shields.io/badge/Axolotl-★_8k-7C3AED?style=flat-square)](https://github.com/OpenAccess-AI-Collective/axolotl)

</td>
</tr>
</table>

### 🌐 **Advanced PyTorch Frameworks (2024-2025)**

| Framework | Description | Stars | Use Case |
|-----------|-------------|-------|----------|
| 🔥 **[PyTorch Lightning](https://github.com/Lightning-AI/lightning)** | High-level PyTorch framework | ![Stars](https://img.shields.io/github/stars/Lightning-AI/lightning?style=social) | Production-ready training |
| ⚡ **[TorchTune](https://github.com/pytorch/torchtune)** | Native PyTorch LLM fine-tuning | ![Stars](https://img.shields.io/github/stars/pytorch/torchtune?style=social) | LLM fine-tuning |
| 🎯 **[Diffusers](https://github.com/huggingface/diffusers)** | State-of-the-art diffusion models | ![Stars](https://img.shields.io/github/stars/huggingface/diffusers?style=social) | Image/Video generation |
| 🧠 **[Unsloth](https://github.com/unslothai/unsloth)** | 2x faster LLM training | ![Stars](https://img.shields.io/github/stars/unslothai/unsloth?style=social) | Efficient fine-tuning |
| 🔬 **[torchao](https://github.com/pytorch/ao)** | PyTorch native quantization | ![Stars](https://img.shields.io/github/stars/pytorch/ao?style=social) | Model optimization |
| 🎪 **[Torchvision](https://github.com/pytorch/vision)** | Computer vision library | ![Stars](https://img.shields.io/github/stars/pytorch/vision?style=social) | Vision tasks |

### 🎓 **Learning Resources 2024-2025**

<div align="center">

| Resource | Type | Level | 🌟 Rating |
|----------|------|-------|-----------|
| **[Deep Learning with PyTorch](https://pytorch.org/tutorials/)** | Official Tutorials | Beginner-Advanced | ⭐⭐⭐⭐⭐ |
| **[Fast.ai Practical Deep Learning](https://course.fast.ai/)** | Course | Intermediate | ⭐⭐⭐⭐⭐ |
| **[d2l.ai - Dive into Deep Learning](https://d2l.ai/)** | Interactive Book | All Levels | ⭐⭐⭐⭐⭐ |
| **[PyTorch Recipes](https://pytorch.org/tutorials/recipes/recipes_index.html)** | Code Snippets | All Levels | ⭐⭐⭐⭐ |
| **[Papers with Code](https://paperswithcode.com/lib/pytorch)** | Research + Code | Advanced | ⭐⭐⭐⭐⭐ |

</div>

### 🎬 **Hot Topics 2024-2025**

```mermaid
mindmap
  root((PyTorch 🔥))
    Large Language Models
      LLaMA 3.3
      Mixtral 8x7B
      Gemma 2
      Phi-4
    Computer Vision
      SAM 2
      YOLOv10
      DINO v2
      Depth Anything
    Generative AI
      Stable Diffusion 3.5
      FLUX
      Sora-like models
      ControlNet
    Optimization
      INT4/INT8 Quantization
      Flash Attention 3
      LoRA/QLoRA
      Model Pruning
```

---

## 🛠️ Installation

### **Quick Start** ⚡

```bash
# Clone the repository
git clone https://github.com/umitkacar/Pytorch-Teaching.git
cd Pytorch-Teaching

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install PyTorch (GPU version - CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install jupyter matplotlib numpy pandas
```

### **Docker Setup** 🐳

```bash
# Pull official PyTorch image
docker pull pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

# Run Jupyter
docker run -it --gpus all -p 8888:8888 -v $(pwd):/workspace pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime jupyter notebook --allow-root
```

---

## 🎯 Roadmap

```
✅ Lesson 1: Tensor Fundamentals
✅ Lesson 2: Math Operations
✅ Lesson 3: CPU/CUDA Conversion
🚧 Lesson 4: Neural Networks Basics (Coming Soon)
🚧 Lesson 5: Convolutional Neural Networks
🚧 Lesson 6: Recurrent Neural Networks
🚧 Lesson 7: Transformers & Attention
🚧 Lesson 8: Transfer Learning
🚧 Lesson 9: Generative Models
🚧 Lesson 10: Production Deployment
```

---

## 💻 System Requirements

<table>
<tr>
<td width="50%">

### **Minimum Requirements**
- 🖥️ **CPU:** Intel Core i5 or equivalent
- 🧠 **RAM:** 8 GB
- 💾 **Storage:** 5 GB free space
- 🐍 **Python:** 3.8+
- 📦 **PyTorch:** 2.0+

</td>
<td width="50%">

### **Recommended Requirements**
- 🖥️ **CPU:** Intel Core i7/AMD Ryzen 7
- 🧠 **RAM:** 16 GB+
- 🎮 **GPU:** NVIDIA RTX 3060+ (8GB VRAM)
- 💾 **Storage:** 20 GB SSD
- 🐍 **Python:** 3.10+
- 📦 **PyTorch:** 2.5+

</td>
</tr>
</table>

---

## 🤝 Contributing

We welcome contributions! 🎉

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🎯 Open a Pull Request

---

## 📊 GitHub Stats

<div align="center">

![GitHub Stats](https://img.shields.io/github/repo-size/umitkacar/Pytorch-Teaching?style=for-the-badge&logo=github&color=blue&label=Repo%20Size)
![Last Commit](https://img.shields.io/github/last-commit/umitkacar/Pytorch-Teaching?style=for-the-badge&logo=github&color=green&label=Last%20Commit)
![Issues](https://img.shields.io/github/issues/umitkacar/Pytorch-Teaching?style=for-the-badge&logo=github&color=red&label=Open%20Issues)

</div>

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Show Your Support

If you find this project helpful, please consider giving it a ⭐!

<div align="center">

### **Made with ❤️ for the PyTorch Community**

[![Star History Chart](https://api.star-history.com/svg?repos=umitkacar/Pytorch-Teaching&type=Date)](https://star-history.com/#umitkacar/Pytorch-Teaching&Date)

---

**Happy Learning! 🚀✨**

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

</div>
