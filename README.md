# [IJCAI 2025] MultiDreamer3D: Multi-concept 3D Customization with Concept-Aware Diffusion Guidance

<p align="center">
  <img src="./asset/overview.png" width="100%"/>
</p>

<p align="center">
  <b>Wooseok Song</b>, <b>Seunggyu Chang</b>, <b>Jaejun Yoo</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.13449">📄 Paper</a> |
  <a href="https://wooseok-song.github.io/MultiDreamer3D/">🌐 Project Page</a>
</p>

---

## 🏆 News
- <b>MultiDreamer3D</b> has been <b>accepted by IJCAI 2025</b>! 🎉
- <b>MultiDreamer3D</b> is currently undergoing major code refactoring. Stay tuned for updates! 🔧

---

## 📝 Abstract
> While single-concept customization has been studied in 3D, multi-concept customization remains largely unexplored. To address this, we propose MultiDreamer3D that can generate coherent multi-concept 3D content in a divide-and-conquer manner. First, we generate 3D bounding boxes using an LLM-based layout controller. Next, a selective point cloud generator creates coarse point clouds for each concept. These point clouds are placed in the 3D bounding boxes and initialized into 3D Gaussian Splatting with concept labels, enabling precise identification of concept attributions in 2D projections. Finally, we refine 3D Gaussians via concept-aware interval score matching, guided by concept-aware diffusion. Our experimental results show that MultiDreamer3D not only ensures object presence and preserves the distinct identities of each concept but also successfully handles complex cases such as property change or interaction. To the best of our knowledge, we are the first to address the multi-concept customization in 3D.

---

## 🚀 Quick Start

### 1. Environment Setup

Tested on:
- <b>Python</b> 3.10
- <b>CUDA</b> 11.8
- <b>Torch</b> 2.1.0+cu118

```bash
# Create environment
conda create -n MultiDreamer3D python=3.10
conda activate MultiDreamer3D

# Install CUDA 11.8
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit=11.8

# Install PyTorch 2.1.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/

# Install Shap-E
git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .
cd ..
```

### 2. Shap-E Checkpoint
- We leverage Shap-E finetuned on Cap3D dataset.
- Download from: [HuggingFace Cap3D finetuned models](https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/our_finetuned_models)
- Place `shapE_finetuned_with_825kdata.pth` under `./load`

---

## 🏃‍♂️ Usage
- **Train with Concept LoRA:**  
    For training Concept LoRA, please follow the official [DreamBooth LoRA training guide](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) from Hugging Face Diffusers.

- **Train with config file:**
  ```bash
  python train.py --opt <path to config file>
  ```
- **Or use provided shell script:**
  ```bash
  bash ./scripts/train.sh
  ```

We will soon provide the config files, LoRA weights, and .ply files used in our experiments.

---

## 📖 Citation
If you find this repository useful for your research, please cite:

```bibtex
@article{song2025multidreamer3d,
  title={MultiDreamer3D: Multi-concept 3D Customization with Concept-Aware Diffusion Guidance},
  author={Song, Wooseok and Chang, Seunggyu and Yoo, Jaejun},
  journal={arXiv preprint arXiv:2501.13449},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This repository heavily depends on:
- [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 