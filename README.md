<!-- CG-SLAM -->

<p align="center">
  <h1 align="center">
    CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field
    <br>
    [ECCV 2024]
  </h1>
  <p align="center">
    <a href="https://github.com/hjr37/"><strong>Jiarui Hu</strong><sup>1</sup><sup>†</sup></a>
    ·
    <a href="https://github.com/CXavireH/"><strong>Xianhao Chen</strong><sup>1</sup><sup>†</sup></a>
    ·
    <a href="https://github.com/JSUISLA"><strong>Boyin Feng</strong><sup>1</sup></a>
    ·
    <a href="https://github.com/liguanglin"><strong>Guanglin Li</strong><sup>1</sup></a>
    ·
    <a href="https://person.zju.edu.cn/ylj/"><strong>Liangjing Yang</strong><sup>2</sup></a>
    <br>
    <a href="http://www.cad.zju.edu.cn/home/bao/"><strong>Hujun Bao</strong><sup>1</sup></a>
    ·
    <a href="http://www.cad.zju.edu.cn/home/gfzhang/"><strong>Guofeng Zhang</strong><sup>1</sup></a>
    ·
    <a href="https://zhpcui.github.io/"><strong>Zhaopeng Cui</strong><sup>1*</sup></a>
    <br>
    <sup>1</sup> State Key Lab of CAD&CG, Zhejiang University<br>
    <sup>2</sup> ZJU-UIUC Institute, International Campus, Zhejiang University<br>
    <sup>*</sup> Corresponding author. <sup>†</sup> Equal contribution.<br>
  </p>
  <h3 align="center"><a href="https://zju3dv.github.io/cg-slam/">Project Page</a> | <a href="https://arxiv.org/abs/2403.16095">Paper</a> | <a href="https://zju3dv.github.io/cg-slam/">Video</a></h3>
  <div align="center"></div>
</p>

<p align="left">
  <p style="text-align: justify;">This repository contains the official implementation of <strong>CG-SLAM</strong>, a dense RGB-D SLAM system built on a consistent uncertainty-aware 3D Gaussian field. The project targets high-quality tracking, mapping, and rendering while keeping the system efficient enough for practical scene reconstruction experiments.</p>
  <a href="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/images/teaser.jpg">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/images/teaser.jpg" alt="CG-SLAM teaser" width="100%">
  </a>
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/video/rviz.gif">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/video/rviz.gif" alt="CG-SLAM visualization" width="100%">
  </a>
</p>

<details open="open" style='padding: 10px; border-radius:5px 15px 15px 5px; border-style: solid; border-width: 1px;'>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#update">Update</a></li>
    <li><a href="#submodule">Submodule</a></li>
    <li><a href="#installation">Installation</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li><a href="#run">Run</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ol>
    </li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

# Update

- [x] Code for Diff-rasterization(w/pose --> 4✖️4 Transformation Matrix <strong>T</strong>)
- [x] <strong>Our paper is accepted by ECCV 2024, and our code is coming soon!!!</strong>
- [x] Code for RGBD-SLAM
- [x] Code for Evaluation

# Submodule
<p style="text-align: justify;">We have proposed a comprehensive mathematical theory on derivatives w.r.t. pose in 3D Gaussian splatting framework. Additionally, we have developed a specialized CUDA framework tailored for the SLAM task, decoupling the tracking and mapping components. For more details, please refer to the provided <a href="https://github.com/hjr37/diff-gaussian-rasterization">diff-gaussian-rasterization</a>.</p>


# Installation

## Requirements

- Linux
- Python 3.8
- CUDA-capable NVIDIA GPU
- PyTorch with a CUDA build compatible with your local toolkit


## Recommended setup

```bash
git clone <your-release-url> CG-SLAM
cd CG-SLAM
git submodule update --init --recursive

conda create -n cg-slam python=3.8
conda activate cg-slam
```

Install PyTorch first. Pick the wheel that matches your CUDA environment. For example:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Install the remaining Python dependencies and local extensions:

```bash
pip install pyyaml scipy scikit-image opencv-python plyfile open3d tensorboard mathutils==2.81.2 tqdm trimesh imageio
pip install -e ./submodules/diff-gaussian-rasterization
pip install -e ./submodules/simple-knn
```

`torch-scatter` should be installed with a wheel that matches your PyTorch and CUDA versions. If it is not already available in your environment, install the matching build before running CG-SLAM.

# Usage

## Run

The provided YAML files under `configs/` are examples for different datasets. Before running, either:

- edit `data.input_folder` and `data.output_folder` in the selected config file, or
- override them from the command line with `--input_folder` and `--output_folder`

### Replica

```bash
python run.py \
  --config ./configs/Replica/office0.yaml \
```

### TUM RGB-D

```bash
python run.py \
  --config ./configs/TUM/fr1_desk.yaml \
```

### ScanNet

```bash
python run.py \
  --config ./configs/ScanNet/scene0000.yaml \
```

## Evaluation

### ATE

For Replica scenes, you can evaluate the estimated trajectory with:

```bash
python tools/eval_ate.py --config ./configs/Replica/office0.yaml
```

For TUM RGB-D and ScanNet, the current script expects the corresponding ground-truth pose file to be available in the output directory as `gt_pose.txt`.



# Acknowledgement

We sincerely thank the authors of the following projects:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/)
- [Diff-Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization/)

Their excellent open-source work made this project possible.

# Citation

```bibtex
@article{hu2024cg,
  title={CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field},
  author={Hu, Jiarui and Chen, Xianhao and Feng, Boyin and Li, Guanglin and Yang, Liangjing and Bao, Hujun and Zhang, Guofeng and Cui, Zhaopeng},
  journal={arXiv preprint arXiv:2403.16095},
  year={2024}
}
```
