<!-- CG-SLAM   -->

# <sup><img src="./assets/Logo.png" alt="Logo" width="70px"></sup>CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

**[Jiarui Hu](https://github.com/hjr37)<sup>1</sup>,
[Xianhao Chen](https://github.com/CXavireH)<sup>2</sup>,
[Boyin Feng]()<sup>1</sup>,
[Guanglin Li]()<sup>1</sup>,
[Liangjing Yang](https://person.zju.edu.cn/ylj)<sup>2</sup>,
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)<sup>1</sup>
[Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)<sup>1</sup>
[Zhaopeng Cui](https://zhpcui.github.io/)<sup>1\*</sup>**<br>
<sup>1 </sup>State Key Lab of CAD\&CG, Zhejiang University<br>
<sup>2 </sup>ZJU-UIUC Institute, International Campus, Zhejiang University<br>
<sup>* </sup>Corresponding author.<br>

-----

### [🌐 Project page](https://zju3dv.github.io/cg-slam) | [📝 Paper]() | [📽️ Video]()

<p align="left">
  <p style="text-align: justify;">This is the official implementation of <strong>CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field</strong>. CG-SLAM can achieve state-of-the-art performance in tracking, mapping, rendering, and efficiency.</p>
  <a href="">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/images/teaser.jpg" alt="CG-SLAM teaser" width="100%">
  </a>
</p>

<p align="center">
  <a href="">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/video/rviz.gif" alt="Rviz" width="100%">
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#Update">Update</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#demo">Online Demo</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li><a href="##Run">Run</a></li>
        <li><a href="##Evaluation">Evaluation</a></li>
      </ol>
    </li>
    <li>
      <a href="#Acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
  </ol>
</details>


# Update

- [ ] Code for Diff-rasterization(w/pose --> 4✖️4 Transformation Matrix <strong>T</strong>)
- [ ] Code for RGBD-SLAM
- [ ] Code for Evaluation

# Installation
We conducted our system testing on two desktop configurations: one with an Intel i9-14900K CPU and an NVIDIA RTX 4090 GPU, and another with an Intel Xeon Gold CPU and an NVIDIA RTX 3090 GPU. Our code has undergone testing with CUDA SDK 11.3 and Python 3.8.

**In terms of hardware requirements**, it is necessary to have a CUDA-enabled GPU with a Compute Capability of 7.0 or higher, as well as a minimum of 10GB VRAM. 

**In terms of software requirements**, we recommend using Conda to manage your environment. Additionally, you need to meet the following three conditions for our code: a C++ Compiler for PyTorch extensions, CUDA SDK 11 for PyTorch extensions, and compatibility between the C++ Compiler and CUDA SDK. 
- ### Method 1 step-by-step set up(Recommended)
```bash
# Cloning the Repository
git clone https://github.com/hjr37/CG-SLAM.git
# Create the python environment and activate
conda create --name cg-slam python=3.8
conda activate cg-slam
```
Please note that you should choose the appropriate version of the installation based on your CUDA SDK version. You can refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for guidance in this regard.
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install cudatoolkit=11.3
git clone --branch v0.6.2 https://github.com/facebookresearch/pytorch3d.gity
cd pytorch3d && pip install -e . && cd ..
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install pyyaml scikit-image torch_scatter opencv-python plyfile open3d tensorboard
```
- ### Method 2 Configure the environment in one line
```bash
conda env create --file environment.yml
conda activate cg-slam
```
# Usage

## Run
### Replica
After downloading the dataset, you need to locate the corresponding YAML file in the `./configs/Replica` directory. Then, modify the value of `input_folder` in the YAML file to the local location of the datasets.

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/Replica/office0.yaml
```
### TUM RGB-D

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/TUM/fr1_desk.yaml
```

### Scannet

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/ScanNet/scene0000.yaml
```

## Evaluation
Once the execution of a scene is completed, you can utilize the `eval_ate.py` in the `./tools/` directoory to evaluate the tracking accuracy of our system in that specific scene.
```bash
python tools/eval_ate.py --config ./configs/Replica/office0.yaml
```
# Acknowledgement

# Citation
