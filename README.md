<!-- CG-SLAM   -->

<p align="center">
  <h1 align="center"><sup><img src="./assets/Logo.png" alt="Logo" width="70px"></sup>CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field</h1>
  <p align="center">
    <a href="https://github.com/hjr37/"><strong>Jiarui Hu</strong><sup>1</sup></a>
    ·
    <a href="https://github.com/CXavireH/"><strong>Xianhao Chen</strong><sup>2</sup></a>
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
    <sup>1 </sup>State Key Lab of CAD&CG, Zhejiang University<br>
    <sup>2 </sup>ZJU-UIUC Institute, International Campus, Zhejiang University<br>
    <sup>* </sup>Corresponding author.<br>
  </p>
  <h3 align="center"><a href="https://zju3dv.github.io/cg-slam/">🌐Project page</a> | <a href="https://arxiv.org/abs/2403.16095">📝Paper</a> | <a href="https://zju3dv.github.io/cg-slam/">📽️Video</a></h3>
  <div align="center"></div>
</p>

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
<details open="open" style='padding: 10px; border-radius:5px 15px 15px 5px; border-style: solid; border-width: 1px;'>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#Update">Update</a>
    </li>
    <li>
      <a href="#Submodule">Submodule</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li><a href="#run">Run</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
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

# Submodule
<p style="text-align: justify;">we have proposed the first comprehensive mathematical theory on derivatives w.r.t. pose in 3D Gaussian splatting framework. Additionally, we have developed a <strong>specialized CUDA framework tailored</strong> for the SLAM task, effectively decoupling the tracking and mapping components. For more details, please refer to the provided <a href="https://github.com/hjr37/diff-gaussian-rasterization">diff-gaussian-rasterization</a>.</p>

# Installation
<!---
<p style="text-align: justify;">We conducted our system testing on two desktop configurations: one with an Intel i9-14900K CPU and an NVIDIA RTX 4090 GPU, and another with an Intel Xeon Gold CPU and an NVIDIA RTX 3090 GPU. Our code has undergone testing with CUDA SDK 11.3 and Python 3.8.</p>
<p style="text-align: justify;"><strong>In terms of hardware requirements,</strong> it is necessary to have a CUDA-enabled GPU with a Compute Capability of 7.0 or higher, as well as a minimum of 10GB VRAM.</p>
<p style="text-align: justify;"><strong>In terms of software requirements,</strong> we recommend using Conda to manage your environment. Additionally, you need to meet the following three conditions for our code: a C++ Compiler for PyTorch extensions, CUDA SDK 11 for PyTorch extensions, and compatibility between the C++ Compiler and CUDA SDK.</p>

- ### Method 1 step-by-step set up(Recommended)
```bash
# Cloning the Repository
git clone https://github.com/hjr37/CG-SLAM.git
# Create the python environment and activate
conda create --name cg-slam python=3.8
conda activate cg-slam
```
<p align="left">
  <p style="text-align: justify;">Please note that you should choose the appropriate version of the installation based on your CUDA SDK version. You can refer to <a href="https://pytorch.org/get-started/previous-versions/">Pytorch</a> for guidance in this regard.</p>
</p>

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install cudatoolkit=11.3
git clone --branch v0.6.2 https://github.com/facebookresearch/pytorch3d.gity
cd pytorch3d && pip install -e . && cd ..
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install pyyaml scikit-image torch_scatter opencv-python plyfile open3d tensorboard mathutils==2.81.2
```
- ### Method 2 Configure the environment in one line
```bash
conda env create --file environment.yml
conda activate cg-slam
```
-->

# Usage
## Run

<!--
### Replica
<p align="left">
  <p style="text-align: justify;">After downloading the dataset, you need to locate the corresponding YAML file in the <code>./configs/Replica</code> directory. Then, modify the value of <code>input_folder</code> in the YAML file to the local location of the datasets.</p>
</p>

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/Replica/office0.yaml
```
### TUM RGB-D
<p style="text-align: justify;">Just like in <strong>Replica</strong>, you need to modify the <code>input_folder</code> to the appropriate path where your data is located.</p>

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/TUM/fr1_desk.yaml
```

### Scannet
<p style="text-align: justify;">Just like in <strong>Replica</strong>, you need to modify the <code>input_folder</code> to the appropriate path where your data is located.</p>

Then you can run CG-SLAM:
```bash
python run.py --config ./configs/ScanNet/scene0000.yaml
```

> [!NOTE]
> Please note that if you intend to run the **light version** of CG-SLAM, you need to add <code>[crop_size]</code> primarily in the <code>replica.yaml</code> file located in the <code>./configs/</code> directory.
-->

## Evaluation
<!--
<p style="text-align: justify;">Once the execution of a scene is completed, you can utilize the <code>eval_ate.py</code> in the <code>./tools/ </code> directoory to evaluate the tracking accuracy of our system in that specific scene.</p>

```bash
python tools/eval_ate.py --config ./configs/Replica/office0.yaml
```
-->

# Acknowledgement
<p style="text-align: justify;">We sincerely thank the author of the <a href="https://github.com/graphdeco-inria/gaussian-splatting/">3D Gaussian Splatting</a> and <a href="https://github.com/graphdeco-inria/diff-gaussian-rasterization/">Diff-Gaussian Rasterization</a> repositories for their valuable contributions. Their exceptional work has been instrumental in advancing our project.</p>

# Citation
<pre><code>@article{hu2024cg,
    title={CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field},
    author={Hu, Jiarui and Chen, Xianhao and Feng, Boyin and Li, Guanglin and Yang, Liangjing and Bao, Hujun and Zhang, Guofeng and Cui, Zhaopeng},
    journal={arXiv preprint arXiv:2403.16095},
    year={2024}
}
