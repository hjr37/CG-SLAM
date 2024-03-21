# <sup><img src="./assets/Logo.png" alt="Logo" width="70px"></sup><font color="blue">CG-SLAM</font>: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-aware 3D Gaussian Field

**[Jiarui Hu]()<sup>1</sup>, [Xianhao Chen]()<sup>2</sup>,
[Boyin Feng]()<sup>1</sup>,
[Guanglin Li]()<sup>1</sup>,
[Liangjing Yang]()<sup>2</sup>,
[Hujun Bao]()<sup>1</sup>
[Guofeng Zhang]()<sup>1</sup>
[Zhaopeng Cui]()<sup>1\*</sup>**<br>
<sup>1 </sup>State Key Lab of CAD\&CG, Zhejiang University<br>
<sup>2 </sup>ZJU-UIUC Institute, International Campus, Zhejiang University<br>
<sup>* </sup>Corresponding author.<br>

-----

### [🌐 Project page](https://zju3dv.github.io/cg-slam) | [📝 Paper]() | [📽️ Video]()

<p align="left">
  <a href="">
    <img src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/images/teaser.jpg" alt="CG-SLAM teaser" width="90%">
  </a>
</p>

<p style="text-align: justify;">We presents an efficient dense RGB-D SLAM system, based on a novel uncertainty-aware 3D Gaussian field with high consistency and geometric stability. In our system, we have introduced a comprehensive mathematical theory for the derivatives with respect to pose in the 3D Gaussian splatting framework. Additionally, we have developed a customized GPU-accelerated rasterization pipeline specifically for SLAM, enabling our system to achieve state-of-the-art accuracy and efficiency in various scenarios.</p>

<video src="https://raw.githubusercontent.com/hjr37/open_access_assets/main/cg-slam/video/rviz.mp4" width="640" height="360" controls></video>

**Table of Contents**

- [Update](#Update)
- [Installation](#installation)
- [Usage](#usage)
    - [Run](#downloading-example-datasets)
    - [Evaluation](#training-a-feature-field)
- [Citation](#citation)


# Update

- [ ] Code for CG-SLAM
