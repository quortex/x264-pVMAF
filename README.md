# pVMAF - Real-time In-loop Perceptual Video Quality Metric for x264

**Contents:**
- [What is pVMAF?](#what-is-pvmaf)
- [Requirements](#requirements)
- [Steps to configure and build x264 with pVMAF](#steps-to-configure-x264-with-pvmaf)
  - [Windows](#windows)
  - [Linux & Mac](#linux--mac)
- [Usage](#usage)
  - [Preparing the source file](#preparing-the-source-file)
  - [Enable pVMAF (--pvmaf)](#enable-pvmaf-with---pvmaf)
  - [Generate Frame Level pVMAF log (-l)](#generate-pvmaf-log-file-with--l)
  - [Generate Candidate Features (-g)](#generate-candidate-features-with--g)
- [About](#about)
    - [Performance Analysis](#performance)
      - [Model Performance Analysis](#model-performance---medium-preset---frame-level-predictions)
      - [Scatter plot - VMAF vs pVMAF](#scatter-plot---vmaf-vs-pvmaf)
      - [Computational Overhead Analysis](#computational-overhead-analysis---single-threaded-execution-with-simd-enabled---medium-preset)
  - [What is the purpose of this activity?](#what-is-the-purpose-of-this-activity)
  - [Contributors](#contributors)
  - [Acknowledgements](#acknowledgements)
  - [Development Notes](#development-notes)
  - [Dev System Info](#dev-system-info)
  - [License](/COPYING)
  - [Citation](#citation)

## What is pVMAF?
VMAF has become the industry standard for evaluating video quality, but its computational demands can hinder its use in real-time applications. To address this, we've developed pVMAF, a more efficient method that closely approximates VMAF's quality predictions in capturing VQ loss due to compression. pVMAF uses a simplified neural network and computationally optimized features combined with information available in the encoding process to achieve produce quality predictions in real-time without sacrificing accuracy. We've kept pVMAF integration into x264 to be as non-intrusive as possible with no changes to the behaviour and the execution flow of x264.

**⚠️ Note: pVMAF currently supports Full-HD (1920x1080) progressive content with YUV420p pixel format encoded on `--medium` preset. With experiments and development still on-going to expand support for more configurations, expect changes to model weights**.

Link to the research paper -> *coming soon!*

Check out our blog series on Video Quality Measurements,
  - [Part 1 - A Brief History on Video Quality Measurements](https://www.synamedia.com/blog/a-brief-history-of-video-quality-measurement-from-psnr-to-vmaf-and-beyond/)
  - [Part 2 - Real-time Video Quality Assessment with pVMAF](https://www.synamedia.com/blog/real-time-video-quality-assessment-with-pvmaf/)
  - Part 3 - *coming soon*
## Requirements
- gcc 5.0 or higher
- nasm-2.13 or higher
- git
- MYSYS2 or WSL (to build the project on Windows)

To Enable SIMD acceleration, processor with support for AVX2, AVX, SSE3 and FMA is required. If your processor lacks support to any of these instruction sets, you can still run x264 with pVMAF by disabling SIMD acceleration. Refer [Dev System Info](#dev-system-info) for more details on specific versions in the dev system.

## Steps to configure x264 with pVMAF

### Windows

Install and configure MYSYS2 following the steps detailed in the official [link](https://www.msys2.org/). Make sure you have gcc and git installed.

Follow the steps for linux & mac to build the project from MYSYS2 terminal.

### Linux & Mac

Clone this repository and then from the root folder of x264, do the following:

**To enable SIMD acceleration,**
```bash
./configure --bit-depth=8 --chroma-format=420 --disable-interlaced
```
(or) \
\
**To disable SIMD acceleration,**
```bash
./configure --disable-asm --bit-depth=8 --chroma-format=420 --disable-interlaced
```
**To build the project,**
```bash
make
```
This builds x264 binary with pVMAF in the root folder.

## Usage
### Command line options
#### Preparing the source file
If your source is of a resolution larger than FHD, we recommend resizing it to FHD resolution. An example command line is given below,
```bash
ffmpeg -f rawvideo -pixel_format yuv420p -video_size 3840x2160 -i 4Kinput.yuv -s 1920x1080 -pix_fmt yuv420p FHDoutput.yuv
```

Looking for a quick compatible source to try it out? Here you go,
```bash
wget https://media.xiph.org/video/av2/y4m/WorldCupFarSky_1920x1080_30p.y4m -O input.y4m
```
#### Enable pVMAF with `--pvmaf`
This argument is used to enable pVMAF inference during the encoding process. To view frame level stats, use it along with `-v` or `--verbose`  and `--log-level debug`.
```bash
./x264 --input-res  1920x1080  --crf 20  --input-csp  i420  --fps  30 -o output.264  input.y4m --threads 10 --pvmaf --preset medium --verbose --log-level debug
```
#### Generate pVMAF log file with `-l`
Use this option to write frame level pVMAF scores onto a CSV file.
```bash
./x264 --input-res  1920x1080  --crf 20  --input-csp  i420  --fps  30 -l pVMAF_score_log.csv -o output.264  input.y4m --threads 10 --pvmaf --preset medium --verbose --log-level debug
```
The log file contains display picture number along with QP, frame type and pVMAF score.
#### Generate candidate features with `-g`
Use this option to create a dump of CSV file with all candidate frame level features. We recommend using this option with `--pvmaf` and `--psnr`.
```bash
./x264 --input-res  1920x1080  --crf 20  --input-csp  i420  --fps  30 -g feature_file.csv -o output.264  input.y4m --threads 10 --pvmaf --psnr --preset medium --verbose --log-level debug
```
## About
### Performance
#### Model performance - medium preset - Frame level predictions
| Metric | SROCC ↑ | PLCC ↑ | RMSE ↓ | MAE ↓ |
|---|---|---|---|---|
| PSNR-Y | 0.868 | 0.867 | 15.15 | 11.21 |
| PSNR-HVS-Y | 0.929 | 0.955 | 11.28 | 7.93 |
| SSIM | 0.881 | 0.865 | 15.27 | 10.84 |
| MS-SSIM | 0.941 | 0.968 | 9.98 | 6.99 |
| pVMAF | 0.986 | 0.990 | 4.38 | 2.97 |

#### Model performance - medium preset - Video level predictions obtained by accumulating frame level predictions
| Metric | SROCC ↑ | PLCC ↑ | RMSE ↓ | MAE ↓ |
|---|---|---|---|---|
| PSNR-Y | 0.915 | 0.911 | 12.49 | 9.59 |
| PSNR-HVS-Y | 0.954 | 0.955 | 9.06 | 6.58 |
| SSIM | 0.922 | 0.911 | 12.52 | 9.38 |
| MS-SSIM | 0.960 | 0.968 | 7.79 | 5.59 |
| pVMAF | 0.994 | 0.996 | 2.72 | 1.91 |

**Caption:** Correlation Coefficients and Error Measures Between VMAF and Other FR Metrics

#### Scatter plot - VMAF vs pVMAF
![pvmaf-vs-vmaf](https://github.com/user-attachments/assets/7a204e91-10b6-4c9e-977e-5f2c9c94cc18)



#### Computational overhead analysis - Single-threaded execution with SIMD enabled - Medium preset
| **Preset**   | **Metric** | **Time per frame (ms)** | **Overhead cycles (%)** |
|--------------|------------|-------------------------|--------------------------|
| Medium       | PSNR       | 0.54                    | 0.43                     |
|              | SSIM       | 0.98                    | 1.55                     |
|              | pVMAF      | 3.17                    | 3.93                     |

#### Inference Time of Various VQM Metrics
| Metric | Time (ms) per frame |
|---|---|
| PSNR-HVS | 63.8 |
| MS-SSIM | 528.39 |
| VMAF | 112.84 |
| pVMAF | 3.17 |

**⚠️Note:** VMAF, PSNR-Y, and MS-SSIM scores were computed using the implementations available in libvmaf, while PSNR-Y, SSIM, and pVMAF were computed using the implementations in x264.

### What is the purpose of this activity?
Our motivation for this work is to advance research on perception based video quality metrics tailored for real-time applications. We encourage the community to explore and evaluate pVMAF within the x264 framework. Additionally, we have made all candidate features accessible, inviting further experimentation and the extension of support to a broader range of encoder settings and content types.

### Contributors
- Axel De Decker - Software Engineer, CTO Office, Synamedia
- Sangar Sivashanmugam - Senior Video Algorithm Engineer, Research & Development, Synamedia
- Jan Codenie - Senior Lead Software Engineer, CTO Office, Synamedia
- Dr. Jan De Cock - Director of Codec Development, CTO Office, Synamedia

### Acknowledgements
This research was supported by Synamedia. We extend our gratitude to Professor Glenn Van Wallendael of Ghent University---imec for his insightful guidance. Additionally, we thank the following individuals for their contributions to the pVMAF project, in no particular order: Hsiang-Yeh Wang, Yongjian Li, Chris Warrington, Steve Warrington, John Lan, Chris Coene, Marc Baillavoine, Karl Stoll (Synamedia), Longji Wang, Marcus Rosen, Lin Zheng, and Cheng-Yu Pai (formerly with Synamedia). We would like to thank the creators and contributors of x264 repository. We would also like to acknowledge and give due credit to the creators of the sample [clip](https://media.xiph.org/video/av2/y4m/WorldCupFarSky_1920x1080_30p.y4m), which is linked here for public use in testing and experimentation.

### Development Notes
**pVMAF v1.2** : This version of pVMAF model is optimized and tuned for FHD progressive content with pixel format YUV420p encoded on `--medium` preset settings without any of the `--tune` options. With development process still on-going for tuning the model preformance on a wider range of settings, please expect changes to the model weights.

### Dev System Info
gcc : gcc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-3.0.2)\
nasm : NASM version 2.15.03 compiled on Apr  8 2021\
OS : Oracle Linux Server 8.9\
Architecture : x86-64

### License
Click [here](/COPYING) for License information.

### Citation
**Coming soon**
