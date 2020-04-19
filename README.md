
# AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation
This repository is for AdaCoF introduced in the following paper

[Hyeongmin Lee](https://hyeongminlee.github.io/), [Taeoh Kim](https://taeoh-kim.github.io/), Tae-young Chung, Daehyun Pak, Yuseok Ban, and Sangyoun Lee, "AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation", CVPR 2020, [[arXiv](https://arxiv.org/abs/1907.10244)], [[Video](https://www.youtube.com/watch?v=Z3q0YrBsNJc)]

<a href="https://arxiv.org/abs/1907.10244v3" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1907.10244v3.pdf.jpg" alt="Paper" width="100%"></a>

## Contents
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Train](#train)
4. [Test](#test)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Video frame interpolation is one of the most challenging tasks in video processing research. Recently, many studies based on deep learning have been suggested. Most of these methods focus on finding locations with useful information to estimate each output pixel using their own frame warping operations. However, many of them have Degrees of Freedom (DoF) limitations and fail to deal with the complex motions found in real world videos. To solve this problem, we propose a new warping module named Adaptive Collaboration of Flows (AdaCoF). Our method estimates both kernel weights and offset vectors for each target pixel to synthesize the output frame. AdaCoF is one of the most generalized warping modules compared to other approaches, and covers most of them as special cases of it. Therefore, it can deal with a significantly wide domain of complex motions. To further improve our framework and synthesize more realistic outputs, we introduce dual-frame adversarial loss which is applicable only to video frame interpolation tasks. The experimental results show that our method outperforms the state-of-the-art methods for both fixed training set environments and the Middlebury benchmark.

![Architecture](/images/Architecture.PNG)
The network architecture.

![VisualComp](/images/visual_comparison.PNG)
Visual Comparisons.

![OffsetVis](/images/offset_visualization.PNG)
Offset Visualizations.

## Environment
- GPU: GTX1080Ti
- Ubuntu 16.04.4
- CUDA 10.0
- python 3.6
- torch 1.2.0
- torchvision 0.4.0
- cupy 6.2.0
- scipy 1.3.1
- pillow 6.1.0
- numpy 1.17.0

## Train
### Prepare training data 

1. Download Vimeo90k training data from [vimeo triplet dataset](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip).

2. In train.py, specify '--train' based on the directory of 'vimeo_triplet'.

For more informaiton about Vimeo90k, please refer to [TOFlow](https://github.com/anchen1011/toflow).

### Begin to train

1. Run train.py with following command.
    ```bash
    python train.py --train [dir_to_vimeo_triplet] --out_dir [dir_to_output_folder]
    ```
2. You can change many other options (epochs, learning rate, hyper parameters, etc.) in train.py.

3. Then you will have the output folder (out_dir) that contains the checkpoints, result images and the configuration file of the training.

## Test
### Evaluation
1. For evaluation, you need the checkpoint file and configuration (configuration is optional).

2. You can use your own trained checkpoint, or we provide our pre-trined model in './checkpoint'.

3. You can set the hyper parameters (kernel size and dilation) manually or you can use config.txt files.

4. Run evaluation.py with following command.

    ```bash
    python evaluation.py --out_dir [output_dir] --checkpoint [checkpoint_dir] --config [configuration_dir]
    ```
5. Then you will have the output folder (out_dir) that contains the results on the test sets 'middlebury_eval', 'middlebury_others', 'davis', 'ucf101'.

### Video Interpolation
1. To interpolate an arbitrary video you have, create a directory which contains the frames of the video like 'sample_video' directory.
2. The starting frame index and zero-padding of indexing can be changed with '--index_from' and '--zpad'.
3. Run interpolate_video.py with following command.

    ```bash
    python interpolate_video.py --input_video [input_video_frames_dir] --output_video [output_video_frames_dir] --checkpoint [checkpoint_dir] --config [configuration_dir]
    ```
4. Then you will have the output folder (output_video_frames_dir) that contains the output video frames.

### Two-frame interpolation
1. To interpolate a frame between arbitrary two frames you have, run interpolate_video.py with following command.

    ```bash
    python interpolate_twoframe.py --first_frame [first_frame] --second_frame [second_frame] --output_frame [output_frame] --checkpoint [checkpoint_dir] --config [configuration_dir]
    ```
2. Then you will have the interpolated output frame.

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{lee2020adacof,
    title={AdaCoF: Adaptive Collaboration of Flows for Video Frame Interpolation},
    author={Hyeongmin Lee, Taeoh Kim, Tae-young Chung, Daehyun Pak, Yuseok Ban, and Sangyoun Lee},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020}
}
```
## Acknowledgements
This code is based on [yulunzhang/RCAN](https://github.com/yulunzhang/RCAN)

