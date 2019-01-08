# 3DGNN for RGB-D segmentation
This is the Pytorch implementation of [3D Graph Neural Networks for RGBD Semantic Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf): 

![](https://github.com/xjqicuhk/3DGNN/blob/master/overallpipeline.png)

### Data Preparation
1. [NYU Depth V1](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html)
2. [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
3. [SUNRGB-D](http://rgbd.cs.princeton.edu/challenge.html)
4. You need transfer depth images to hha by yourself [here](https://github.com/charlesCXK/Depth2HHA) and save in `datasets/data/hha/`. 

### Emviroment
Required CUDA (8.0) + pytorch 0.4.1


