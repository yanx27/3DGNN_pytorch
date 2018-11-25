# *_*coding:utf-8 *_*
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import h5py
from models import Model
from torch.autograd import Variable

def show_grapth(img):
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.axis('off')
    plt.show()

data_path = 'datasets/data/'
data_file = 'nyu_depth_v2_labeled.mat'
f = h5py.File(data_path + data_file)
rgb_images_fr = np.transpose(f['images'], [0, 2, 3, 1]).astype(np.float32)
depth_images_fr = np.array((f['depths']))
label_images_fr = np.array(f['labels'])
class_name = f['names']

show_grapth(rgb_images_fr[0])
show_grapth(depth_images_fr[0])
show_grapth(label_images_fr[0])

rgb = rgb_images_fr[0].astype(np.float32)
hha = rgb_images_fr[0].astype(np.float32)
rgb_hha = np.concatenate([rgb, hha], axis=2).astype(np.float32)
rgb_hha = torch.Tensor(rgb_hha)
rgb_hha = rgb_hha.unsqueeze(0)
xy = torch.Tensor(np.zeros_like(rgb)[:,:,0:2].astype(np.float32))
xy = xy.unsqueeze(0)
xy = xy.permute(0, 3, 1, 2).contiguous()
input = rgb_hha.permute(0, 3, 1, 2).contiguous()
input = input.type(torch.FloatTensor)

model = Model(14, 1,use_gpu = False)
output = model(Variable(input), gnn_iterations=3, k=64, xy=xy, use_gnn=True)

from torchviz import make_dot
draw = make_dot(output, params=dict(model.named_parameters()))
draw.view()