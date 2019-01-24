from torch.utils.data import Dataset
import glob
import numpy as np
import cv2
import h5py


class Dataset(Dataset):
    def __init__(self, flip_prob=None, crop_type=None, crop_size=0):

        self.flip_prob = flip_prob
        self.crop_type = crop_type
        self.crop_size = crop_size

        data_path = 'datasets/data/'
        data_file = 'nyu_depth_v2_labeled.mat'

        # read mat file
        print("Reading .mat file...")
        f = h5py.File(data_path + data_file)

        # as it turns out, trying to pickle this is a shit idea :D

        rgb_images_fr = np.transpose(f['images'], [0, 2, 3, 1]).astype(np.float32)
        label_images_fr = np.array(f['labels'])

        f.close()

        self.rgb_images = rgb_images_fr
        self.label_images = label_images_fr


    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb = self.rgb_images[idx].astype(np.float32)
        hha = np.transpose(cv2.imread("datasets/data/hha/" + str(idx+1) + ".png", cv2.COLOR_BGR2RGB), [1, 0, 2])
        rgb_hha = np.concatenate([rgb, hha], axis=2).astype(np.float32)
        label = self.label_images[idx].astype(np.float32)
        label[label >= 14] = 0
        xy = np.zeros_like(rgb)[:,:,0:2].astype(np.float32)

        # random crop
        if self.crop_type is not None and self.crop_size > 0:
            max_margin = rgb_hha.shape[0] - self.crop_size
            if max_margin == 0:  # crop is original size, so nothing to crop
                self.crop_type = None
            elif self.crop_type == 'Center':
                rgb_hha = rgb[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
                label = label[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2]
                xy = xy[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
            elif self.crop_type == 'Random':
                x_ = np.random.randint(0, max_margin)
                y_ = np.random.randint(0, max_margin)
                rgb_hha = rgb_hha[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
                label = label[y_:y_ + self.crop_size, x_:x_ + self.crop_size]
                xy = xy[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
            else:
                print('Bad crop')  # TODO make this more like, you know, good software
                exit(0)

        # random flip
        if self.flip_prob is not None:
            if np.random.random() > self.flip_prob:
                rgb_hha = np.fliplr(rgb_hha).copy()
                label = np.fliplr(label).copy()
                xy = np.fliplr(xy).copy()

        return rgb_hha, label, xy

