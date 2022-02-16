import pandas as pd
import numpy as np
from src import datasets
import os
from PIL import Image
import torch
import os
import os.path as osp
import random
import matplotlib.pyplot as plt
import collections
import torchvision
from torch.utils import data

class GTA5DataSet:
    def __init__(self, root, datadir, split=None, transform=None,n_samples=None):
        # self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.split = split
        self.datadir = datadir
        self.n_classes = 19
        self.transform = transform
        self.crop_size = (1280, 720)
        # self.scale = scale
        # self.ignore_label = ignore_label
        self.mean = np.array((104, 117, 123), dtype=np.int16)
        # self.mean = (128, 128, 128)
        # self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(datadir)]
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        # TODO: split by n_samples 
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        
        # mask_classes = Image.open(self.path + "/masks/"+ self.mask_names[index] + ".png").convert('L')
#             mask_classes = torch.from_numpy(np.array(mask_classes)).float() / 255.

        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image = image - self.mean
        image = image.transpose((2, 0, 1))

        
        batch = {"images": image.copy(),
                 "mask_classes": label_copy.copy(),
                 "meta": {"index": index,
                          "image_id": index}}
        
        # return image.copy(), label_copy.copy(), np.array(size), name
        return batch


# if __name__ == '__main__':
#     dst = GTA5DataSet("./data", is_transform=True)
#     trainloader = data.DataLoader(dst, batch_size=4)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         if i == 0:
#             img = torchvision.utils.make_grid(imgs).numpy()
#             img = np.transpose(img, (1, 2, 0))
#             img = img[:, :, ::-1]
#             plt.imshow(img)
#             plt.show()
