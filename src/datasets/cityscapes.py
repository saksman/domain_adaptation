import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

class CityscapesDataSet:
    def __init__(self, root, datadir, split=None, transform=None,n_samples=None):
        # self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.split = split
        self.datadir = datadir
        self.n_classes = 19
        self.transform = transform
        # self.crop_size = (2048, 1024) # TODO: maybe upsample?
        self.crop_size = (1280, 720)
        # self.scale = scale
        # self.ignore_label = ignore_label
        # self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        # self.mean = (128, 128, 128)
        # self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(osp.join(datadir, "%s.txt" % self.split))]
        
        if self.split == "val":
            self.label_ids = [i_id.strip() for i_id in open(osp.join(datadir, "label.txt"))]
        # if not max_iters==None:
        #     self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        with open(osp.join(self.datadir, 'info.json'), 'r') as fp:
            info = json.load(fp)
        num_classes = np.int(info['classes'])
        print('Num classes', num_classes)
        name_classes = np.array(info['label'], dtype=np.str)
        mapping = np.array(info['label2train'], dtype=np.int)
        hist = np.zeros((num_classes, num_classes))


        if self.split == "val":
            for i in range(len(self.img_ids)):
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.split, self.img_ids[i]))
                label_file = osp.join(self.root, "gtFine/%s/%s" % (self.split, self.label_ids[i]))

                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": self.img_ids[i],
                    "label_meta": {"name_classes": name_classes,
                             "mapping": mapping,
                             "hist": hist}
                })
        elif self.split == "train":
            for name in self.img_ids:
                img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.split, name))

                self.files.append({
                    "img": img_file,
                    "name": name
                })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        # print(datafiles["label"])
        # print(label)
        # print(label)
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        label = label_mapping(np.array(label), datafiles["label_meta"]["mapping"])

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))


        batch = {"images": image.copy(),
                 "mask_classes": label.copy(),
                 "meta": {"index": index,
                          "image_id": index,
                          "split": self.split}}
        
        # return image.copy(), label_copy.copy(), np.array(size), name
        return batch

    