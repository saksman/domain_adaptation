import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from . import clf_wrapper, reg_wrapper, loc_wrapper, seg_wrapper

def get_wrapper(wrapper_name, model, opt=None):
    if wrapper_name == "clf_wrapper":
        return clf_wrapper.ClfWrapper(model, opt)

    if wrapper_name == "reg_wrapper":
        return reg_wrapper.RegWrapper(model, opt)

    if wrapper_name == "loc_wrapper":
        return loc_wrapper.LocWrapper(model, opt)

    if wrapper_name == "seg_wrapper":
        return seg_wrapper.SegWrapper(model, opt)

# ===============================================
# Trainers
def train_on_loader(model, train_loader, val_loader):
    model.train()

    n_batches = len(train_loader)
    train_monitor = TrainMonitor()
    print('Training')
    for e in range(1):
        for i, batch in enumerate(tqdm.tqdm(train_loader)):
            score_dict = model.train_on_batch(batch, val_loader)
            
            train_monitor.add(score_dict)
        
    return train_monitor.get_avg_score()

@torch.no_grad()
def val_on_loader(model, val_loader, val_monitor, n_classes):
    model.eval()

    n_batches = len(val_loader)
    val_monitor = SegMonitor(n_classes=n_classes)
    losses = []
    print('Validating')
    for i, batch in enumerate(tqdm.tqdm(val_loader)):
        score, loss_pred = model.val_on_batch(batch, n_classes=n_classes)
        # print(score)
        # if type(score) == int:
        #     val_monitor.add(np.array([0, 0]))
        # else:
        val_monitor.add(score)
        losses.extend(loss_pred.cpu().numpy())
    return val_monitor.get_avg_score(), losses


@torch.no_grad()
def vis_on_loader(model, vis_loader, savedir):
    model.eval()

    n_batches = len(vis_loader)
    split = vis_loader.dataset.split
    for i, batch in enumerate(vis_loader):
        print("%d - visualizing %s image - savedir:%s" % (i, batch["meta"]["split"][0], savedir.split("/")[-2]))
        model.vis_on_batch(batch, 
        savedir_image=os.path.join(savedir, f'{i}.png'))
        
@torch.no_grad()
def vis_on_loader_no_gt_mask(model, vis_loader, savedir):
    model.eval()

    n_batches = len(vis_loader)
    split = vis_loader.dataset.split
    for i, batch in enumerate(vis_loader):
        print("%d - visualizing %s image - savedir:%s" % (i, batch["meta"]["split"][0], savedir.split("/")[-2]))
        model.vis_on_batch_no_gt_mask(batch, 
        savedir_image=os.path.join(savedir, f'{i}.png'))

@torch.no_grad()
def test_on_loader(model, test_loader):
    model.eval()
    ae = 0.
    n_samples = 0.

    n_batches = len(test_loader)
    pbar = tqdm.tqdm(total=n_batches)
    for i, batch in enumerate(test_loader):
        pred_count = model.predict(batch, method="counts")

        ae += abs(batch["counts"].cpu().numpy().ravel() - pred_count.ravel()).sum()
        n_samples += batch["counts"].shape[0]

        pbar.set_description("TEST mae: %.4f" % (ae / n_samples))
        pbar.update(1)

    pbar.close()
    score = ae / n_samples
    print({"test_score": score, "test_mae":score})

    return {"test_score": score, "test_mae":score}


class TrainMonitor:
    def __init__(self):
        self.score_dict_sum = {}
        self.n = 0

    def add(self, score_dict):
        for k,v in score_dict.items():
            if k not in self.score_dict_sum:
                self.score_dict_sum[k] = score_dict[k]
            else:
                self.n += 1
                self.score_dict_sum[k] += score_dict[k]

    def get_avg_score(self):
        return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

class SegMonitor:
    def __init__(self, n_classes):
        self.cf = None
        self.n_classes = n_classes

    def add(self, cf):
        if self.cf is None:
            self.cf = cf 
        else:
            self.cf += cf

    def per_class_iu(hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def get_avg_score(self):
        if self.n_classes == 2:
            # return -1 
            Inter = np.diag(self.cf)
            G = self.cf.sum(axis=1)
            P = self.cf.sum(axis=0)
            union = G + P - Inter

            nz = union != 0
            mIoU = Inter[nz] / union[nz]
            mIoU = np.mean(mIoU)
        else:
            print(per_class_iu(self.cf))
            mIoU = np.mean(per_class_iu(self.cf))


        return {"val_seg_mIoU": mIoU}