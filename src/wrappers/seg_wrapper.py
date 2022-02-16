import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
import os
import numpy as np
import time
from src import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from torch.autograd import Variable
from src import wrappers, models

from haven import haven_utils as hu
from active_learning import ActiveLearning
from active_loss import LossPredictionLoss
from active_learning_utils import *
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from PIL import Image

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

def H(x):
    '''Calculate entropy of a float'''
    return -1 * torch.sum(torch.exp(x) * x, dim=1)
    
def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = ut.CrossEntropy2d().cuda()
    return criterion(pred, label)

class SegWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = ActiveLearning(model)
        self.opt = opt
        self.num_steps = 250000
        self.power = 0.9
        self.learning_rate = 2.5e-4
        self.lambda_adv_target = 0.001
        self.learning_rate_D = 1e-4
        # self.loss_lambda = 0.01
        self.loss_lambda = 0.1
        self.clue_softmax_T = 1.0
        
    def val_on_loader(self, val_loader, n_classes):
        self.n_classes = n_classes
        val_monitor = SegMonitor(self.n_classes)
        return wrappers.val_on_loader(self, val_loader, val_monitor=val_monitor, n_classes=n_classes)

    def vis_on_loader(self, vis_loader, savedir): # TODO: add n_classes
        return wrappers.vis_on_loader(self, vis_loader, savedir=savedir)

    def vis_on_loader_no_gt_mask(self, vis_loader, savedir): # TODO: add n_classes
        return wrappers.vis_on_loader_no_gt_mask(self, vis_loader, savedir=savedir)
    
    def train_on_loader(self, model, train_loader, val_loader, domain_adaptation, sampling_strategy, n_classes, n=None):
        
        model.train()
        train_monitor = TrainMonitor()
        print('Training')
        
        val_loader_iterator = iter(val_loader)
    
        # initialize discriminator model D
        model_D = models.discriminator.FCDiscriminator(num_classes=2)
        model_D.train()
        model_D.cuda()
        optimizer_D = optim.Adam(model_D.parameters(), lr=self.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D.zero_grad()
        bce_loss = torch.nn.MSELoss()
        
        # labels for adversarial training
        source_label = 0
        target_label = 1
        
        # weighted sample entropy list for AADA
        sample_weights = {}
        
        for i, train_batch in enumerate(tqdm.tqdm(train_loader)):
            # Normal training without domain adaptation
            if domain_adaptation == 0:
                self.opt.zero_grad()
                self.train()

                images = train_batch["images"].cuda()
                logits, loss_pred = self.model.forward(images)
                
                if n_classes == 2:
                    p_log = F.log_softmax(logits, dim=1)
                    p = F.softmax(logits, dim=1)
                    FL = p_log*(1.-p)**2.
                    loss = F.nll_loss(FL, train_batch["mask_classes"].cuda().long(), reduction='mean')
                else:
                    loss = loss_calc(logits, train_batch["mask_classes"])
                loss.backward()
                
                self.opt.step()
            # Unsupervised domain adaptation
            else:
                try:
                    val_batch = next(val_loader_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    val_batch = next(val_loader_iterator)
                
                loss_seg_value = 0
                loss_adv_target_value = 0
                loss_D_value = 0
                
                self.opt.zero_grad()
                optimizer_D.zero_grad()

                self.train()

                images = train_batch["images"].cuda()
                logits, loss_pred = self.model.forward(images)
                
                if n_classes == 2:
                    p_log = F.log_softmax(logits, dim=1)
                    p = F.softmax(logits, dim=1)
                    FL = p_log*(1.-p)**2.
                    loss = F.nll_loss(FL, train_batch["mask_classes"].cuda().long(), reduction='mean')
                else:
                    loss = loss_calc(logits, train_batch["mask_classes"])
                loss.backward()

                for param in model_D.parameters():
                    param.requires_grad = False
                    
                target_images = val_batch["images"].cuda()
                target_logits, target_loss_pred = self.model.forward(target_images)
                D_out = model_D(F.softmax(target_logits))

                loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
                loss = self.lambda_adv_target * loss_adv_target
                loss.backward()
                loss_adv_target_value += loss_adv_target.data.cpu().numpy()

                # train D on source
                # bring back requires_grad
                for param in model_D.parameters():
                    param.requires_grad = True

                logits_detached = logits.detach()
                D_out = model_D(F.softmax(logits_detached))
                loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda()))
                loss_D.backward()
                loss_D_value += loss_D.data.cpu().numpy()

                # train D on target
                target_logits_detached = target_logits.detach()
                D_out = model_D(F.softmax(target_logits_detached))
                loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda()))
                loss_D.backward()
                loss_D_value += loss_D.data.cpu().numpy()

                self.opt.step()
                optimizer_D.step()

            # Active DA step
            if sampling_strategy == "aada" and domain_adaptation == 1:
        
                mean_D_out = D_out.mean(-1).mean(-1) 
                mean_logits = target_logits_detached.mean(-1).mean(-1) # num logits = num samples
                
                # weight = mean discriminator output per batch
                w = (1 - mean_D_out) / mean_D_out
                
                # samples = discriminator output * entropy
                s = w * H(mean_logits)
                s = s.mean(0).detach().cpu().tolist()
                
                batch_size = len(val_batch['meta']['index'])
                for i in range(batch_size):
                    idx = val_batch['meta']['index'][i].item()
                    sample_weights.update({idx: s[i]})

            elif sampling_strategy == "learned_loss" and domain_adaptation == 0:
                criterion_lp = LossPredictionLoss()

                # Format training loss for AL function
                al_train_loss = loss.unsqueeze(0)
                # Concatenate losses for each item in batch 
                final_al_train_loss = torch.cat([al_train_loss, al_train_loss]).unsqueeze(1)

                lp = self.loss_lambda * criterion_lp(loss_pred, final_al_train_loss)
                loss += lp
            # TODO: not completely implemented yet
            elif sampling_strategy == "ada_clue" and domain_adaptation == 0:
                # Replicate ADA-CLUE except with logits instead of penultimate layer embeddings
                logits_detached = logits.sum(1).sum(2).detach().cpu().numpy()
                scores = F.softmax(logits / self.clue_softmax_T, dim=1)
                scores += 1e-8
                sample_weights = -(scores * torch.log(scores)).sum(1).sum(2).detach().cpu().numpy()
                
                # Run K-means to produce cluster centroids 
                km = KMeans(n)
                km.fit(logits_detached, sample_weight=sample_weights)
                
                # Find nearest neighbors to inferred centroids
                dists = euclidean_distances(km.cluster_centers_, logits_detached)
                sort_idxs = dists.argsort(axis=1)
                q_idxs = []
                ax, rem = 0, n
                while rem > 0:
                    q_idxs.extend(list(sort_idxs[:, ax][:rem]))
                    q_idxs = list(set(q_idxs))
                    rem = n-len(q_idxs)
                    ax += 1
                print(q_idxs)
            

        score_dict = {"train_seg_nll_loss":loss.item()}
        train_monitor.add(score_dict)
        return train_monitor.get_avg_score(), sample_weights

    def val_on_batch(self, batch, n_classes, **extras):
        pred_seg, loss_pred = self.predict_on_batch(batch)
        cm_pytorch = confusion(torch.from_numpy(pred_seg).cuda().float(), 
                                batch["mask_classes"].cuda().float(), 
                                n_classes=n_classes)
                
        return cm_pytorch, loss_pred


    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask

    def predict_on_batch(self, batch):
        self.eval()
        images = batch["images"].cuda()
        output, loss_pred = self.model.forward(images)
        pred_mask = output.data.max(1)[1].squeeze().cpu().numpy()
        return pred_mask[None], loss_pred

    def vis_on_batch(self, batch, savedir_image):
        from skimage.segmentation import mark_boundaries
        from skimage import data, io, segmentation, color
        from skimage.measure import label
        self.eval()
        pred_mask, _ = self.predict_on_batch(batch)

        # img = hu.get_image(batch["images"], denorm="rgb") # replaced with line below b/c this was causing errors, need to debug
        img = hu.get_image(batch["images"])
        img_np = np.array(img)
        pm = pred_mask.squeeze()
        out = color.label2rgb(label(pm), image=(img_np), image_alpha=1.0, bg_label=0)
        img_mask = mark_boundaries(out.squeeze(),  label(pm).squeeze())
        out = color.label2rgb(label(batch["mask_classes"][0]), image=(img_np), image_alpha=1.0, bg_label=0)
        img_gt = mark_boundaries(out.squeeze(),  label(batch["mask_classes"]).squeeze())
        hu.save_image(savedir_image, np.hstack([img_gt, img_mask]))
        
    def vis_on_batch_no_gt_mask(self, batch, savedir_image):
        from skimage.segmentation import mark_boundaries
        from skimage import data, io, segmentation, color
        from skimage.measure import label

        self.eval()
        pred_mask, _ = self.predict_on_batch(batch)
        pm = pred_mask.squeeze()
        
        img = hu.get_image(batch["images"])
        img_np = np.array(img)
        pm = pred_mask.squeeze()
        out = color.label2rgb(label(pm), image=(img_np), kind='overlay', image_alpha=1.0, bg_label=0)
        img_mask = mark_boundaries(out.squeeze(),  label(pm).squeeze())
        hu.save_image(savedir_image, img_mask)


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
        if n_classes == 2:
            # return -1 
            Inter = np.diag(self.cf)
            G = self.cf.sum(axis=1)
            P = self.cf.sum(axis=0)
            union = G + P - Inter

            nz = union != 0
            mIoU = Inter[nz] / union[nz]
            mIoU = np.mean(mIoU)
        else:
            mIoU = np.mean(per_class_iu(self.cf))


        return {"val_seg_mIoU": mIoU}
          
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

# seg
def confusion(prediction, truth, n_classes):
    if n_classes == 2:
        confusion_vector = prediction / truth
        tp = torch.sum(confusion_vector == 1).item()
        fp = torch.sum(confusion_vector == float('inf')).item()
        tn = torch.sum(torch.isnan(confusion_vector)).item()
        fn = torch.sum(confusion_vector == 0).item()
        cm = np.array([[tn,fp],[fn,tp]])
    else:
        hist = np.zeros((n_classes, n_classes))
        assert len(truth.flatten()) == len(prediction.flatten())
        cm = fast_hist(truth.cpu().numpy().flatten().astype(int), prediction.cpu().numpy().flatten().astype(int), n_classes)
    return cm

