import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from torch.utils.data import sampler
import json

# ========================================================
# Sampler functions
# ========================================================
class SubsetSampler(sampler.Sampler):
    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class BalancedSampler(sampler.Sampler):
    def __init__(self, data_source, n_samples):
        self.data_source = data_source
        self.n_samples = n_samples
        self.n = len(self.data_source)
        self.nf = (self.data_source.labels!=0).sum()
        self.nb = (self.data_source.labels==0).sum()

        self.nb_ind = (self.data_source.labels==0)
        self.nf_ind = (self.data_source.labels!=0)
        
    def __iter__(self):
        p = np.ones(len(self.data_source))
        p[self.nf_ind] =  self.nf 
        p[self.nb_ind] =  self.nb
        p = p / p.sum()

        indices = np.random.choice(np.arange(self.n), 
                                   self.n_samples, 
                                   replace=False, 
                                   p=p)
        # self.data_source.labels[indices]
        return iter(indices)

    def __len__(self):
        return self.n_samples

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

    
Habitats_list = \
["7117",
"7393",
"7398",
"7426",
"7434",
"7463",
"7482",
"7490",
"7585",
"7623",
"9852",
"9862",
"9866",
"9870",
"9892",
"9894",
"9898",
"9907",
"9908"]

Habitats_dict = \
{"7117":	"Rocky Mangrove - prop roots"
,"7268":	"Sparse algal bed"
,"7393":	"Upper Mangrove – medium Rhizophora"
,"7398":	"Sandy mangrove prop roots"
,"7426":	"Complex reef"
,"7434":	"Low algal bed"
,"7463":	"Seagrass bed"
,"7482":	"Low complexity reef"
,"7490":	"Boulders"
,"7585":	"Mixed substratum mangrove - prop roots"
,"7623":	"Reef trench"
,"9852":	"Upper mangrove - tall rhizophora"
,"9862":	"Large boulder"
,"9866":	"Muddy mangrove - pneumatophores and trunk"
,"9870":	"Muddy mangrove – pneumatophores"
,"9892":	"Bare substratum"
,"9894":	"Mangrove - mixed pneumatophore prop root"
,"9898":	"Rocky mangrove - large boulder and trunk"
,"9907":	"Rock shelf"
,"9908":	"Large boulder and pneumatophores"}


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url