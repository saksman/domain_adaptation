import torch
import numpy as np
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision
from haven import haven_utils as hu
from haven import haven_chk as hc
from src import datasets, models
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler
from src import wrappers
from haven import haven_wizard as hw
import warnings
warnings.filterwarnings("ignore")

def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """


    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.use_cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    else:
        device = 'cpu'

    datadir = args.datadir
    
    if exp_dict["dataset"] == "fish_seg":
        if args.domain_shift:
            src_train_datadir = datadir + '/domain_adaptation/source/'
            
            val_datadir = datadir + '/domain_adaptation/target/'

        test_datadir = datadir + '/domain_adaptation/target/test/'
    elif exp_dict["dataset"] == "gta5cityscapes" and args.domain_shift:
        src_train_datadir = '/global/cfs/cdirs/m3691/segmentation_datasets/GTA5/'
        src_train_labeldir = datadir + '/gta5cityscapes/gta5_list/train.txt'
        target_train_datadir = '/global/cfs/cdirs/m3691/segmentation_datasets/CITYSCAPES/'
        target_train_labeldir = datadir + '/gta5cityscapes/cityscapes_list/'
        target_val_datadir = '/global/cfs/cdirs/m3691/segmentation_datasets/CITYSCAPES/'
        target_val_labeldir = datadir + '/gta5cityscapes/cityscapes_list/'
        
        
    print('Running on device: %s' % device)
    
    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

    score_list = []

    # Checkpointing
    # =============
    score_list_path = os.path.join(savedir, "score_list.pkl")
    model_path = os.path.join(savedir, "model_state_dict.pth")
    opt_path = os.path.join(savedir, "opt_state_dict.pth")

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = hu.load_pkl(score_list_path)
        model.load_state_dict(torch.load(model_path))
        opt.load_state_dict(torch.load(opt_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Load datasets
    if args.domain_shift and exp_dict["dataset"] == "fish_seg":
        n_classes = 2
        src_train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                         split="train", 
                                         transform=exp_dict.get("transform"),
                                         datadir=src_train_datadir)
        target_train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                         split="target_train", 
                                         transform=exp_dict.get("transform"),
                                         datadir=val_datadir)
        target_val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="target_val",
                                           transform=exp_dict.get("transform"),
                                           datadir=val_datadir)
        unlabeled_idx = list(range(len(target_train_set)))
    elif args.domain_shift and exp_dict["dataset"] == "gta5cityscapes":
        n_classes = 19
        src_train_set = datasets.get_dataset(dataset_name='gta5',
                                         root=src_train_datadir,
                                         datadir=src_train_labeldir,
                                         split="train", 
                                         transform=exp_dict.get("transform"))
        target_train_set = datasets.get_dataset(dataset_name='cityscapes',
                                         root=target_train_datadir,
                                         split="train", 
                                         transform=exp_dict.get("transform"),
                                         datadir=target_train_labeldir)
        target_val_set = datasets.get_dataset(dataset_name='cityscapes', 
                                           root=target_val_datadir,
                                           split="val",
                                           transform=exp_dict.get("transform"),
                                           datadir=target_val_labeldir)
        unlabeled_idx = list(range(len(target_train_set)))
    else:
        train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                         split="train", 
                                         transform=exp_dict.get("transform"),
                                         datadir=datadir)
        val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                           transform=exp_dict.get("transform"),
                                           datadir=datadir)

        unlabeled_idx = list(range(len(train_set)))
        
    sampling_strategy = args.sampling_strategy
    rand_state = np.random.RandomState(1311)
    
    if sampling_strategy != 'None':
        print("The sampling strategy is: ", sampling_strategy)
        
        n_samples = args.n_samples
        print("# samples: ", n_samples)
        
        if sampling_strategy == "learned_loss":
            n_random = min(n_samples, 40)
            rand_idx = rand_state.choice(unlabeled_idx, n_random, replace=False)
        else:
            rand_idx = rand_state.choice(unlabeled_idx, n_samples, replace=False)
        for id in rand_idx:
            unlabeled_idx.remove(id)
    
    # Run training and validation
    for epoch in range(s_epoch, args.n_epochs):     
        score_dict = {"epoch": epoch}
        
        # Active learning on entire source domain
        if sampling_strategy != "None" and args.domain_adaptation == 0 and args.domain_shift == 0:
            if sampling_strategy == "random" or (epoch < 4 and sampling_strategy == "learned_loss"):
                print("random!")
                print(rand_idx)
                train_loader = DataLoader(train_set, 
                        sampler=ut.SubsetSampler(train_set, indices=rand_idx),
                        batch_size=exp_dict["batch_size"]) 
            elif sampling_strategy == "learned_loss" and epoch == 4:
                print("active learning epoch #4: choosing labels for learning!")
                # Set labels for active learning once
                train_loader = DataLoader(train_set, 
                        sampler=ut.SubsetSampler(train_set, indices=unlabeled_idx),
                        batch_size=exp_dict["batch_size"]) 

                with torch.no_grad():
                    score, losses = model.val_on_loader(train_loader, n_classes=n_classes)
                    losses = np.array(losses)
                    idx = losses.argsort()[-n_samples:][::-1]

                    new_labeled_idx = []
                    for id in idx:
                        new_labeled_idx.append(unlabeled_idx[id])
                        
                    new_labeled_idx.extend(rand_idx)
                    print(new_labeled_idx)
                    train_loader = DataLoader(train_set, 
                                        sampler=ut.SubsetSampler(train_set, indices=new_labeled_idx),
                                        batch_size=exp_dict["batch_size"])

            elif sampling_strategy == "learned_loss" and epoch > 4:
                print("active learning after epoch #4!")
                print(new_labeled_idx)
                train_loader = DataLoader(train_set, 
                                        sampler=ut.SubsetSampler(train_set, indices=new_labeled_idx),
                                        batch_size=exp_dict["batch_size"])
            # TODO: not completely implemented.
            elif sampling_strategy == "ada_clue":
                train_loader = DataLoader(train_set, 
                        sampler=ut.SubsetSampler(train_set, indices=rand_idx),
                        batch_size=exp_dict["batch_size"]) 
                val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
                model.train_on_loader(model, train_loader, val_loader, args.domain_adaptation, args.sampling_strategy, n_samples, n_classes=n_classes)
                
            val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
            
            # train
            score, _ = model.train_on_loader(model, train_loader, val_loader, args.domain_adaptation, args.sampling_strategy, n_classes=n_classes)
            score_dict.update(score)
            # Add score_dict to score_list
            score_list += [score_dict]
            
            # validate
            score, losses = model.val_on_loader(val_loader, n_classes=n_classes)
            score_dict.update(score)
            
            # visualize
            if exp_dict["dataset"] == "fish_seg":
                vis_loader_val = DataLoader(val_set, sampler=ut.SubsetSampler(val_set, indices=[0, 2, 4, 10, 12, 25]),
                                            batch_size=1)

                model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
            
        # Train on source dataset with a domain shift, with optional domain adaptation (no active learning)
        elif args.domain_shift:
            src_train_loader = DataLoader(src_train_set, shuffle=False, batch_size=exp_dict["batch_size"]) 
            target_train_loader = DataLoader(target_train_set, shuffle=False, batch_size=exp_dict["batch_size"])
            target_val_loader = DataLoader(target_val_set, shuffle=False, batch_size=exp_dict["batch_size"])
            
            # Collect sample weights on last DA epoch for AADA 
            if sampling_strategy == "aada" and args.domain_adaptation and epoch == (args.n_epochs - 1):
                
                # train
                score, sample_weights = model.train_on_loader(model, src_train_loader, target_train_loader, args.domain_adaptation, args.sampling_strategy, n_classes=n_classes)
                score_dict.update(score)
                
                # Choose samples with highest uncertainty and diversity for active learning
                sample_weights_sorted = sorted(sample_weights, key=sample_weights.get)
                aada_idx = [*sample_weights_sorted][:n_samples]
            else:
                # train
                score, _ = model.train_on_loader(model, src_train_loader, target_train_loader, args.domain_adaptation, args.sampling_strategy, n_classes=n_classes)
                score_dict.update(score)
                
                
            # Add score_dict to score_list
            score_list += [score_dict]
            
            score, losses = model.val_on_loader(target_val_loader, n_classes=n_classes)
            score_dict.update(score)
         
            # visualize
            vis_loader_val = DataLoader(target_val_set, sampler=ut.SubsetSampler(target_val_set, indices=[0, 2, 4, 10, 12, 25]),
                                        batch_size=1)
            model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
 
            
        # Train on entire source dataset (base case)
        else:
            train_loader = DataLoader(train_set, shuffle=False, batch_size=exp_dict["batch_size"]) 
            val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
            
            # train
            score, _ = model.train_on_loader(model, train_loader, train_loader, args.domain_adaptation, args.sampling_strategy, n_classes=n_classes)
            score_dict.update(score)
            # Add score_dict to score_list
            score_list += [score_dict]
            
            # validate
            score, losses = model.val_on_loader(val_loader, n_classes=n_classes)
            score_dict.update(score)
            
            # visualize on validation set
            vis_loader_val = DataLoader(val_set, sampler=ut.SubsetSampler(val_set, indices=[0, 2, 4, 10, 12, 25]),
                                        batch_size=1)
            model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
        
        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved in %s" % savedir)
    
    # Active learning + domain adaptation
    if args.domain_adaptation and sampling_strategy != "None": 
        
        # Domain shift dataset
        if args.domain_shift:
            unlabeled_idx = list(range(len(target_train_set)))
            if sampling_strategy == "random":
                rand_idx = rand_state.choice(unlabeled_idx, n_samples, replace=False)
                print(rand_idx)
                target_train_loader = DataLoader(target_train_set, 
                        sampler=ut.SubsetSampler(target_train_set, indices=rand_idx),
                        batch_size=exp_dict["batch_size"]) 
            elif sampling_strategy == "learned_loss":
                # Set labels for active learning once
                target_train_loader = DataLoader(target_train_set, 
                        sampler=ut.SubsetSampler(target_train_set, indices=unlabeled_idx),
                        batch_size=exp_dict["batch_size"]) 

                with torch.no_grad():
                    score, losses = model.val_on_loader(target_train_loader, n_classes=n_classes)
                    losses = np.array(losses)
                    idx = losses.argsort()[-n_samples:][::-1]

                    new_labeled_idx = []
                    for id in idx:
                        new_labeled_idx.append(unlabeled_idx[id])

                    print(new_labeled_idx)
                    target_train_loader = DataLoader(target_train_set, 
                                        sampler=ut.SubsetSampler(target_train_set, indices=new_labeled_idx),
                                        batch_size=exp_dict["batch_size"]) 
            elif sampling_strategy == "aada":
                # Set labels for active learning once
                target_train_loader = DataLoader(target_train_set, 
                        sampler=ut.SubsetSampler(target_train_set, indices=aada_idx),
                        batch_size=exp_dict["batch_size"]) 
            
            for epoch in range(s_epoch, args.n_epochs):
                # train
                score, _ = model.train_on_loader(model, target_train_loader, target_val_loader, 0, args.sampling_strategy, n_classes=n_classes)
                score_dict.update(score)
                # Add score_dict to score_list
                score_list += [score_dict]

                # validate
                score, losses = model.val_on_loader(target_val_loader, n_classes=n_classes)
                score_dict.update(score)

            # visualize on validation set
            vis_loader_val = DataLoader(target_val_set, sampler=ut.SubsetSampler(target_val_set, indices=[0, 2, 4, 10, 12, 25]),
                                        batch_size=1)
            model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
        
        # Base case dataset
        else:
            unlabeled_idx = list(range(len(train_set)))
            if sampling_strategy == "random":
                rand_idx = rand_state.choice(unlabeled_idx, n_samples, replace=False)
                print(rand_idx)
                train_loader = DataLoader(train_set, 
                        sampler=ut.SubsetSampler(train_set, indices=rand_idx),
                        batch_size=exp_dict["batch_size"]) 
            elif sampling_strategy == "learned_loss":
                # Set labels for active learning
                train_loader = DataLoader(train_set, 
                        sampler=ut.SubsetSampler(train_set, indices=unlabeled_idx),
                        batch_size=exp_dict["batch_size"]) 

                with torch.no_grad():
                    score, losses = model.val_on_loader(train_loader, n_classes=n_classes)
                    losses = np.array(losses)
                    idx = losses.argsort()[-n_samples:][::-1]

                    new_labeled_idx = []
                    for id in idx:
                        new_labeled_idx.append(unlabeled_idx[id])

                    print(new_labeled_idx)
                    train_loader = DataLoader(train_set, 
                                        sampler=ut.SubsetSampler(train_set, indices=new_labeled_idx),
                                        batch_size=exp_dict["batch_size"])
            
            for epoch in range(s_epoch, args.n_epochs):
                # train
                score, _ = model.train_on_loader(model, train_loader, val_loader, 0, args.sampling_strategy, n_classes=n_classes)
                score_dict.update(score)
                # Add score_dict to score_list
                score_list += [score_dict]

                # validate
                score, losses = model.val_on_loader(val_loader, n_classes=n_classes)
                score_dict.update(score)

            # visualize on validation set
            vis_loader_val = DataLoader(val_set, sampler=ut.SubsetSampler(val_set, indices=[0, 2, 4, 10, 12, 25]),
                                        batch_size=1)
            model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
            
        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved in %s" % savedir)
    
    # Fine-tune on target
    elif args.domain_adaptation == 0 and args.domain_shift and sampling_strategy != "None":            
        if sampling_strategy == "random":
            print(rand_idx)
            target_train_loader = DataLoader(target_train_set, 
                    sampler=ut.SubsetSampler(target_train_set, indices=rand_idx),
                    batch_size=exp_dict["batch_size"]) 
        elif sampling_strategy == "learned_loss":
            # Set labels for active learning once
            target_train_loader = DataLoader(target_train_set, 
                    sampler=ut.SubsetSampler(target_train_set, indices=unlabeled_idx),
                    batch_size=exp_dict["batch_size"]) 

            with torch.no_grad():
                score, losses = model.val_on_loader(target_train_loader, n_classes=n_classes)
                losses = np.array(losses)
                idx = losses.argsort()[-n_samples:][::-1]

                new_labeled_idx = []
                for id in idx:
                    new_labeled_idx.append(unlabeled_idx[id])

                new_labeled_idx.extend(rand_idx)
                print(new_labeled_idx)
                target_train_loader = DataLoader(target_train_set, 
                                    sampler=ut.SubsetSampler(target_train_set, indices=new_labeled_idx),
                                    batch_size=exp_dict["batch_size"])

        target_val_loader = DataLoader(target_val_set, shuffle=False, batch_size=exp_dict["batch_size"])
        
        for epoch in range(s_epoch, args.n_epochs):
            # train
            score, _ = model.train_on_loader(model, target_train_loader, target_val_loader, 0, args.sampling_strategy, n_classes=n_classes)
            score_dict.update(score)
            # Add score_dict to score_list
            score_list += [score_dict]

            # validate
            score, losses = model.val_on_loader(target_val_loader, n_classes=n_classes)
            score_dict.update(score)

        # visualize on validation set
        vis_loader_val = DataLoader(target_val_set, sampler=ut.SubsetSampler(target_val_set, indices=[0, 2, 4, 10, 12, 25]),
                                    batch_size=1)
        model.vis_on_loader(vis_loader_val, savedir=os.path.join(savedir, "val_images"))
        
        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved in %s" % savedir)


if __name__ == '__main__':
    # define a list of experiments
    import exp_configs

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help='Define which exp groups to run.')
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument('-d', '--datadir', default=None,
                        help='Define the dataset directory.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("--debug",  default=False, type=int,
                        help='Debug mode.')
    parser.add_argument("-ei", "--exp_id", default=None,
                        help='Run a specific experiment based on its id.')
    parser.add_argument("-j", "--run_jobs", default=0, type=int,
                        help='Run the experiments as jobs in the cluster.')
    parser.add_argument("-nw", "--num_workers", type=int, default=0,
                        help='Specify the number of workers in the dataloader.')
    parser.add_argument("-v", "--visualize_notebook", type=str, default='',
                        help='Create a jupyter file to visualize the results.')
    parser.add_argument("-uc", "--use_cuda", type=int, default=1)
    parser.add_argument("-da", "--domain_adaptation", type=int, default=0)
    parser.add_argument("-ss", "--sampling_strategy", type=str, default='None')
    parser.add_argument("-ds", "--domain_shift", type=int, default=0)
    parser.add_argument("-n", "--n_samples", type=int, default=310)
    parser.add_argument("-ne", "--n_epochs", type=int, default=5)
    
    args, others = parser.parse_known_args()

    # Launch experiments
    hw.run_wizard(func=trainval, exp_groups=exp_configs.EXP_GROUPS, args=args)