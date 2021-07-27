
'''
Semi-Supervised Active Learning with Temporal Output Discrepancy.
Siyu Huang, Tianyang Wang, Haoyi Xiong, Jun Huan, Dejing Dou.
ICCV, 2021.
'''

# General
import os
import random
import argparse
import numpy as np
import importlib

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10, SVHN, ImageFolder

# Custom
import models.resnet as resnet
from models.lossnet import *
from utils.sampler import SubsetSequentialSampler

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    if AUXILIARY == 'TOD':
        models['ema'].train()
    global iters
    
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        
        # task loss
        scores, cons_scores, features, features_list = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        loss = torch.sum(target_loss) / target_loss.size(0)
        
        # unsupervised loss
        if AUXILIARY == 'TOD':
            u_inputs, _ = next(iter(dataloaders['extra']))
            u_inputs = u_inputs.cuda()
            u_scores, cons_u_scores, features_u, u_features_list = models['backbone'](u_inputs)
            ema_scores, _, _, _ = models['ema'](inputs)
            ema_u_scores, _, _, _ = models['ema'](u_inputs)
            res_loss = F.mse_loss(scores, cons_scores) + F.mse_loss(u_scores, cons_u_scores)
            consistency_loss = F.mse_loss(cons_scores, ema_scores) + F.mse_loss(cons_u_scores, ema_u_scores)
            loss = loss + WEIGHT * (res_loss + consistency_loss)
            
        loss.backward()
        optimizers['backbone'].step()
        if AUXILIARY == 'TOD':
            update_ema_variables(models['backbone'], models['ema'], 0.999, iters)

def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _, _  = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, cycle):
    print('>> Train a Model...')
    best_acc = 0.

    for epoch in range(num_epochs):

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()

        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print(DATASET, 'Cycle:', cycle+1, 'Epoch:', epoch, '---', 'Val Acc: {:.2f} \t Best Acc: {:.2f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')

def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['cod'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            scores, _, _, _ = models['backbone'](inputs)
                
            if SAMPLING == 'TOD':
                scores = F.softmax(scores, dim=1)
                cod_scores, _, _, _ = models['cod'](inputs)
                cod_scores = F.softmax(cod_scores, dim=1)
                pred_loss = (scores - cod_scores).pow(2).sum(1) / 2   

            uncertainty = torch.cat((uncertainty, pred_loss), dim=0)
    return uncertainty.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-Supervised Active Learning')
    parser.add_argument('--config', default='cifar10', type=str, help='dataset config path')
    parser.add_argument('--sampling', default='TOD', type=str, help='data sampling method', choices=['RANDOM', 'TOD'])
    parser.add_argument('--auxiliary', default='TOD', type=str, help='auxiliary training loss', choices=['NONE', 'TOD'])
    args = parser.parse_args()

    config = importlib.import_module('config.'+args.config)
    config.SAMPLING = args.sampling # Random | TOD
    config.AUXILIARY = args.auxiliary # NONE | TOD
    to_import = [name for name in dir(config) if not name.startswith('_')]
    globals().update({name: getattr(config, name) for name in to_import})
        
    # Data
    if DATASET == 'cifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        data_train = CIFAR10(DATA_DIR, train=True, download=False, transform=train_transform)
        data_unlabeled = CIFAR10(DATA_DIR, train=True, download=False, transform=test_transform)
        data_test = CIFAR10(DATA_DIR, train=False, download=False, transform=test_transform)
        
    elif DATASET == 'cifar100':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        data_train = CIFAR100(DATA_DIR, train=True, download=False, transform=train_transform)
        data_unlabeled = CIFAR100(DATA_DIR, train=True, download=False, transform=test_transform)
        data_test = CIFAR100(DATA_DIR, train=False, download=False, transform=test_transform)
        
    elif DATASET == 'svhn':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])

        data_train = SVHN(root=DATA_DIR, split='train', transform=train_transform, download=False)
        data_unlabeled = SVHN(root=DATA_DIR, split='train', transform=train_transform, download=False)
        data_test = SVHN(root=DATA_DIR, split='test', transform=test_transform, download=False)
        
    elif DATASET == 'caltech101':
        train_transform = T.Compose(
            [
             T.Resize((256, 256)),
             T.CenterCrop(224),
             T.RandomHorizontalFlip(),
             T.ToTensor(),
             T.Normalize([0.5020, 0.5020, 0.5020], [1.0, 1.0, 1.0])
            ])
        test_transform = T.Compose(
            [
             T.Resize((256, 256)),
             T.CenterCrop(224),
             T.ToTensor(),
             T.Normalize([0.5020, 0.5020, 0.5020], [1.0, 1.0, 1.0])
            ])

        ratio = [0.9, 0.1]
        dataset = ImageFolder(DATA_DIR)
        character = [[] for i in range(len(dataset.classes))]
        for x, y in dataset.imgs:     #.samples:    use .imgs for torchvision 0.2.0, and .samples for 0.4.2
            #if y != 0:     # remove the Background class, will incur cuda issue
            character[y].append(x)
        del character[0]       # use this to remove the Background class

        train_inputs, val_inputs, test_inputs = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        for i, data in enumerate(character):
            num_sample_train = int(len(data) * ratio[0])
            num_sample_val = len(data) - num_sample_train 
            index = list(range(len(data)))
            random.shuffle(index)
            train_index = index[:num_sample_train]
            val_index = index[num_sample_train:num_sample_train+num_sample_val]
            for x in train_index:
                train_inputs.append(str(data[x]))
                train_labels.append(i)
            for x in val_index:
                val_inputs.append(str(data[x]))
                val_labels.append(i)
                
        from utils.custom_dataset import MyDataset
        data_train = MyDataset(train_inputs, train_labels, transform=train_transform)
        data_unlabeled = MyDataset(train_inputs, train_labels, transform=train_transform)
        data_test = MyDataset(val_inputs, val_labels, transform=test_transform)
  

    for trial in range(TRIALS):
        global iters
        iters = 0
        
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:START]
        unlabeled_set = indices[START:]

        train_loader = DataLoader(data_train, batch_size=BATCH,     # BATCH
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        extra_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(unlabeled_set),
                                  pin_memory=True)

        dataloaders = {'train': train_loader, 'test': test_loader, 'extra': extra_loader}

        # Model
        if DATASET == 'caltech101':
            import models.imagenet_resnet as in_resnet
            
            backbone_net = in_resnet.ResNet18(num_classes=CLASS).cuda()
            pretrained_dict = torch.load('./resnet18-5c106cde.pth')
            model_dict = backbone_net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            backbone_net.load_state_dict(model_dict)

            cod_model = in_resnet.ResNet18(num_classes=CLASS).cuda()
            ema_model = in_resnet.ResNet18(num_classes=CLASS).cuda()
        else:
            backbone_net = resnet.ResNet18(num_classes=CLASS).cuda()
            cod_model = resnet.ResNet18(num_classes=CLASS).cuda()    
            ema_model = resnet.ResNet18(num_classes=CLASS).cuda()
      
        for param in cod_model.parameters():
            param.detach_()
        for param in ema_model.parameters():
            param.detach_()

        models = {'backbone': backbone_net, 'ema': ema_model, 'cod': cod_model}
        torch.backends.cudnn.benchmark = True

        # Active learning cycles
        for cycle in range(CYCLES):

            if cycle > 0:
                checkpoint = torch.load('./weights/{}_auxiliary_{}_sampling_{}_trial{}_cycle{}.pth'.format(DATASET, AUXILIARY, SAMPLING, trial+1, cycle))
                models['cod'].load_state_dict(checkpoint['state_dict_backbone'])

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone}  
            schedulers = {'backbone': sched_backbone}  

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, cycle)
            acc = test(models, dataloaders, mode='test')
            print('{} auxiliary:{} sampling:{} Trial:{}/{} || Cycle:{}/{} || Label set size:{} ||  Test acc:{:.2f}'.format(DATASET, AUXILIARY, SAMPLING, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc), flush=True)
            
            # Active sampling
            random.shuffle(unlabeled_set)
            if SAMPLING == 'RANDOM':
                subset = unlabeled_set[:ADDENDUM]
                labeled_set += subset  
                unlabeled_set = unlabeled_set[ADDENDUM:]
            else:
                subset = unlabeled_set[:SUBSET]
                
                # Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(subset), 
                                              pin_memory=True)

                # Measure uncertainty of each data points in the subset
                uncertainty = get_uncertainty(models, unlabeled_loader)

                # Index in ascending order
                arg = np.argsort(uncertainty)

                # Update the labeled dataset and the unlabeled dataset, respectively
                if cycle > 0:
                    labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())  
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
                else:
                    labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())  
                    unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,   
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            dataloaders['extra'] = DataLoader(data_train, batch_size=BATCH,  
                                              sampler=SubsetRandomSampler(unlabeled_set),
                                              pin_memory=True)

            if not os.path.exists('weights'):
                os.makedirs('weights')
            torch.save({
                    'cycle': cycle + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                },
                './weights/{}_auxiliary_{}_sampling_{}_trial{}_cycle{}.pth'.format(DATASET, AUXILIARY, SAMPLING, trial+1, cycle+1))

