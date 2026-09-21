import torch
import torch.nn as nn

from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import GradScaler

import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import tarfile
import io

import lira_attack
from resnet import resnet18


BATCH_SIZE = 128
LR = 0.1
WEIGHT_DECAY=0.0001
MOMENTUM=0.9
NUM_WORKERS=2
NUM_EPOCHS=50

IN_MU = None
OUT_MU = None
IN_STD = None
OUT_STD = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_cifar10_datasets(root_path):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(root_path, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root_path, train=False, transform=test_transform, download=True)

    return train, test

def train_model(m, train_subset=range(50000), num_epochs=NUM_EPOCHS, data_dir='/net/scratch/crc/datasets/cifar10'):
    train, _ = load_cifar10_datasets(data_dir)
    trainset = torch.utils.data.Subset(train, train_subset)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    # test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    opt = SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for epoch in (pbar := tqdm(range(num_epochs))):
        for i, (x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            for param in m.parameters():
                param.grad = None

            preds = m(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()

def get_cifar10_images(data_dir='/net/scratch/crc/datasets/cifar10'):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train, batch_size=50000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=10000, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    for i,l in train_loader:
        X = i.to(device)
        Y = l.to(device)

    for i,l in test_loader:
        test_X = i.to(device)
        test_Y = l.to(device)

    return X, Y, test_X, test_Y


def load_cifar10(root_path, shuffle=True, batch_size=BATCH_SIZE, half_train=False):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train = datasets.CIFAR10(root_path, train=True, transform=transform, download=True)
    test = datasets.CIFAR10(root_path, train=False, transform=transform, download=True)
    if half_train:
        subset = np.random.choice(range(50000), size=25000, replace=False)
        train = torch.utils.data.Subset(train, subset)
    else:
        subset = list(range(50000))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader, subset

def get_model_ind(save_dir):
    inds = [int(x.stem.split('_')[-2]) for x in save_dir.glob('*.pt')]
    if len(inds) == 0:
        return 1
    return max(inds) + 1

def get_logits(m, loader):
    logits = []
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            logits.append(m(X))
    return torch.cat(logits)

def fill_lira_params(X, Y):
    logits = torch.load('./data/logits_50epochs_updated.pt', map_location=device)
    flipped_logits = torch.load('./data/flipped_logits_50epochs_updated.pt', map_location=device)
    subsets = {}
    tar: tarfile.TarFile = tarfile.open('./data/subsets.tar.gz', 'r:gz')
    for ind in logits:
        # subset = np.load(f'./data/subsets/resnet18_50epochs_{ind}_updated.npy')
        f = tar.extractfile(f'./data/subsets/resnet18_50epochs_{ind}_updated.npy')
        if f is not None:
            subset = np.load(io.BytesIO(f.read()))
        else:
            raise FileNotFoundError('Missing subsets data')
        subsets[ind] = subset

    global IN_MU, IN_STD, OUT_MU, OUT_STD
    IN_MU, IN_STD, OUT_MU, OUT_STD = lira_attack.get_params([], subsets, X, Y, logits=logits, flipped_logits=flipped_logits)

def pairwise_cosine_sim(a, b, eps=1e-8):
    a_norm = a / torch.clamp(a.norm(dim=1)[:,None], eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:,None], eps)
    return a_norm @ b_norm.T

def knn_sv(train_logits, train_labels, test_logits, test_labels, k=10):
    N, N_test = len(train_labels), len(test_labels)
    vals = torch.zeros((N_test, N)).to(device)
    sims_matrix = pairwise_cosine_sim(train_logits, test_logits)
    sorted_matrix = torch.sort(sims_matrix.T, descending=True).indices.to(device)
    vals[torch.arange(len(vals)),sorted_matrix[:,0].cpu()] = (test_labels == train_labels[sorted_matrix[:,0]])/N

    for i in tqdm(range(1,N)):
        prev_vals = vals[torch.arange(len(vals)),sorted_matrix[:,i-1]]
        d1 = (test_labels == train_labels[sorted_matrix[:,i]]).float()
        d2 = (test_labels == train_labels[sorted_matrix[:,i-1]]).float()
        diff = (d1 - d2) / k
        vals[torch.arange(len(vals)), sorted_matrix[:,i].cpu()] = prev_vals + (diff * min(k,i) / i)

    return torch.sum(vals, dim=0) / N_test

def get_unlearn_inds(args, num_unlearned, train_inds, loss, m, X, Y, test_X, test_Y, trainloader, testloader):
    unlearn_inds = None
    if args.unlearn_type == 'random':
        ind_list = list(train_inds)
        perm = torch.randperm(len(ind_list))
        unlearn_inds = [ind_list[i] for i in perm][:num_unlearned]
    if args.unlearn_type == 'loss':
        # mem_inds = torch.tensor([x for x in train_inds if loss[x] == 0]).to(device)
        # perm = torch.randperm(len(mem_inds))
        # unlearn_inds = [x.item() for x in mem_inds[perm][:num_unlearned]]
        ordering = torch.argsort(loss)
        unlearn_inds = [x.item() for x in ordering if x.item() in train_inds][:num_unlearned]
    if args.unlearn_type == 'lira':
        y, pred, score = lira_attack.lira(m, train_inds, IN_MU, IN_STD, OUT_MU, OUT_STD, trainloader)
        in_point_scores = {ind:score[ind] for ind in train_inds}
        unlearn_inds = sorted(in_point_scores, key=in_point_scores.get)[-num_unlearned:]
    if args.unlearn_type == 'shapley' or args.unlearn_type == 'min_shapley':
        train_ind_list = list(train_inds)
        index_mapping = {i:ind for i,ind in enumerate(train_ind_list)}
        # new_X =  X[train_ind_list]
        new_Y = Y[train_ind_list]
        with torch.no_grad():
            logits = get_logits(m, trainloader)
            logits = logits[train_ind_list]
            test_logits = get_logits(m, testloader)
        scores = knn_sv(logits, new_Y, test_logits, test_Y)
        ordering = torch.argsort(scores, descending=True)
        if args.unlearn_type == 'shapley':
            unlearn_inds = [index_mapping[x.item()] for x in ordering][:num_unlearned]
        else:
            unlearn_inds = [index_mapping[x.item()] for x in ordering][-num_unlearned:]

    return unlearn_inds

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ind', type=int)
    parser.add_argument('--unlearn_type', choices=['random', 'loss', 'lira', 'shapley', 'min_shapley'], default='loss')
    parser.add_argument('--num_unlearned', type=int, default=500, help='Number of points to unlearn on each iteration')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=10)
    parser.add_argument('--non_adaptive', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--half_train', action='store_true', help='Start training models on half of dataset')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    assert (not args.half_train) or (args.suffix == 'half_train')

    trainloader, testloader, subset = load_cifar10('/net/scratch/crc/datasets/cifar10', shuffle=False, half_train=args.half_train)
    full_trainloader, _, _ = load_cifar10('/net/scratch/crc/datasets/cifar10', shuffle=False)
    X, Y, test_X, test_Y = get_cifar10_images()

    if args.unlearn_type == 'lira':
        fill_lira_params(X, Y)

    model_dir = Path(f'/net/scratch/crc/memorization/experiment1/{args.unlearn_type}/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(f'/net/scratch/crc/memorization/experiment1/{args.unlearn_type}/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    train_inds = set(subset)
    unlearn_inds = list(set(range(50000)) - train_inds)
    model_ind = args.ind
    start_iter = args.start_iter
    save_data = {}

    non_adaptive_str = ''

    if start_iter > 0:
        try:
            save_data = torch.load(data_dir / f'resnet18_{args.num_unlearned}unlearn_{model_ind}_{args.suffix}.pt', map_location='cpu')
            unlearn_inds = [item for ls in [x['unlearn_pts'] for i,x in save_data.items() if i <= start_iter] for item in ls]
        except FileNotFoundError:
            print(f'Error: Cannot find save file for model {args.ind}')
            exit()

    init_m = None
    init_loss = None
    for i in range(start_iter, args.num_iters):
        if args.non_adaptive:
            num_unlearned = args.num_unlearned * (i+1)
        else:
            num_unlearned = args.num_unlearned

        print(f'Iteration {i}')
        m = resnet18(num_classes=10).to(device)
        train_model(m, train_subset = list(train_inds), num_epochs=args.num_epochs)
        preds = get_logits(m, full_trainloader)
        test_preds = get_logits(m, testloader)
        accuracy = ((torch.argmax(test_preds, axis=1) == test_Y).sum() / len(test_Y)).item()
        print(f'Accuracy: {accuracy:0.3f}')
        print()
        loss = torch.nn.CrossEntropyLoss(reduce=False)(preds, Y)
        if init_m is None:
            init_m = m
            init_loss = loss

        model_data = {
            'acc': accuracy,
            'unlearn_pts': unlearn_inds,
        }
        
        save_data[i] = model_data

        if args.non_adaptive:
            train_inds = set(subset)
            unlearn_inds = get_unlearn_inds(args, num_unlearned, train_inds, init_loss, init_m, X, Y, test_X, test_Y, full_trainloader, testloader)
        else:
            unlearn_inds = get_unlearn_inds(args, num_unlearned, train_inds, loss, m, X, Y, test_X, test_Y, full_trainloader, testloader)


        train_inds = train_inds - {x for x in unlearn_inds}

        if args.non_adaptive:
            non_adaptive_str = '_nonadaptive'
    
        torch.save(m, model_dir/f'resnet18_{args.num_unlearned}unlearn_{args.num_epochs}epochs_{model_ind}_{i}{non_adaptive_str}_{args.suffix}.pt')
        
    torch.save(save_data, data_dir/f'resnet18_{args.num_unlearned}unlearn_{args.num_epochs}epochs_{model_ind}{non_adaptive_str}_{args.suffix}.pt')